# SPDX-License-Identifier: Apache-2.0

"""Real draft/filter/verification graphs before native worker qualification."""

import os

import pytest
import torch

from ignition.components.draft_sampling import PreparedDraftDistribution
from ignition.components.parallel_draft import prepare_sampling_scratch
from ignition.components.sampling import DeviceRequestState, RequestAdmission
from ignition.components.speculation import PreparedSpeculativeDistributions, PreparedStochasticVerification, SpeculativeRngState
from ignition.preparation.capture import capture_forwards
from ignition.preparation.execution import GraphSignature
from ignition.preparation.external_session import ExternalPreparationSession
from ignition.preparation.scratch import GraphBufferPool, ScratchArena


pytestmark = pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1' or not torch.cuda.is_available(),
    reason='requires FLARE_RUN_CUDA_TESTS=1 and an assigned PPU/CUDA device')

VOCAB = 248320


class SamplingForward:
    def __init__(self, batch, width, state, rng, pool):
        self.batch, self.width, self.state = batch, width, state
        self.logits = torch.zeros(batch, width, VOCAB, device='cuda:0')
        self.proposal_logits = torch.zeros(batch, 1, VOCAB, device='cuda:0')
        self.slots = torch.arange(batch, device='cuda:0', dtype=torch.int32)
        self.epochs = torch.full((batch,), 7, device='cuda:0', dtype=torch.int64)
        self.live = torch.full((1,), batch, device='cuda:0', dtype=torch.int32)
        self.ids = torch.zeros(batch, width - 1, device='cuda:0', dtype=torch.int64)
        self.probabilities = torch.empty(batch, width - 1, VOCAB, device='cuda:0')
        self.valid = torch.zeros(batch, width - 1, device='cuda:0', dtype=torch.bool)
        self.seed = torch.arange(batch, device='cuda:0', dtype=torch.int64) + 81
        self.offset, self.sampled = torch.zeros_like(self.seed), torch.zeros_like(self.seed)
        work = prepare_sampling_scratch(batch, width, VOCAB, 'cuda:0', buffers=pool.cursor())
        self.proposals = [PreparedDraftDistribution(self.proposal_logits, state, self.slots,
            self.epochs, self.live, self.ids, prefix_offset=index, seed=self.seed, offset=self.offset,
            sampled=self.sampled, workspace=work['proposal']) for index in range(width - 1)]
        for proposal in self.proposals:
            proposal.prepare()
        self.target = PreparedSpeculativeDistributions(self.logits, state, self.slots, self.epochs,
            self.live, self.ids, workspace=work['target'])
        self.target.prepare()
        self.verify = PreparedStochasticVerification(self.probabilities, self.ids, self.target.probabilities,
            state, self.slots, self.epochs, self.live, rng=rng, draft_valid=self.valid,
            target_valid=self.target.valid, workspace=work['verify'])
        self.verify.prepare()

    def __call__(self):
        self.ids.zero_()
        for index, proposal in enumerate(self.proposals):
            self.offset.fill_(index * 4)
            proposal()
            self.ids[:, index].copy_(self.sampled)
            self.probabilities[:, index].copy_(proposal.probabilities[:, 0])
            self.valid[:, index].copy_(proposal.valid[:, 0])
        self.target()
        self.verify()

    def stage(self, slots, *, cutoff=None, live=None, stale=False):
        self.slots.copy_(torch.tensor(slots, dtype=torch.int32))
        self.seed.copy_(torch.tensor([81 + slot for slot in slots]))
        self.epochs.fill_(7)
        if stale:
            self.epochs[-1] = 99
        self.live.fill_(self.batch if live is None else live)
        self.logits.fill_(-torch.inf); self.proposal_logits.fill_(-torch.inf)
        for row, slot in enumerate(slots):
            if cutoff is None:
                scores = torch.linspace(-2., 3., 64, device=self.logits.device) + slot * .01
                self.logits[row, :, :64].copy_(scores)
                self.proposal_logits[row, 0, :64].copy_(scores.flip(0))
            else:
                self.proposal_logits[row, 0, 3 + slot] = 0
                self.logits[row, :, 3 + slot] = 0
                self.logits[row, cutoff].fill_(-torch.inf)
                self.logits[row, cutoff, 19 + slot] = 0
        if live is not None:
            self.logits[live:].fill_(torch.nan)
            self.proposal_logits[live:].fill_(torch.nan)


def filtered_reference(logits, prefix):
    # Only the first 64 vocabulary entries have finite scores in this fixture.
    counts = torch.bincount(torch.tensor([1, 2, *prefix]), minlength=64).double()
    scores = logits.double()
    scores = torch.where(counts > 0, torch.where(scores > 0, scores / 1.1, scores * 1.1), scores)
    scores = (scores - (counts > 0) * .2 - counts * .1) / .8
    scores[scores < scores.topk(13).values[-1]] = -torch.inf
    probabilities = scores.softmax(-1)
    ordered = probabilities.sort(descending=True).values
    boundary = ordered[(ordered.cumsum(-1) < .83).sum()]
    probabilities[probabilities < boundary] = 0
    probabilities /= probabilities.sum()
    probabilities[probabilities < probabilities.max() * .07] = 0
    return (probabilities / probabilities.sum()).float()


def check_filtered(forward):
    ids = forward.ids.cpu()
    target = forward.target.probabilities.cpu()
    draft = forward.probabilities.cpu()
    assert not target[..., 64:].count_nonzero() and not draft[..., 64:].count_nonzero()
    for row in range(forward.batch):
        for column in range(forward.width):
            expected = filtered_reference(forward.logits[row, column, :64].cpu(), ids[row, :column].tolist())
            torch.testing.assert_close(target[row, column, :64], expected, rtol=2e-5, atol=1e-8)
            if column < forward.width - 1:
                expected = filtered_reference(forward.proposal_logits[row, 0, :64].cpu(), ids[row, :column].tolist())
                torch.testing.assert_close(draft[row, column, :64], expected, rtol=2e-5, atol=1e-8)


@pytest.mark.parametrize('compilation', ['none', 'catalog'])
@torch.no_grad()
def test_draft_target_verification_share_a_graph_and_switch_shapes(tmp_path, compilation):
    stream = torch.cuda.Stream(device=0)
    graphs, compiled = {}, {}
    with torch.cuda.stream(stream), ExternalPreparationSession(device='cuda:0', checkpoint='sampling-components',
            namespace={'qualification': 'external-sampling-v1'}, compile_workers=1) as session:
        state = DeviceRequestState.allocate('cuda:0', vocab_size=VOCAB, capacity=4)
        rng = SpeculativeRngState.allocate('cuda:0', capacity=4)
        def admit(filtered=False):
            for slot in range(4):
                state.admit(slot, RequestAdmission(generation=7, seed=81 + slot, max_new_tokens=100,
                    temperature=.8 if filtered else 1., top_k=13 if filtered else 0,
                    top_p=.83 if filtered else 1., min_p=.07 if filtered else 0.,
                    repetition_penalty=1.1 if filtered else 1., presence_penalty=.2 if filtered else 0.,
                    frequency_penalty=.1 if filtered else 0., eos_token_ids=()), [1, 2])
            rng.generation.fill_(7)
            rng.target_offset.copy_(torch.arange(4, device='cuda:0') * 8)
        admit()
        pool = GraphBufferPool('cuda:0', chunk_bytes=2 << 20)
        forwards = {f'{batch}x{width}': SamplingForward(batch, width, state, rng, pool)
                    for batch, width in ((1, 1), (1, 7), (4, 7))}
        session.register_scratch(ScratchArena('cuda:0', overlap=False))
        session.tune()
        try:
            graphs, compiled = capture_forwards(forwards,
                {name: GraphSignature('speculative', f.batch, f.width) for name, f in forwards.items()},
                session=session, device='cuda:0', stream=stream, pool=torch.cuda.graph_pool_handle(),
                cache_root=tmp_path, compile_decoder=compilation == 'catalog', retain_graphs=True, report_catalog=False)
            session.freeze()
            def replay(name):
                for chunks in pool.chunks.values():
                    for tensor in chunks:
                        tensor.fill_(float('nan') if tensor.is_floating_point() else 255)
                before = {name: value.cpu().clone() for name, value in vars(state).items()}
                allocated = torch.cuda.memory_stats()['allocation.all.allocated']
                graphs[name].replay(); stream.synchronize()
                assert torch.cuda.memory_stats()['allocation.all.allocated'] == allocated
                for field, value in before.items():
                    torch.testing.assert_close(getattr(state, field).cpu(), value, rtol=0, atol=0)

            for name in ('4x7', '1x1', '1x7', '4x7'):
                f = forwards[name]
                admit()
                for cutoff in range(f.width):
                    slots = [3, 0, 2, 1][:f.batch]
                    f.stage(slots, cutoff=cutoff)
                    replay(name)
                    out = f.verify.output
                    assert out.valid.all() and (out.accepted_drafts == cutoff).all()
                    assert (out.proposed_count == cutoff + 1).all()
                    assert (out.rng_increment == f.verify.draw_reservation).all()
                    for row, slot in enumerate(slots):
                        assert out.token_ids[row, :cutoff + 1].tolist() == [3 + slot] * cutoff + [19 + slot]
                for live, stale in ((0, False), (f.batch, True), (max(0, f.batch - 1), False)):
                    f.stage(slots, cutoff=0, live=live, stale=stale)
                    replay(name)
                    valid = torch.arange(f.batch) < live
                    if stale:
                        valid[-1] = False
                    torch.testing.assert_close(out.valid.cpu(), valid, rtol=0, atol=0)
                    assert (out.token_ids.cpu()[~valid] == -1).all()
                    assert not out.proposed_count.cpu()[~valid].count_nonzero()
                    assert not out.rng_increment.cpu()[~valid].count_nonzero()

            # A request's seed/offset and probabilities must determine its draw
            # independently of packing order or the other requests in the batch.
            admit(filtered=True)
            baseline = {}
            for name, slots in (('4x7', [0, 1, 2, 3]), ('1x7', [2]), ('4x7', [2, 0, 3, 1])):
                f = forwards[name]
                f.stage(slots)
                replay(name)
                check_filtered(f)
                out = f.verify.output
                assert out.valid.all()
                for row, slot in enumerate(slots):
                    count = int(out.proposed_count[row])
                    result = (f.ids[row].tolist(), int(out.accepted_drafts[row]), out.token_ids[row, :count].tolist())
                    if slot in baseline:
                        assert result == baseline[slot]
                    else:
                        baseline[slot] = result
        finally:
            stream.synchronize()
            for graph in graphs.values():
                graph.reset()
            compiled.clear()
