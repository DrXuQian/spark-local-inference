# SPDX-License-Identifier: Apache-2.0

"""Qualify actlizeLA QSA with Flare BF16 page storage and native package replay."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch

from qualify.actlize_export import export_fixture, save
from qualify.qsa_reference import compress_history, encode_index, paged_attention, select_positions


class Forward:
    def __init__(self, operation, output, observed):
        self.operation, self.output, self.observed = operation, output, observed

    def __call__(self):
        self.operation()
        self.observed.copy_(self.output)


class Fixture:
    def __init__(self, library, rows=7, *, device='cuda:0'):
        from actlize_la.serving import qsa_workspace_specs
        from actlize_la.serving.flare import bind_qsa
        from ignition.recipes.qwen38.attention import QSAState, QSAMetadata, RotaryTables

        device = torch.device(device)
        self.rows = rows
        self.requests = min(rows, 2)
        self.context = max(2176, ((2051 + rows + 63) // 64) * 64)
        self.pages = self.context // 64
        self.state = QSAState.allocate(pages=self.pages, slots=3, device=device,
            draft_tokens=7, kv_dtype=torch.bfloat16)
        self.metadata = QSAMetadata.allocate(tokens=self.rows, batch=self.requests, context=self.context, device=device)
        config = SimpleNamespace(head_dim=256, rope_parameters=dict(partial_rotary_factor=.25,
            mrope_section=[11, 11, 10], rope_theta=10000000))
        self.rotary = RotaryTables.allocate(config, self.context + 32, device)
        self.co, self.si, self.axes = self.rotary.cos.cpu(), self.rotary.sin.cpu(), self.rotary.axes.cpu()
        self.live = torch.zeros(1, dtype=torch.int32, device=device)
        self.inputs = {name: torch.zeros(shape, dtype=torch.float32 if name.endswith('_weight') else torch.bfloat16, device=device)
            for name, shape in dict(query=(self.rows, 24, 256), key=(self.rows, 2, 256),
                value=(self.rows, 2, 256), index_query=(self.rows, 4, 128),
                raw_index_key=(self.rows, 128), index_q_norm_weight=(128,),
                index_k_norm_weight=(128,)).items()}
        self.work = {name: torch.zeros(shape, dtype=dtype, device=device)
            for name, (shape, dtype) in qsa_workspace_specs(self.rows, self.requests).items()}
        self.operation = bind_qsa(library, **self.inputs, rotary=self.rotary, state=self.state,
            metadata=self.metadata, live_rows=self.live, workspace=self.work)
        self.observed = torch.zeros_like(self.work['output'])
        self.forward = Forward(self.operation, self.work['output'], self.observed)
        self.buffers = dict(self.inputs, observed=self.observed, live=self.live,
            **{'metadata.' + name: getattr(self.metadata, name) for name in self.metadata.__dataclass_fields__},
            **{'cache.' + name: getattr(self.state, name) for name in self.state.__dataclass_fields__},
            **{'work.' + name: value for name, value in self.work.items()})
        self.perm, self.compressed_perm = torch.arange(self.pages).roll(1), torch.arange(self.pages).roll(5)

    def rope(self, value, positions):
        columns = torch.arange(32)
        co = self.co[positions[..., self.axes], columns]
        si = self.si[positions[..., self.axes], columns]
        while co.ndim < value.ndim:
            co, si = co.unsqueeze(-2), si.unsqueeze(-2)
        value = value.float()
        lo, hi = value[..., :32], value[..., 32:64]
        return torch.cat((lo * co - hi * si, hi * co + lo * si, value[..., 64:]), -1)

    def compressed(self, end):
        return compress_history(self.data['raw'][:end], self.inputs['index_k_norm_weight'].cpu(),
                                self.positions[:end], self.rope)

    def seed(self, context, seed):
        torch.manual_seed(seed)
        self.data = {name: (torch.randn(self.context, *shape) * .2).bfloat16()
            for name, shape in dict(query=(24, 256), key=(2, 256), value=(2, 256),
                index_query=(4, 128), raw=(128,)).items()}
        self.positions = torch.arange(self.context)[:, None].expand(-1, 3).clone()
        self.positions[:, 1] += 1; self.positions[:, 2] += 2
        if context >= 2048:
            self.data['raw'].zero_(); self.data['index_query'].zero_()  # exact top-k ties
        s, m = self.state, self.metadata
        for name in s.__dataclass_fields__:
            value = getattr(s, name)
            value.fill_(-1 if value.dtype == torch.int64 else 0)
        for name in ('index_q_norm_weight', 'index_k_norm_weight'):
            self.inputs[name].copy_(torch.randn(128) * .02)
        m.main_block_table.fill_(-1); m.compressed_block_table.fill_(-1)
        m.main_block_table[0].copy_(self.perm); m.compressed_block_table[0].copy_(self.compressed_perm)
        for t in range(context):
            page, offset = int(self.perm[t // 64]), t % 64
            s.main_k_cache[page, offset].copy_(self.data['key'][t])
            s.main_v_cache[page, offset].copy_(self.data['value'][t])
        for group, value in enumerate(self.compressed(context)):
            s.compressed_k_cache[self.compressed_perm[group // 16], group % 16].copy_(value)
        for t in range(max(0, context - 16), context):
            s.raw_k_ring[1, t % 16].copy_(self.data['raw'][t])
            s.raw_logical_positions[1, t % 16] = t
            s.raw_rope_positions[1, t % 16].copy_(self.positions[t])
        s.raw_interval_start_positions[1] = context - 1

    def interval(self, first, width, accepted):
        s, m = self.state, self.metadata
        self.first, self.width = first, width
        for name in ('query', 'key', 'value', 'index_query'):
            self.inputs[name].zero_(); self.inputs[name][:width].copy_(self.data[name][first:first + width])
        self.inputs['raw_index_key'].zero_()
        self.inputs['raw_index_key'][:width].copy_(self.data['raw'][first:first + width])
        m.request_ids.fill_(-1); m.request_ids[:width].fill_(0)
        m.query_positions.fill_(-1); m.query_positions[:width].copy_(torch.arange(first, first + width))
        m.rope_positions.zero_(); m.rope_positions[:width].copy_(self.positions[first:first + width])
        m.raw_state_slot_ids.fill_(-1); m.raw_state_slot_ids[0] = 1
        m.slot_mapping.fill_(-1)
        p = torch.arange(first, first + width)
        m.slot_mapping[:width].copy_(self.perm[p // 64] * 64 + p % 64)
        m.query_start_loc.fill_(width); m.query_start_loc[0] = 0
        m.sequence_lengths.zero_(); m.sequence_lengths[0] = first + width
        m.num_accepted_tokens.zero_(); m.num_accepted_tokens[0] = accepted
        m.is_prefilling.fill_(width > 7); self.live.fill_(self.requests if width else 0)

    def check(self, before):
        s, m, first, width = self.state, self.metadata, self.first, self.width
        assert not self.work['errors'].any().item()
        if not width:
            for name, value in before.items():
                assert torch.equal(getattr(s, name).cpu(), value)
            assert not self.observed.count_nonzero().item()
            return
        encoded = encode_index(self.data['index_query'][first:first + width],
            self.inputs['index_q_norm_weight'].cpu(), self.positions[first:first + width], self.rope)
        compressed = self.compressed(first + width)
        selected = select_positions(encoded, compressed, torch.arange(first, first + width), first + width)
        assert torch.equal(self.work['selected'][:width].cpu(), selected)
        for t in range(first, first + width):
            page, offset = int(self.perm[t // 64]), t % 64
            before['main_k_cache'][page, offset].copy_(self.data['key'][t])
            before['main_v_cache'][page, offset].copy_(self.data['value'][t])
        for name in ('main_k_cache', 'main_v_cache'):
            assert torch.equal(getattr(s, name).cpu(), before[name])
        for group in range((first + width) // 4):
            actual = s.compressed_k_cache[self.compressed_perm[group // 16], group % 16].cpu()
            torch.testing.assert_close(actual, compressed[group], rtol=2**-6, atol=2e-3)
        expected = paged_attention(self.inputs['query'][:width].cpu(),
            before['main_k_cache'], before['main_v_cache'], m.main_block_table.cpu(),
            m.request_ids[:width].cpu(), selected, m.query_positions[:width].cpu())
        actual = self.observed[:width].cpu().float()
        relative = torch.linalg.vector_norm(actual - expected.float()) / torch.linalg.vector_norm(expected.float())
        assert relative < .006
        torch.testing.assert_close(actual, expected.float(), rtol=.01, atol=2e-5)
        assert not self.observed[width:].count_nonzero().item()
        for name in ('raw_k_ring', 'raw_logical_positions', 'raw_rope_positions', 'raw_interval_start_positions'):
            assert torch.equal(getattr(s, name)[0].cpu(), before[name][0])
            assert torch.equal(getattr(s, name)[2].cpu(), before[name][2])


def prepare(args):
    from actlize_la.serving import NativeLibrary
    from b12x.preparation import PreparationSession
    from ignition.preparation.capture import capture_forwards
    from ignition.preparation.execution import GraphSignature
    from ignition.preparation.module_images import ModuleImages
    from ignition.preparation.package_storage import PackageStorage

    torch.set_num_threads(1); torch.cuda.set_device(0)
    graphs = {}
    with ModuleImages() as images, PackageStorage() as storage:
        library = NativeLibrary(args.library, target='cuda_sm120')
        with PreparationSession(device='cuda:0', autotune=False, compile_workers=1, prime=False) as session:
            try:
                signatures = {'verify': GraphSignature('speculative', 2, 6)}
                common = dict(session=session, device=torch.device('cuda:0'), stream=torch.cuda.Stream(),
                    pool=storage.graph_pool.id, cache_root=args.output / 'compiled', retain_graphs=True)
                old = Fixture(library)
                graphs, compiled = capture_forwards({'verify': old.forward}, signatures, **common)
                graphs['verify'].reset()
                f = Fixture(library)
                assert f.operation.handle != old.operation.handle
                graphs, compiled = capture_forwards({'verify': f.forward}, signatures, **common)
                assert compiled['verify'].cache_hit
                del old
                session.freeze()
                scenarios = [(f'context-{n}', n, None) for n in (0, 1, 2, 3, 63, 64, 65, 2048, 2051)]
                scenarios += [(f'prefix-{n}', 63, n) for n in range(7)]
                scenarios += [('inactive', 63, None)]
                cases = []
                for index, (name, context, accepted) in enumerate(scenarios):
                    f.seed(context, 803 + index)
                    f.interval(context, 6, 1)
                    if accepted is not None:
                        graphs['verify'].replay(); torch.cuda.synchronize()
                        start = context + accepted
                        for field in ('query', 'key', 'value', 'raw', 'index_query'):
                            f.data[field][start:start + 6].mul_(-1)
                        f.interval(start, 6, accepted)
                    if name == 'inactive':
                        f.interval(context, 0, 1)
                    before = {key: getattr(f.state, key).cpu().clone() for key in f.state.__dataclass_fields__}
                    uploads = {key: save(args.output, f'{name}-in-{key}', value)
                        for key, value in f.buffers.items() if not key.startswith('work.') and key != 'observed'}
                    allocated = torch.cuda.memory_stats()['allocation.all.allocated']
                    graphs['verify'].replay(); torch.cuda.synchronize()
                    assert torch.cuda.memory_stats()['allocation.all.allocated'] == allocated
                    f.check(before)
                    expected = {key: save(args.output, f'{name}-out-{key}', value)
                        for key, value in f.buffers.items() if key.startswith('cache.') or
                            key in ('work.errors', 'work.selected', 'work.selected_count', 'observed')}
                    cases.append(dict(name=name, graph='verify', uploads=uploads, expected=expected))
                    print(f'{name}: Flare reference and compiled QSA replay passed', flush=True)
                export_fixture(args, images, storage, graphs, signatures, f.buffers,
                    (f, compiled, library), cases, 'actlize-qsa-export-v1')
            finally:
                torch.cuda.synchronize()
                for graph in graphs.values():
                    graph.reset()


if __name__ == '__main__':
    from qualify.actlize_qsa_export import prepare

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    prepare(args)
