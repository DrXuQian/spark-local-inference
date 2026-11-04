# SPDX-License-Identifier: Apache-2.0

"""Qualify actlizeLA prepared calls across Flare compilation and package replay."""

from pathlib import Path
import argparse
import json

import torch


class Forward:
    def __init__(self, operation, commit, norm, output, observed):
        self.operation, self.commit = operation, commit
        self.norm, self.output, self.observed = norm, output, observed

    def __call__(self):
        self.output.zero_()
        self.operation()
        self.observed.zero_()
        self.norm()
        self.commit()


def fixture(library):
    from actlize_la.serving import gdn_workspace_specs
    from actlize_la.serving.flare import bind_gdn, bind_gdn_commit
    from ignition.components.gated_norm import PreparedGatedRMSNorm
    from ignition.components.recurrent import DeltaMetadata

    torch.manual_seed(718)
    device = torch.device('cuda:0')
    rows, requests, width, slots = 12, 2, 8, 5
    mixed = torch.randn(rows, 10240, device=device, dtype=torch.bfloat16)
    gates = torch.randn(rows, 96, device=device, dtype=torch.bfloat16)
    backing = torch.zeros(slots, 48 * 16384 + 64, device=device)
    state = backing[:, :48 * 16384].view(slots, 48, 128, 128)
    metadata = DeltaMetadata.allocate(requests, device, verification_columns=8)
    output = torch.zeros(rows, 48, 128, device=device, dtype=torch.bfloat16)
    observed = torch.zeros_like(output)
    workspace = {name: torch.zeros(shape, dtype=dtype, device=device)
        for name, (shape, dtype) in gdn_workspace_specs(requests, width, mode='verify', target=library.target).items()}
    counts = torch.zeros(requests, dtype=torch.int32, device=device)
    errors = torch.zeros_like(counts)
    a_log = torch.linspace(-3, -1, 48, device=device)
    bias = torch.linspace(-.2, .1, 48, device=device)
    z = torch.randn_like(output)
    weight = torch.ones(128, dtype=torch.bfloat16, device=device)
    operation = bind_gdn(library, mixed_qkv=mixed, a=gates[:, 48:], b=gates[:, :48],
        A_log=a_log, dt_bias=bias,
        state=state, metadata=metadata, output=output, workspace=workspace, mode='verify', width=width)
    commit = bind_gdn_commit(library, state=state, metadata=metadata, records=workspace,
        consumed_prefix=counts, errors=errors)
    norm = PreparedGatedRMSNorm(output, z, weight, observed, metadata.num_tokens,
        eps=1e-6, round_normalized=True)
    forward = Forward(operation, commit, norm, output, observed)
    buffers = dict(mixed=mixed, gates=gates, state=backing, output=output, observed=observed,
        counts=counts, commit_errors=errors, a_log=a_log, bias=bias, z=z, norm_weight=weight,
        **{'metadata.' + name: getattr(metadata, name) for name in metadata.__dataclass_fields__},
        **{'work.' + name: value for name, value in workspace.items()})
    return forward, buffers, state, metadata


def save(root, name, tensor):
    path = name + '.bin'
    (root / path).write_bytes(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return path


def reference_binding(buffers, metadata):
    from tests.recipes.test_qwen38_gdn_recovery_cuda import _reference

    device = buffers['mixed'].device
    tensors = dict(mixed_qkv=buffers['mixed'], a=buffers['gates'][:, 48:],
        b=buffers['gates'][:, :48], A_log=buffers['a_log'], dt_bias=buffers['bias'],
        recurrent_state=torch.zeros(17, 48, 128, 128, device=device), z=buffers['z'],
        norm_weight=buffers['norm_weight'], query_start_loc=metadata.query_start_loc,
        num_accepted_tokens=metadata.accepted_tokens,
        state_indices=torch.zeros(2, 8, dtype=torch.int64, device=device),
        num_seqs=metadata.num_seqs, num_tokens=metadata.num_tokens,
        output=torch.zeros_like(buffers['observed']))
    return tensors, _reference(tensors, 8)


def check_reference(buffers, initial, state, lengths, slots, counts, live, canonical):
    """Use Flare's unchanged per-token checkpointing reference and bounds."""
    from b12x.sequence import gdn_decode

    device = state.device
    tensors, binding = canonical
    reference, indices = tensors['recurrent_state'], tensors['state_indices']
    reference.zero_(); indices.zero_()
    for row, slot in enumerate(slots):
        if row < live and slot > 0 and lengths[row]:
            base = 1 + 8 * row
            indices[row] = torch.arange(base, base + 8, device=device)
            reference[base].copy_(initial[slot])
    wanted = tensors['output']; wanted.zero_()
    gdn_decode.run(binding, eps=1e-6)
    torch.cuda.synchronize()
    actual = buffers['observed']
    torch.testing.assert_close(actual, wanted, rtol=2**-6, atol=2e-2)
    assert (actual == wanted).float().mean() > .99
    expected = initial.clone()
    for row, (slot, count) in enumerate(zip(slots, counts, strict=True)):
        if row < live and slot > 0 and count:
            expected[slot].copy_(reference[1 + 8 * row + count - 1])
    torch.testing.assert_close(state, expected, rtol=1e-5, atol=1e-6)
    untouched = set(range(state.shape[0])) - {slot for row, (slot, count) in
        enumerate(zip(slots, counts, strict=True)) if row < live and slot > 0 and count}
    for slot in untouched:
        assert torch.equal(state[slot], initial[slot])


def qualify(args, images, storage):
    from actlize_la.serving import NativeLibrary
    from b12x.preparation import PreparationSession
    from helios.tests.fixture_frontend import server_config
    from helios.tests.signing import publisher
    from ignition.preparation.capture import capture_forwards
    from ignition.preparation.execution import GraphSignature, PreparedExecution
    from ignition.preparation.runtime_package import export_model

    library = NativeLibrary(args.library, target='cuda_sm120')
    device = torch.device('cuda:0')
    signatures = {'verify': GraphSignature('speculative', 2, 6)}
    graphs, compiled = {}, {}
    with PreparationSession(device=device, autotune=False, compile_workers=1, prime=False) as session:
        try:
            first, _, _, _ = fixture(library)
            common = dict(session=session, device=device, stream=torch.cuda.Stream(device=device),
                pool=storage.graph_pool.id, cache_root=args.output / 'compiled', retain_graphs=True)
            graphs, compiled = capture_forwards({'verify': first}, signatures, **common)
            graphs['verify'].reset()
            forward, buffers, state, metadata = fixture(library)
            assert forward.operation.handle != first.operation.handle
            canonical = reference_binding(buffers, metadata)
            graphs, compiled = capture_forwards({'verify': forward}, signatures, **common)
            assert compiled['verify'].cache_hit, 'fresh addresses/handles should reuse the AOT artifact'
            del first  # The restored forward must retain only its new native handles.
            session.freeze()
            cases = []
            scenarios = [(f'width-{w}', (w, max(1, w - 1)), (w, 1), (1, 3), 2)
                         for w in (1, 2, 4, 6)]
            scenarios += [(f'prefix-{count}', (6, 6), (count, 6 - count), (3, 1), 2)
                          for count in range(7)]
            scenarios += [('inactive', (0, 0), (0, 0), (1, 1), 0),
                          ('empty-row', (6, 0), (2, 0), (1, 1), 2),
                          ('null-row', (6, 6), (2, 0), (1, 0), 2),
                          ('one-live', (6, 0), (6, 0), (3, 1), 1)]
            input_names = ('mixed', 'gates', 'state', 'counts', 'z',
                'metadata.query_start_loc', 'metadata.state_indices', 'metadata.num_seqs', 'metadata.num_tokens')
            checked = ('state', 'output', 'observed', 'work.errors', 'commit_errors',
                'work.recorded_lengths', 'work.recorded_slots')
            for index, (name, lengths, counts, slots, live) in enumerate(scenarios):
                torch.manual_seed(801 + index)
                buffers['mixed'].normal_(std=.5); buffers['gates'].normal_(std=.8)
                buffers['z'].normal_(std=.3); buffers['state'].fill_(.03125)
                state.normal_(std=.02); state[0].zero_()
                metadata.query_start_loc.copy_(torch.tensor([0, lengths[0], sum(lengths)], device=device))
                metadata.state_indices.zero_()
                metadata.state_indices[:, 0].copy_(torch.tensor(slots, device=device))
                metadata.num_seqs.fill_(live); metadata.num_tokens.fill_(sum(lengths[:live]))
                buffers['counts'].copy_(torch.tensor(counts, device=device))
                initial = state.clone()
                uploads = {key: save(args.output, f'{name}-in-{key}', buffers[key]) for key in input_names}
                before = torch.cuda.memory_stats(device)['allocation.all.allocated']
                graphs['verify'].replay(); torch.cuda.synchronize()
                assert torch.cuda.memory_stats(device)['allocation.all.allocated'] == before
                assert not buffers['work.errors'].any().item()
                assert not buffers['commit_errors'].any().item()
                assert buffers['state'][:, -64:].eq(.03125).all().item()
                check_reference(buffers, initial, state, lengths, slots, counts, live, canonical)
                expected = {key: save(args.output, f'{name}-out-{key}', buffers[key]) for key in checked}
                cases.append(dict(name=name, graph='verify', uploads=uploads, expected=expected))
                print(f'{name}: Flare reference and compiled replay passed', flush=True)
            # Use the production fixed-VMM export path. Automatic discovery retains
            # native C-struct pointers at their recorded addresses; relocation would
            # require explicit per-parameter bindings for every captured kernel.
            buffers.update(frame_index=torch.zeros(1, dtype=torch.int32, device=device),
                submission_sequence=torch.zeros(1, dtype=torch.int64, device=device),
                completion_records=torch.zeros(2, 2, 32, dtype=torch.int64, device=device))
            prepared = PreparedExecution(device=0, graphs=graphs, buffers=buffers,
                signatures=signatures, max_active=2, resources=(forward, compiled, library, storage))
            prepared.server_config = server_config(args.output)
            prepared.package_data = None
            package = export_model(prepared, images, storage, args.output / 'package',
                options={}, publisher=publisher())
            references = {}
            for name, tensor in buffers.items():
                allocation, = [a for a in package.allocations if a.address <= tensor.data_ptr()
                    and tensor.data_ptr() + tensor.nbytes <= a.address + a.nbytes]
                references[name] = dict(allocation=allocation.name, offset=tensor.data_ptr() - allocation.address,
                    nbytes=tensor.nbytes)
            manifest = dict(kind='actlize-gdn-export-v1', cases=cases,
                addresses={a.name: a.address for a in package.allocations},
                buffers=references, export_counts=package.export_counts)
            (args.output / 'expected.json').write_text(json.dumps(manifest, indent=2) + '\n')
            print(json.dumps(dict(cases=len(cases), export=package.export_counts)), flush=True)
        finally:
            torch.cuda.synchronize()
            for graph in graphs.values():
                graph.reset()


def prepare(args):
    from ignition.preparation.module_images import ModuleImages
    from ignition.preparation.package_storage import PackageStorage

    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    with ModuleImages() as images, PackageStorage() as storage:
        qualify(args, images, storage)


if __name__ == '__main__':
    from qualify.actlize_export import prepare

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    prepare(args)
