# SPDX-License-Identifier: Apache-2.0

"""Qualify the PPU matmul record bridge with actual GGUF weights on NVIDIA."""

import argparse
from pathlib import Path

import torch

from qualify.actlize_export import export_fixture, save


class Forward:
    def __init__(self, operation, source, first_ids, second_ids, first_output, output, seed):
        self.operation, self.source = operation, source
        self.first_ids, self.second_ids = first_ids, second_ids
        self.first_output, self.output, self.seed = first_output, output, seed

    def __call__(self):
        self.operation.device_error.copy_(self.seed)
        self.operation(self.source, ids=self.first_ids, output=self.first_output)
        # A later valid call must preserve a preceding invalid route's error bit.
        self.operation(self.source, ids=self.second_ids, output=self.output)


class Fixture:
    capacities = (1, 2, 4, 6, 9, 64)

    def __init__(self, library, weight):
        from ignition.components.ppu_gguf_projection import PreparedPpuGgufProjection, projection_sizes

        if weight.shape != (512, 2560, 640) or weight.qtype not in (7, 8):
            raise ValueError('qualification requires a complete Qwen Q5_1/Q8_0 down-expert tensor')
        device = weight.raw.device
        self.weight = weight
        self.source = torch.zeros((64, 10, 647), dtype=torch.bfloat16, device=device)
        self.first_ids = torch.zeros((64, 17), dtype=torch.int32, device=device)
        self.second_ids = torch.zeros_like(self.first_ids)
        self.first_output = torch.full((64, 10, 2567), -123, dtype=torch.bfloat16, device=device)
        self.output = torch.full_like(self.first_output, -123)
        self.seed = torch.zeros((), dtype=torch.int32, device=device)
        views = [(m, self.source[:m, :, :640], self.first_ids[:m, :10], self.output[:m, :, :2560])
                 for m in self.capacities]
        required = max(projection_sizes(library, source, weight, output, ids=ids)[1].workspace_bytes
                       for _, source, ids, output in views)
        self.workspace = torch.empty(required, dtype=torch.uint8, device=device)
        self.forwards, self.errors = {}, {}
        for m, source, ids, output in views:
            operation = PreparedPpuGgufProjection(library, source, weight, output, ids=ids, workspace=self.workspace)
            name = f'project{m}'
            self.forwards[name] = Forward(operation, source, ids, self.second_ids[:m, :10],
                self.first_output[:m, :, :2560], output, self.seed)
            self.errors[name] = operation.device_error
        self.buffers = dict(source=self.source, first_ids=self.first_ids, second_ids=self.second_ids,
            first_output=self.first_output, output=self.output, seed=self.seed,
            **{f'error.{name}': error for name, error in self.errors.items()})

    def inputs(self, width, mode):
        torch.manual_seed(401 + width)
        self.source.normal_(std=.2)
        self.first_ids.fill_(-2); self.second_ids.fill_(-2)
        self.first_output.fill_(-123); self.output.fill_(-123)
        if mode == 'duplicates':
            first, second, bits = [511] * 10, [0, 1, 511, 1, 0] * 2, 0
        elif mode == 'inactive':
            first, second, bits = [-1, 511] * 5, [-1] * 10, 0
        else:
            first, second, bits = [-2, 512] + [0] * 8, [1] * 10, 2
        self.first_ids[:width, :10].copy_(torch.tensor([first] * width, device=self.source.device))
        self.second_ids[:width, :10].copy_(torch.tensor([second] * width, device=self.source.device))
        self.seed.fill_(bits)
        return first, second, bits | int(mode == 'invalid-then-valid')

    def check(self, width, routes, reference):
        first, second, bits = routes
        assert self.errors[f'project{width}'].item() == bits
        for output, ids in ((self.first_output, first), (self.output, second)):
            assert bool((output[..., 2560:] == -123).all())
            assert bool((output[width:] == -123).all())
            x = self.source[:width, :, :640].cpu().double()
            for slot, expert in enumerate(ids):
                actual = output[:width, slot, :2560].cpu().double()
                if not 0 <= expert < 512:
                    assert not torch.count_nonzero(actual)
                    continue
                w = reference[expert]
                expected = x[:, slot] @ w.T
                magnitude = x[:, slot].abs() @ w.abs().T
                bound = expected.abs() * torch.finfo(torch.bfloat16).eps + magnitude * (2e-6 + 640 * 2**-24)
                assert bool(((actual - expected).abs() <= bound + 1e-7).all())


def qualify(args, images, storage):
    import gguf
    from b12x.preparation import PreparationSession
    from quactlize.cuda_compat import CudaPpuMatmulLibrary
    from ignition.components.gguf_weights import GgufWeightLoader
    from ignition.preparation.capture import capture_forwards
    from ignition.preparation.execution import GraphSignature
    from qualify.qwen38_gguf import component_checkpoint

    library = CudaPpuMatmulLibrary(args.library)
    manifest = component_checkpoint(args.target_shard, args.metadata, [44])
    with GgufWeightLoader(manifest, 'cuda:0', read_mode='pread') as loader:
        weight = loader.load_quantized('blk.44.ffn_down_exps.weight')
    reference = {expert: torch.from_numpy(gguf.dequantize(weight.raw[expert].cpu().numpy().reshape(2560, -1),
        gguf.GGMLQuantizationType(weight.qtype))).bfloat16().double() for expert in (0, 1, 511)}
    print(f'Loaded all 512 actual {weight.encoding} experts; NVIDIA implementation of PPU records', flush=True)
    device = torch.device('cuda:0')
    signatures = {f'project{m}': GraphSignature('prefill', 1, m) for m in Fixture.capacities}
    graphs, compiled = {}, {}
    with PreparationSession(device=device, autotune=False, compile_workers=1, prime=False) as session:
        try:
            common = dict(session=session, device=device, stream=torch.cuda.Stream(device=device),
                pool=storage.graph_pool.id, cache_root=args.output / 'compiled', retain_graphs=True)
            first = Fixture(library, weight)
            graphs, compiled = capture_forwards(first.forwards, signatures, **common)
            for graph in graphs.values():
                graph.reset()
            current = Fixture(library, weight)
            assert all(current.forwards[name].operation.handle != first.forwards[name].operation.handle for name in signatures)
            graphs, compiled = capture_forwards(current.forwards, signatures, **common)
            assert all(call.cache_hit for call in compiled.values()), 'fresh native handles must reuse the AOT artifacts'
            del first
            session.freeze()
            cases = []
            inputs = ('source', 'first_ids', 'second_ids', 'first_output', 'output', 'seed')
            for width in Fixture.capacities:
                graph_name = f'project{width}'
                for mode in ('duplicates', 'inactive', 'invalid-then-valid'):
                    name = f'{graph_name}-{mode}'
                    routes = current.inputs(width, mode)
                    uploads = {key: save(args.output, f'{name}-in-{key}', current.buffers[key]) for key in inputs}
                    before = torch.cuda.memory_stats(device)['allocation.all.allocated']
                    graphs[graph_name].replay(); torch.cuda.synchronize()
                    assert torch.cuda.memory_stats(device)['allocation.all.allocated'] == before
                    current.check(width, routes, reference)
                    expected = {key: save(args.output, f'{name}-out-{key}', current.buffers[key])
                                for key in ('first_output', 'output', f'error.{graph_name}')}
                    cases.append(dict(name=name, graph=graph_name, uploads=uploads, expected=expected))
                    print(f'{name}: independent GGUF/FP64 reference and compiled replay passed', flush=True)
            export_fixture(args, images, storage, graphs, signatures, current.buffers,
                (current, compiled, library, weight), cases, 'quactlize-ppu-abi-nvidia-v1')
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
    from qualify.quactlize_abi_export import prepare

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', required=True, type=Path, help='explicit NVIDIA QUactlize ABI library')
    parser.add_argument('--target-shard', required=True, type=Path)
    parser.add_argument('--metadata', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'package').exists():
        raise FileExistsError(args.output / 'package')
    prepare(args)
