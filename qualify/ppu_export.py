# SPDX-License-Identifier: Apache-2.0

"""Qualify PPU module recording, global pointers and native package replay.

Compile the tiny test module on the deployment platform. This imports no model
or operator package; graph export uses the production recorder and VMM allocator.
The generated package is for test_actlize_runtime_package, not language serving.
"""

import argparse
import ctypes as C
from pathlib import Path
import struct
import subprocess


GLOBALS = {'qwen_export_counter', 'qwen_export_zero', 'qwen_export_constants', 'qwen_export_pointer'}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--target', choices=('ppu10', 'ppu15'), required=True)
    result.add_argument('--compiler', type=Path, default=Path('/usr/local/PPU_SDK/bin/hgcc'))
    result.add_argument('--output', type=Path, required=True)
    result.add_argument('--compile-only', action='store_true', help='offline module check; does not export or execute')
    return result


def compile_module(args):
    from ignition.preparation.module_images import global_symbols, strip_debug

    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'package').exists() or (args.output / 'expected.json').exists():
        raise FileExistsError('use a fresh qualification output directory')
    source = Path(__file__).resolve().parents[1] / 'tests/preparation/fixtures/ppu_module_globals.cu'
    module = args.output / 'ppu-module.hgbin'
    subprocess.run([str(args.compiler), '--hgbin', '--gpu-architecture=' + {'ppu10': 'ppu_10', 'ppu15': 'ppu_15'}[args.target],
                    '-O2', str(source), '-o', str(module)], check=True)
    data = module.read_bytes()
    if len(data) < 64 or data[:6] != b'\x7fELF\x02\x01' or struct.unpack_from('<H', data, 18)[0] != 270:
        raise ValueError('the compiler did not emit a native PPU ELF64 module')
    symbols = set(global_symbols(data))
    if not GLOBALS <= symbols or set(global_symbols(strip_debug(data))) != symbols:
        raise ValueError('PPU module globals must survive production debug stripping')
    print(f'PPU module compiled and ELF globals checked: {module}', flush=True)
    return module


def write_global(cuda, module, name, value):
    address, size = C.c_uint64(), C.c_size_t()
    cuda.call('cuModuleGetGlobal_v2', C.byref(address), C.byref(size), module, C.c_char_p(name.encode()))
    if size.value != C.sizeof(value):
        raise ValueError(f'{name}: global extent {size.value} != {C.sizeof(value)}')
    cuda.call('cuMemcpyHtoD_v2', address, C.byref(value), size)


def launch(cuda, function, output, control, tag, stream):
    values = (C.c_uint64(output.data_ptr()), C.c_uint64(control.data_ptr()), C.c_uint32(tag))
    parameters = (C.c_void_p * len(values))(*(C.addressof(value) for value in values))
    cuda.call('cuLaunchKernel', function, C.c_uint(1), C.c_uint(1), C.c_uint(1),
              C.c_uint(32), C.c_uint(1), C.c_uint(1), C.c_uint(0),
              C.c_void_p(stream.cuda_stream), parameters, None)


def scenarios():
    # Each complete cycle ends at the exported state, so the independent Rust
    # process can repeat it twice without host access to device module globals.
    return (
        ('initial', 'low', (0, 101, 37), (211, 101, 37, 31, 47, 17)),
        ('continued', 'high', (0, 101, 37), (211, 343, 38, 31, 47, 91)),
        ('restore', 'high', (1, 101, 37), (211, 585, 39, 31, 47, 91)),
        ('restored', 'low', (1, 101, 37), (211, 101, 37, 31, 47, 17)),
    )


def prepare(args, module_path):
    import torch
    from ignition.preparation.execution import GraphSignature
    from ignition.preparation.module_images import ModuleImages
    from ignition.preparation.package_storage import PackageStorage
    from qualify.actlize_export import export_fixture, save

    graphs = {}
    with ModuleImages() as images:
        torch.cuda.set_device(0)
        with PackageStorage() as storage:
            cuda = storage.cuda
            module, function, indirect = C.c_void_p(), C.c_void_p(), C.c_uint64()
            cuda.call('cuModuleLoad', C.byref(module), C.c_char_p(str(module_path).encode()))
            try:
                # This allocation is neither a graph parameter nor a registered
                # I/O buffer. Only the module-global pointer can keep it reachable.
                cuda.call('cuMemAlloc_v2', C.byref(indirect), C.c_size_t(65536))
                cuda.call('cuMemsetD8_v2', indirect, C.c_ubyte(0), C.c_size_t(65536))
                value = C.c_uint64(211)
                cuda.call('cuMemcpyHtoD_v2', indirect, C.byref(value), C.c_size_t(8))
                cuda.call('cuModuleGetFunction', C.byref(function), module, C.c_char_p(b'qwen_export_state_probe'))
                write_global(cuda, module, 'qwen_export_constants', (C.c_uint64 * 2)(31, 47))
                write_global(cuda, module, 'qwen_export_pointer', indirect)
                output = torch.zeros(6, dtype=torch.int64, device='cuda:0')
                control = torch.tensor([1, 101, 37], dtype=torch.int64, device='cuda:0')
                stream = torch.cuda.Stream(device=0)
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for name, tag in (('low', 17), ('high', 91)):
                        launch(cuda, function, output, control, tag, stream)
                        stream.synchronize()
                        graph = torch.cuda.CUDAGraph(keep_graph=True)
                        graphs[name] = graph
                        with torch.cuda.graph(graph, pool=storage.graph_pool.id, stream=stream):
                            launch(cuda, function, output, control, tag, stream)
                        graph.instantiate()
                    write_global(cuda, module, 'qwen_export_counter', C.c_uint64(101))
                    write_global(cuda, module, 'qwen_export_zero', C.c_uint64(37))
                    cases = []
                    for name, graph_name, inputs, expected in scenarios():
                        control.copy_(torch.tensor(inputs, dtype=torch.int64))
                        output.fill_(-1)
                        uploads = {'control': save(args.output, name + '-control', control)}
                        allocated = torch.cuda.memory_stats()['allocation.all.allocated']
                        graphs[graph_name].replay()
                        stream.synchronize()
                        assert torch.cuda.memory_stats()['allocation.all.allocated'] == allocated
                        assert output.cpu().tolist() == list(expected), name
                        expected_path = name + '-expected.bin'
                        (args.output / expected_path).write_bytes(struct.pack('<6Q', *expected))
                        cases.append(dict(name=name, graph=graph_name, uploads=uploads,
                                          expected={'output': expected_path}))
                    print('Four Python graph cases passed; exporting native package', flush=True)
                    package = export_fixture(args, images, storage, graphs,
                        {name: GraphSignature('decode', 1, 1, ('control',)) for name in graphs},
                        {'control': control, 'output': output}, (module, indirect, stream), cases,
                        'ppu-module-export-v1', operator_target=args.target)
                    exported = {name.rsplit('.', 1)[-1] for _, _, name in package.module_globals}
                    assert GLOBALS <= exported, exported
                    assert any(a.name.startswith('external-') and
                        a.address <= indirect.value < a.address + a.nbytes for a in package.allocations)
                print('PPU package exported; run test_actlize_runtime_package after this process exits', flush=True)
            finally:
                # Failed synchronization must not unload modules or free storage
                # that a pending graph may still use.
                torch.cuda.synchronize()
                for graph in graphs.values():
                    graph.reset()
                if indirect.value:
                    cuda.call('cuMemFree_v2', indirect)
                cuda.call('cuModuleUnload', module)


def main(argv=None):
    args = parser().parse_args(argv)
    args.output = args.output.resolve()
    module = compile_module(args)
    if args.compile_only:
        print('Compile-only: device recording, export and Rust replay remain untested', flush=True)
    else:
        prepare(args, module)


if __name__ == '__main__':
    main()
