# SPDX-License-Identifier: Apache-2.0

"""Check installed PPU interfaces and run operator/graph gates without b12x.

This qualifies the port's dependencies and prepared components. It does not
claim complete Qwen model export or Rust replay on PPU. No models are downloaded,
no packages are installed, and supplied source trees/libraries are not modified.
"""

import argparse
import importlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


def existing_path(value):
    try:
        return Path(value).resolve(strict=True)
    except OSError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--quactlize-source', type=existing_path, required=True)
    result.add_argument('--actlize-source', type=existing_path, required=True)
    result.add_argument('--quactlize-library', type=existing_path, required=True,
                        help='PPU DSO exporting both matmul v1 and serving helpers v2/v1')
    result.add_argument('--actlize-library', type=existing_path, required=True)
    result.add_argument('--actlize-target', choices=('ppu10', 'ppu15'), required=True)
    result.add_argument('--blas-library', type=existing_path, required=True)
    result.add_argument('--blas-api', choices=('acblas', 'cublas'), default='acblas')
    result.add_argument('--run-device-tests', action='store_true',
                        help='run actual numerical, captured graph and framework binding gates')
    result.add_argument('--output', type=Path, help='new directory for device-test logs and raw results')
    return result


def pytest_result(path):
    """A skipped device gate is not qualification, even if pytest exits zero."""
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == 'testsuite' else root.findall('.//testsuite')
    counts = {key: sum(int(s.get(key, 0)) for s in suites)
              for key in ('tests', 'failures', 'errors', 'skipped')}
    return counts, counts['tests'] > 0 and not any(counts[k] for k in ('failures', 'errors', 'skipped'))


def device_suites(args, root, output):
    python = sys.executable
    return (
        ('framework', [python, '-m', 'pytest', '-q',
            'tests/components/test_quactlize_helpers_cuda.py',
            'tests/components/test_ppu_gguf_projection_cuda.py',
            'tests/components/test_blas_projection_cuda.py',
            'tests/components/test_actlize_gdn_sequence_cuda.py',
            'tests/components/test_actlize_mixing_cuda.py',
            'tests/components/test_actlize_ple_cuda.py',
            '--junitxml=' + str(output / 'framework.xml')], root, 'framework.xml'),
        ('attention-api', [python, '-m', 'pytest', '-q', 'tests/components/test_flash_attn3_paged_cuda.py',
            '--junitxml=' + str(output / 'attention-api.xml')], root, 'attention-api.xml'),
        ('attention-prepared', [python, '-m', 'pytest', '-q', 'tests/components/test_flash_attn3_prepared_cuda.py',
            '--junitxml=' + str(output / 'attention-prepared.xml')], root, 'attention-prepared.xml'),
        ('compiled-catalog', [python, '-m', 'pytest', '-q', 'tests/preparation/test_external_aot_cuda.py',
            'tests/preparation/test_external_session_cuda.py',
            '--junitxml=' + str(output / 'compiled-catalog.xml')], root, 'compiled-catalog.xml'),
        ('quactlize', [python, '-m', 'pytest', '-q', 'tests/serving/test_serving_device.py',
            '--junitxml=' + str(output / 'quactlize.xml')], args.quactlize_source, 'quactlize.xml'),
        ('actlize', [python, str(args.actlize_source / 'tests/serving_device.py'),
            '--library', str(args.actlize_library), '--target', args.actlize_target,
            '--results', str(output / 'actlize.json')], args.actlize_source, 'actlize.json'),
        ('actlize-aux', [python, str(args.actlize_source / 'tests/aux_device.py'),
            '--library', str(args.actlize_library), '--target', args.actlize_target,
            '--results', str(output / 'actlize-aux.json')], args.actlize_source, 'actlize-aux.json'),
    )


def check_interfaces(args):
    results = {}

    def check(name, action):
        try:
            detail = action()
        except Exception as exc:
            results[name] = False
            print(f'FAIL {name}: {type(exc).__name__}: {exc}', flush=True)
        else:
            results[name] = True
            print(f'PASS {name}: {detail}', flush=True)

    def package(name):
        module = importlib.import_module(name)
        return f'{getattr(module, "__version__", "version unreported")} {module.__file__}'

    for name in ('torch', 'gguf', 'numpy', 'pytest', 'google.protobuf', 'quactlize.serving', 'actlize_la.serving'):
        check(name, lambda name=name: package(name))

    def device():
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError('a real CUDA-compatible PPU and platform Torch are required')
        props = torch.cuda.get_device_properties(0)
        return f'{props.name}; capability={props.major}.{props.minor}; VRAM={props.total_memory}'

    check('device', device)

    def preparation_api():
        import torch
        from ignition.preparation.aot import _GuardPickler
        from ignition.preparation.forward_metadata import dumps, loads
        from ignition.preparation.torch_aot_api import TORCH29

        if TORCH29:
            from torch._dynamo.aot_compile import AOTCompiledFunction
            from torch._functorch._aot_autograd.autograd_cache import BundledAOTAutogradCacheEntry
            from ignition.preparation.inductor29 import BundledArtifact

            if (not callable(AOTCompiledFunction.deserialize) or not isinstance(BundledAOTAutogradCacheEntry, type)
                    or not callable(BundledArtifact.load_entry)):
                raise RuntimeError('platform Torch 2.9 lacks the bundled forward/guard protocol')
        else:
            from torch._inductor.standalone_compile import AOTCompiledArtifact
            from torch._functorch._aot_autograd.aot_autograd_result import deserialize_bundled_cache_entry

            if not callable(deserialize_bundled_cache_entry) or not isinstance(AOTCompiledArtifact, type):
                raise RuntimeError('platform Torch does not provide the required compiled-forward serialization API')
        if 'keep_graph' not in inspect.signature(torch.cuda.CUDAGraph).parameters:
            raise RuntimeError('platform Torch CUDAGraph lacks keep_graph for graph export')
        value = torch.empty(3, 11)
        original, view = loads(dumps((value, value[1:, 3:])))
        if (original.untyped_storage()._cdata != view.untyped_storage()._cdata
                or view.stride() != (11, 1) or view.storage_offset() != 14):
            raise RuntimeError('platform Torch metadata serialization changed tensor aliases')
        import io

        _GuardPickler(io.BytesIO()) if TORCH29 else _GuardPickler({}, {}, {}, {}, io.BytesIO())
        return 'AOT serialization/guards and retained graphs present; metadata aliases preserved (compilation/replay untested)'

    check('torch-preparation-api', preparation_api)

    def quactlize():
        import ctypes as C
        from quactlize.compat.native import Library, Sizes, problem
        from quactlize.compat.serving_native import Library as HelpersLibrary

        library = Library(args.quactlize_library)
        HelpersLibrary(args.quactlize_library)
        checked = 0
        for qtype in (7, 8, 10, 11, 12, 13, 14):
            for rows in (1, 9):
                for n, k, channels in ((1280, 2560, 1), (2560, 640, 10)):
                    if qtype >= 10 and k % 256:
                        continue  # GGUF K-block rows require complete 256-element blocks.
                    call = problem(qtype, rows, n, k, experts=512, topk=10, channels=channels)
                    sizes = Sizes()
                    library.check(library.query(C.byref(call), None, C.byref(sizes)))
                    if not sizes.raw_bytes or not sizes.workspace_bytes or sizes.output_bytes != rows * 10 * n * 2:
                        raise RuntimeError('matmul query returned incompatible Qwen geometry')
                    checked += 1
        return f'PPU matmul v1 ({checked} Qwen geometry queries), router/SwiGLU v2, row lookup v1'

    def actlize():
        import ctypes as C
        from actlize_la.serving import NativeLibrary
        from actlize_la.serving.auxiliary import TensorV1, RowsV1, PointwiseV1, ShortConvV1, HashV1
        from actlize_la.serving.gdn_transaction import TransactionV2

        library = NativeLibrary(args.actlize_library, target=args.actlize_target)
        for name in ('ala_aux_v1_validate', 'ala_aux_v1_struct_size', 'ala_pointwise_v1_run',
                     'ala_shortconv_v1_run', 'ala_ple_hash_v1_run', 'ala_gdn_transaction_v2_size'):
            getattr(library.lib, name)
        size = library.lib.ala_aux_v1_struct_size
        size.argtypes, size.restype = [C.c_int], C.c_int
        for index, structure in enumerate((TensorV1, RowsV1, PointwiseV1, ShortConvV1, HashV1)):
            if size(index) != C.sizeof(structure):
                raise RuntimeError(f'auxiliary descriptor size mismatch at index {index}')
        if library.lib.ala_gdn_transaction_v2_size() != C.sizeof(TransactionV2):
            raise RuntimeError('GDN transaction descriptor size mismatch')
        return f'{library.target}; GDN/QSA, Norm/HC/PLE/ShortConv and transaction v2 symbols'

    def blas():
        from ignition.components.blas_projection import BlasLibrary

        BlasLibrary(args.blas_library, api=args.blas_api)
        return f'{args.blas_api} GemmEx, stream and caller-owned workspace API'

    check('quactlize-native', quactlize)
    check('actlize-native', actlize)
    check('dense-blas', blas)

    def flash_attention():
        import torch

        module = importlib.import_module('flash_attn_3._C')
        operation = torch.ops.flash_attn_3.fwd.default
        fields = {argument.name for argument in operation._schema.arguments}
        required = {'q', 'k', 'v', 'k_new', 'v_new', 'out', 'cu_seqlens_q', 'cu_seqlens_k_new',
                    'seqused_k', 'max_seqlen_q', 'page_table', 'is_causal', 'num_splits'}
        if not required <= fields:
            raise RuntimeError('missing FA3 forward arguments: ' + ', '.join(sorted(required - fields)))
        return f'{module.__file__}; ABI present, page64/D256/append still require attention-api suite'

    check('flash-attention3-api', flash_attention)

    def flashinfer():
        from flashinfer import sampling

        if not callable(getattr(sampling, '_top_p_renorm_workspace_size', None)):
            raise RuntimeError('missing _top_p_renorm_workspace_size; stock sampling API is insufficient')
        parameters = inspect.signature(sampling.chain_speculative_sampling).parameters
        if 'row_independent_rng' not in parameters:
            raise RuntimeError('MTP verification requires chain_speculative_sampling(row_independent_rng=...)')
        if not results['device']:
            raise RuntimeError('device required to load and check prepared sampling exports')
        import torch

        # Registration is lazy; load before checking torch.ops, outside capture.
        sampling.get_sampling_module()
        for name in ('top_k_mask_logits_out', 'top_p_renorm_probs_out'):
            getattr(torch.ops.flashinfer, name).default
        return 'caller-owned filtering outputs/workspace and per-request MTP RNG interface'

    check('flashinfer-sampling', flashinfer)
    forbidden = sorted(name for name in sys.modules if name == 'b12x' or name.startswith(('b12x.', 'cuda.tile')))
    if forbidden:
        results['no-cutile-import'] = False
        print('FAIL no-cutile-import: ' + ', '.join(forbidden[:8]), flush=True)
    else:
        results['no-cutile-import'] = True
        print('PASS no-cutile-import: inspected path did not import b12x or cuda.tile', flush=True)
    return results


def main(argv=None):
    options = parser()
    args = options.parse_args(argv)
    if args.run_device_tests and args.output is None:
        options.error('--run-device-tests requires --output pointing to a new directory')
    if args.run_device_tests and args.output.exists():
        options.error('--output already exists; select a new directory to keep test results separate')
    root = Path(__file__).resolve().parents[1]
    sources = (args.quactlize_source, args.actlize_source)
    for source, package in zip(sources, ('quactlize', 'actlize_la'), strict=True):
        entry = source / package / ('serving' if package == 'actlize_la' else 'serving.py')
        if not entry.exists():
            options.error(f'{source}: updated {package} serving package is missing')
    # Use the exact requested source declarations with the explicitly named DSOs.
    sys.path[:0] = [str(p) for p in sources]
    results = check_interfaces(args)
    if args.run_device_tests:
        required = ('device', 'torch', 'gguf', 'numpy', 'pytest', 'google.protobuf',
                    'quactlize.serving', 'actlize_la.serving', 'quactlize-native', 'actlize-native', 'dense-blas')
        if not all(results[name] for name in required):
            print('Device suites not started: fix the failed device/operator prerequisites.', flush=True)
            return 1
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
        env = dict(os.environ)
        env['PYTHONPATH'] = os.pathsep.join(map(str, (*sources, root))) + (
            os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
        env.update(FLARE_RUN_CUDA_TESTS='1', FLARE_QUACTLIZE_HELPERS_LIBRARY=str(args.quactlize_library),
                   FLARE_TEST_PPU_MATMUL_LIBRARY=str(args.quactlize_library),
                   QUACTLIZE_SERVING_LIBRARY=str(args.quactlize_library),
                   FLARE_ACTLIZE_LIBRARY=str(args.actlize_library), FLARE_ACTLIZE_TARGET=args.actlize_target,
                   FLARE_TEST_BLAS_LIBRARY=str(args.blas_library), FLARE_TEST_BLAS_API=args.blas_api)
        for name, command, cwd, result in device_suites(args, root, output):
            log = output / (name + '.log')
            print(f'RUN {name}: output in {log}', flush=True)
            with log.open('w') as stream:
                process = subprocess.run(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT)
            path = output / result
            passed = False
            if process.returncode == 0 and path.is_file():
                if path.suffix == '.xml':
                    detail, passed = pytest_result(path)
                else:
                    detail = json.loads(path.read_text())
                    passed = detail.get('status') == 'PASS' and bool(detail.get('cases'))
                    detail = dict(status=detail.get('status'), cases=len(detail.get('cases', [])))
            else:
                detail = dict(returncode=process.returncode, log=str(log))
            results['suite-' + name] = passed
            print(f'{"PASS" if passed else "FAIL"} {name}: {detail}', flush=True)
    print('PENDING: complete GGUF/MTP catalog, normal CLI target admission, '
          'and PPU native export/Rust replay. Operator tests do not establish bubble-free serving.', flush=True)
    return 0 if all(results.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
