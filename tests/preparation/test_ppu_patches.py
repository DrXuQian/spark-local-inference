# SPDX-License-Identifier: Apache-2.0

"""PPU setup and unmodified Torch 2.9 static caching with short kernel symbols."""

import ast
import importlib.util
from importlib.metadata import distribution
from importlib.resources import files
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import pytest

from ignition import patches
from ignition.errors import PreparationError


@pytest.fixture
def installation(tmp_path, monkeypatch):
    installed = {name: distribution(name) for name in ('torch', 'triton')}
    if any(installed[name].version.partition('+')[0] != version for name, version in patches._PPU_VERSIONS.items()):
        pytest.skip('requires the inspected Torch 2.9.0/Triton 3.5.0 sources')
    paths = ('torch/_inductor/runtime/triton_heuristics.py', 'triton/runtime/build.py')
    original = {}
    for relative in paths:
        source = Path(installed[relative.split('/')[0]].locate_file(relative)).read_bytes()
        # A box may already have applied these fixes. Normalize only the test
        # copy so both first application and idempotence still execute here.
        # Also undo the old Torch workaround in the fixture, so boxes that
        # already applied it still exercise unmodified static-loader behavior.
        for patch in (*patches._PPU_PATCHES, 'torch29-inductor-triton-cache-filenames.patch'):
            for target, before, after in patches._hunks(files(patches).joinpath(patch).read_text()):
                if target == relative and source.count(after.encode()) == 1:
                    source = source.replace(after.encode(), before.encode(), 1)
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source)
        original[relative] = source
        monkeypatch.delitem(sys.modules, relative.removesuffix('.py').replace('/', '.'), raising=False)
    packages = {name: SimpleNamespace(version=version + '+platform', locate_file=lambda path: tmp_path / path)
                for name, version in patches._PPU_VERSIONS.items()}
    monkeypatch.setattr(patches, 'distribution', packages.__getitem__)
    return tmp_path, packages, original


@pytest.mark.parametrize('target', ['ppu10', 'ppu15'])
def test_platform_patch_set_leaves_torch_unchanged(installation, target):
    root, _, original = installation
    torch_path = root / 'torch/_inductor/runtime/triton_heuristics.py'
    before = torch_path.stat()
    changed = patches.apply(operator_target=target)
    assert {path.relative_to(root).as_posix() for path in changed} == {'triton/runtime/build.py'}
    assert patches.apply(operator_target=target) == ()
    assert torch_path.read_bytes() == original[torch_path.relative_to(root).as_posix()]
    assert (torch_path.stat().st_ino, torch_path.stat().st_mtime_ns) == (before.st_ino, before.st_mtime_ns)


@pytest.mark.parametrize('name,version', [('torch', '2.8.0'), ('torch', '2.9.1'), ('triton', '3.6.0')])
def test_wrong_platform_version_fails_before_any_write(installation, name, version):
    root, packages, original = installation
    packages[name].version = version
    with pytest.raises(PreparationError, match=f'require {name}=='):
        patches.apply(operator_target='ppu15')
    assert all((root / path).read_bytes() == value for path, value in original.items())


def test_platform_source_mismatch_never_partially_patches_the_installation(installation):
    root, _, original = installation
    path = root / 'triton/runtime/build.py'
    path.write_text(path.read_text().replace('from platform import machine, system, architecture', 'pass'))
    original['triton/runtime/build.py'] = path.read_bytes()
    with pytest.raises(PreparationError, match='preserve the installed artifacts'):
        patches.apply(operator_target='ppu15')
    assert all((root / path).read_bytes() == value for path, value in original.items())


def test_platform_patch_cli_selects_the_ppu_environment(installation, capsys):
    from ignition.patches.__main__ import main

    assert main(['--operator-target', 'ppu10']) == 0
    assert capsys.readouterr().out.count('Patched ') == 1
    assert main(['--operator-target', 'ppu10']) == 0
    assert 'already applied' in capsys.readouterr().out


def test_short_symbols_compile_and_reload_statically_without_patching_torch(installation, tmp_path, monkeypatch):
    """Real Inductor naming and offline compilation, including a cold disk reload."""
    _, _, original = installation
    import torch
    import triton
    from torch._inductor.codegen.triton import TritonScheduling
    from torch._inductor.runtime import triton_heuristics as runtime
    from torch._inductor.utils import Placeholder
    from torch._inductor.virtualized import V
    from torch.utils._ordered_set import OrderedSet
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from ignition.preparation.compilation import _OPTIONS

    graph, origins = torch.fx.Graph(), []
    for op in ('__rshift__', '_to_copy', 'bitwise_and', 'bitwise_not', 'bitwise_or', 'clamp', 'div',
               'floor_divide', 'gt', 'index', 'mul', 'remainder', 'scalar_tensor', 'slice', 'sub',
               'unsqueeze', 'view', 'where'):
        node = graph.call_function(torch.neg, (graph.placeholder(op),))
        node.meta['original_aten'] = SimpleNamespace(_overloadpacket=SimpleNamespace(__name__=op))
        origins.append(node)
    schedule = [SimpleNamespace(node=SimpleNamespace(origins=OrderedSet(origins)))]
    source = f'''import triton
import triton.language as tl
# @triton_heuristics.pointwise
# {Placeholder.DESCRIPTIVE_NAME}
@triton.jit
def {Placeholder.KERNEL_NAME}(source, output, N: tl.constexpr):
    indices = tl.arange(0, N)
    tl.store(output + indices, tl.load(source + indices))
'''

    def no_device(*args, **kwargs):
        pytest.fail('static cache filename regression must not initialize CUDA')

    monkeypatch.setattr(torch.cuda, '_lazy_init', no_device)

    def result_class(source):
        tree = ast.parse(source)
        definition, = [node for node in tree.body if isinstance(node, ast.ClassDef)
                       and node.name == 'StaticTritonCompileResult']
        imports = [node for node in tree.body if isinstance(node, ast.ImportFrom) and node.module == '__future__']
        namespace = vars(runtime).copy()
        exec(compile(ast.Module(body=[*imports, definition], type_ignores=[]), '<installed-inductor>', 'exec'), namespace)
        return namespace['StaticTritonCompileResult']

    unpatched = result_class(original['torch/_inductor/runtime/triton_heuristics.py'])
    options = dict(_OPTIONS, compile_threads=1, use_static_cuda_launcher=True,
                   strict_static_cuda_launcher=True, cpp_wrapper=False)
    restored_paths = []
    with triton.knobs.cache.scope(), torch._inductor.config.patch(options):
        for unique in (True, _OPTIONS['triton.unique_kernel_names']):
            definitions = []
            wrapper = SimpleNamespace(src_to_kernel={}, next_kernel_suffix=lambda: '2', comment='#',
                                      define_kernel=lambda *args: definitions.append(args))
            context = SimpleNamespace(wrapper_code=wrapper,
                                      get_current_device_or_throw=lambda: torch.device('cuda:0'))
            with V.set_graph_handler(context), torch._inductor.config.patch({'triton.unique_kernel_names': unique}):
                descriptive = TritonScheduling.define_kernel(None, source, schedule, None)
            call = ast.parse(definitions[0][1]).body[0].value
            symbol, generated = (ast.literal_eval(arg) for arg in call.args)
            path = tmp_path / ('unique.py' if unique else 'short.py')
            path.write_text(generated)
            spec = importlib.util.spec_from_file_location(path.stem, path)
            module = importlib.util.module_from_spec(spec)
            monkeypatch.setitem(sys.modules, spec.name, module)
            spec.loader.exec_module(module)
            triton.knobs.cache.dir = str(tmp_path / ('unique-cache' if unique else 'short-cache'))
            compiled = triton.compile(
                ASTSource(getattr(module, symbol), signature={'source': '*fp32', 'output': '*fp32', 'N': 'constexpr'},
                          constexprs={'N': 32}),
                target=GPUTarget('cuda', 80, 32), options={'num_warps': 4, 'num_stages': 1})
            metadata = {'device_type': 'cuda', 'device': 0}
            if unique:
                assert len(symbol) == 164
                with pytest.raises(runtime.CannotStaticallyLaunchKernel, match='Cubin path not found'):
                    unpatched.can_statically_launch(compiled, {}, metadata, runtime.HeuristicType.POINTWISE)
                continue
            assert symbol == 'triton_' and len(descriptive) == 164
            owner = unpatched.can_statically_launch(compiled, {}, metadata, runtime.HeuristicType.POINTWISE)
            assert Path(owner.cubin_path).read_bytes() == compiled.asm['cubin']
            other = triton.compile(
                ASTSource(getattr(module, symbol), signature={'source': '*fp32', 'output': '*fp32', 'N': 'constexpr'},
                          constexprs={'N': 64}),
                target=GPUTarget('cuda', 80, 32), options={'num_warps': 4, 'num_stages': 1})
            other_owner = unpatched.can_statically_launch(other, {}, metadata, runtime.HeuristicType.POINTWISE)
            assert owner.name == other_owner.name == 'triton_'
            assert owner.hash != other_owner.hash and owner.cubin_path != other_owner.cubin_path
            assert Path(other_owner.cubin_path).read_bytes() == other.asm['cubin']
            owner.cubin_raw = compiled.asm['cubin']
            saved = unpatched(pickle.loads(pickle.dumps(owner)), triton.Config({}), metadata, {})
            saved.kernel.cubin_path = None
            triton.knobs.cache.dir = str(tmp_path / 'restored-cache')
            saved.reload_cubin_path()
            assert Path(saved.kernel.cubin_path).name == 'triton_.cubin'
            assert Path(saved.kernel.cubin_path).read_bytes() == compiled.asm['cubin']
            restored_paths.append(saved.kernel.cubin_path)
    assert len(restored_paths) == 1
