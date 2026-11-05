# SPDX-License-Identifier: Apache-2.0

"""Real file bytes and metadata routing; CUDA copies have separate device tests."""

import ctypes
import mmap
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from ignition.preparation.gguf import inspect_gguf_checkpoint
from ignition.preparation.gguf_pread import ALIGNMENT, BLOCK_BYTES, GgufPreadSession, read_direct, routed_rows
from ignition.preparation.weight_ranges import chunks

gguf = pytest.importorskip('gguf')
pytestmark = pytest.mark.skipif(not hasattr(os, 'O_DIRECT'), reason='POSIX direct reads')


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / 'tiny.gguf'
    data = np.arange(3 * 4 * 20 * 24, dtype=np.int32).astype(np.uint8).reshape(3, 4, 480)
    writer = gguf.GGUFWriter(path, 'qwen4exp')
    writer.add_tensor('gate.weight', data, raw_dtype=gguf.GGMLQuantizationType.Q5_1)
    writer.add_tensor('norm.weight', np.arange(17, dtype=np.float32))
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    return inspect_gguf_checkpoint(path), data.reshape(3, 4, 20, 24)


def test_real_metadata_sources_and_progress_do_not_import_b12x(checkpoint):
    manifest, _ = checkpoint
    code = '''
import sys
class NoB12x:
    def find_spec(self, fullname, *args):
        if fullname == 'b12x' or fullname.startswith('b12x.'):
            raise AssertionError('b12x dependency during GGUF pread')
sys.meta_path.insert(0, NoB12x())
from ignition.preparation.gguf import inspect_gguf_checkpoint
from ignition.components.gguf_weights import GgufWeightLoader
from pathlib import Path
manifest = inspect_gguf_checkpoint(Path(sys.argv[1]))
progress = []
with GgufWeightLoader(manifest, 'cuda:0', read_mode='pread', progress=progress.append) as loader:
    assert set(loader._sources) == {'gate.weight', 'norm.weight'}
    assert loader._sources['gate.weight'].shape == (3, 4, 20, 24)
    assert loader._session.stats()['payload_bytes'] == 0
assert any(message.endswith('0.00 GiB transferred') for message in progress)
'''
    result = subprocess.run([sys.executable, '-c', code, str(manifest.shard_paths[0])],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('selection', ['full', 'columns', 'join', 'transpose'])
def test_metadata_routes_read_exact_selected_gguf_bytes(checkpoint, selection):
    manifest, values = checkpoint
    with GgufPreadSession(manifest, device=0) as session:
        source = dict(session.weights(manifest.shard_paths))['gate.weight']
        expected = torch.from_numpy(values.copy())
        if selection == 'columns':
            source, expected = source[1:3, 1:4, 1:19], expected[1:3, 1:4, 1:19]
        elif selection == 'transpose':
            source, expected = source.transpose(0, 1), expected.transpose(0, 1)
        if selection == 'join':
            backing = torch.full((3, 8, 20, 24), 231, dtype=torch.uint8)
            output = backing[:, 4:]
        else:
            output = torch.empty(source.shape, dtype=source.dtype)
        fd, records = session._routes(output, source)
        with mmap.mmap(-1, BLOCK_BYTES + 2 * ALIGNMENT) as staging:
            for record in records:
                for offset, width, rows, pitch, pointer, stride in chunks(*record):
                    content, _ = read_direct(fd, offset, width + (rows - 1) * pitch, memoryview(staging))
                    for row in range(rows):
                        ctypes.memmove(pointer + row * stride,
                            ctypes.addressof(ctypes.c_char.from_buffer(content)) + row * pitch, width)
                    del content
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        if selection == 'join':
            assert torch.all(backing[:, :4] == 231)


@pytest.mark.parametrize('offset,size', [(0, 1), (31, 17), (4093, 65539), (31, BLOCK_BYTES),
                                       (BLOCK_BYTES + 64, 13)])
def test_direct_io_handles_unaligned_ranges_block_boundaries_and_eof(tmp_path, offset, size):
    path = tmp_path / 'bytes'
    data = (np.arange(BLOCK_BYTES + 77, dtype=np.int32) % 251).astype(np.uint8).tobytes()
    path.write_bytes(data)
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    try:
        with mmap.mmap(-1, BLOCK_BYTES + 2 * ALIGNMENT) as staging:
            content, physical = read_direct(fd, offset, size, memoryview(staging))
            assert bytes(content) == data[offset:offset + size]
            assert size <= physical <= BLOCK_BYTES + ALIGNMENT
            del content
    finally:
        os.close(fd)


def test_direct_io_rejects_short_payload_and_unaligned_staging(tmp_path):
    path = tmp_path / 'short'
    path.write_bytes(b'abc')
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    try:
        with mmap.mmap(-1, BLOCK_BYTES + 2 * ALIGNMENT) as staging:
            with pytest.raises(RuntimeError, match='short GGUF'):
                read_direct(fd, 0, 4, memoryview(staging))
            with pytest.raises(ValueError, match='aligned'):
                read_direct(fd, 0, 1, memoryview(staging)[1:])
    finally:
        os.close(fd)


@pytest.mark.parametrize('change', ['arithmetic', 'dtype', 'overlap', 'mutation', 'file'])
def test_changed_or_ambiguous_sources_fail_before_transfers(checkpoint, change):
    manifest, _ = checkpoint
    with pytest.raises((ValueError, RuntimeError), match='inspected|overlapping|modified|source changed'):
        with GgufPreadSession(manifest, device=0) as session:
            source = dict(session.weights(manifest.shard_paths))['gate.weight']
            if change == 'arithmetic':
                source = source + 1
            elif change == 'dtype':
                source = source.view(torch.int8)
            elif change == 'overlap':
                source = source.as_strided((2, 2), (1, 1))
            elif change == 'mutation':
                source[0].zero_()
                session._validate()
            else:
                with manifest.shard_paths[0].open('r+b') as output:
                    output.seek(manifest.tensors['gate.weight'].offset)
                    output.write(b'changed')
                # Some filesystems coalesce timestamps for rapid same-size writes.
                stat = manifest.shard_paths[0].stat()
                os.utime(manifest.shard_paths[0], ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
                session._validate()
            session._routes(torch.empty(source.shape, dtype=source.dtype), source)


def test_expansion_and_empty_routes_preserve_element_geometry():
    source = torch.empty((2, 5), device='meta', dtype=torch.bfloat16)
    output = torch.empty((2, 7), dtype=torch.float32)[:, 1:6]
    assert routed_rows(source, output, offset=64, nbytes=20) == ((64, 10, 2, 10, output.data_ptr(), 28, 1),)
    empty = source[:0]
    assert routed_rows(empty, output[:0], offset=64, nbytes=20) == ()
    with pytest.raises(ValueError, match='precision'):
        routed_rows(source, output.half(), offset=64, nbytes=20)


def test_failed_device_drain_retains_transfer_owners(checkpoint):
    from ignition.preparation.gguf_pread import _undrained

    manifest, _ = checkpoint
    class FailedStream:
        def synchronize(self):
            raise RuntimeError('device drain failed')
    session = GgufPreadSession(manifest, device=0)
    try:
        with pytest.raises(RuntimeError, match='device drain failed'), session:
            dict(session.weights(manifest.shard_paths))
            session.stream = FailedStream()
            session._drain()
        assert session in _undrained and session.files and session.sources
    finally:
        _undrained.remove(session)
        session.retained = False
        session.__exit__(RuntimeError, None, None)
