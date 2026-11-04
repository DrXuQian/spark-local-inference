# SPDX-License-Identifier: Apache-2.0

"""GPU hashes, native sealed-RAM delivery and external IQ4_NL decode in one graph."""

import os
import time

import gguf
import numpy as np
import pytest
import torch
import flare_heliopause
from b12x.preparation import PreparationSession
from b12x.sequence import ple_embedding, ple_hash
from b12x.sequence._shared.disk_table import MappedHostAllocation
from b12x.sequence.ple_hash.reference import ple_hash_packed_reference

from ignition.components.ngram_delivery import PreparedPLELookup
from ignition.components.readiness import ReadinessTimings
from ignition.preparation.aot import ForwardCache
from ignition.preparation.execution import GraphSignature, PreparedExecution
from ignition.preparation.runtime_package import FileSource
from ignition.recipes.qwen38.gguf_ple import GgufPLELayout, GgufPLETable
from ignition.recipes.qwen38.ple import PLEMetadata


pytestmark = pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1', reason='explicit CUDA assignment required')


class Forward:
    def __init__(self, lookup, source, records, sequence):
        self.lookup, self.source, self.records, self.sequence = lookup, source, records, sequence

    def __call__(self):
        self.lookup.publish()
        self.source.add_(1)
        self.lookup(self.source)
        self.records.zero_()
        self.records[:, 0, 0].fill_(1)
        self.records[:, 0, 1].copy_(self.sequence[0].expand(2))
        self.records[:, 0, 4].fill_(-1)
        self.records[:, 0, 5].fill_(1)
        self.records[:, 0, 12].fill_(-1)
        self.records[:, 0, 15].fill_(1)
        self.records[:, 0, 16:].fill_(-1)


@pytest.mark.parametrize('tokens', [1, 64, 4096])
def test_iq4_rows_native_graph_delivery(tmp_path, tokens):
    caps = ple_embedding.Caps(device='cuda:0', max_tokens=tokens, max_seqs=1,
        vocab_size=256, eos_token_id=2, max_order=3, heads_per_order=8,
        dense_layer_ordinal=0, base_table_size=101, embedding_dim=2560, tp_size=1, tp_rank=0,
        quant_mode='bf16')
    hash_caps = ple_hash.Caps(**{field: getattr(caps, field) for field in ple_hash.Caps.__dataclass_fields__})
    geometry = ple_hash.compute_geometry(hash_caps)
    rows, heads = geometry.padded_vocab_size, hash_caps.head_count
    rng = np.random.default_rng(8917)
    raw = rng.integers(0, 256, (rows, 5, 18), dtype=np.uint8)
    raw[..., :2] = rng.uniform(-.01, .01, (rows, 5)).astype('<f2')[..., None].view(np.uint8)
    raw = raw.reshape(rows, 90)
    path = tmp_path / 'ple.gguf'
    writer = gguf.GGUFWriter(path, 'qwen4exp')
    writer.add_tensor('per_layer_token_embd.weight', raw, raw_dtype=gguf.GGMLQuantizationType.IQ4_NL)
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    tensor = gguf.GGUFReader(path).tensors[0]
    source = FileSource(path, tensor.data_offset, tensor.n_bytes)
    mailbox = MappedHostAllocation((tokens * heads + 1,), torch.int64, torch.device('cuda:0'))
    mailbox.host_view.zero_()
    ready = torch.cuda.Event(external=True); ready.record()
    frame = torch.zeros(1, dtype=torch.int32, device='cuda')
    sequence = torch.zeros(1, dtype=torch.int64, device='cuda')
    table = GgufPLETable({'vocab_size': 256}, GgufPLELayout(hash_caps, geometry),
        ple_hash.allocate_geometry(geometry, device='cuda:0'),
        torch.zeros((tokens * heads, 90), dtype=torch.uint8, device='cuda'),
        torch.full((tokens, heads), -1, dtype=torch.int64, device='cuda'), mailbox, ready, ((source,),),
        sequence, ReadinessTimings.allocate(frame, 1))
    metadata = PLEMetadata.allocate(tokens, 1, torch.device('cuda:0'), eos_token_id=2)
    output = torch.empty((tokens, 2560), dtype=torch.bfloat16, device='cuda')
    progress = torch.zeros(1, dtype=torch.int32, device='cuda')
    records = torch.zeros((2, 1, 32), dtype=torch.int64, device='cuda')
    worker = graph = None
    try:
        with PreparationSession(device='cuda:0', autotune=False, compile_workers=0) as session:
            lookup = PreparedPLELookup(table, metadata, output, caps, session=session, name='iq4-ple')
            forward = Forward(lookup, progress, records, sequence)
            signature = GraphSignature('prefill', 1, tokens, ('token_ids', 'query_start', 'num_seqs', 'num_tokens'))
            cache = ForwardCache(namespace={'model': 'iq4-native-delivery'}, device='cuda:0', root=tmp_path / 'cache')
            compiled = cache.prepare_many({'prefill': forward}, {'prefill': signature})['prefill']
            compiled(); torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph): compiled()
            buffers = dict(table.buffers(), output=output, token_ids=metadata.token_ids, query_start=metadata.query_start_loc,
                num_seqs=metadata.num_seqs, num_tokens=metadata.num_tokens, frame_index=frame,
                submission_sequence=sequence, completion_records=records)
            execution = PreparedExecution(device=0, graphs={'prefill': graph}, buffers=buffers,
                signatures={'prefill': signature}, max_active=1, resources=(table, lookup, compiled, session),
                export_events=table.events())
            resident = flare_heliopause.EmbeddingTables(table.resident_catalog())
            peer = flare_heliopause.EmbeddingTables(table.resident_catalog(), descriptors=resident.descriptors)
            del resident
            path.unlink()
            manifest = execution.manifest()
            manifest['engram'] = peer.bind(manifest, execution.export_events)
            worker = flare_heliopause.GraphExecutor(execution, manifest)
            cpu_table = torch.from_numpy(gguf.dequantize(raw, gguf.GGMLQuantizationType.IQ4_NL)).bfloat16()
            for generation, live in enumerate((tokens, min(tokens, 17), tokens // 2, 0, tokens), 1):
                ids = (torch.arange(tokens) * 17 + generation) % 256
                ids[0] = 2
                output.fill_(float('nan')); torch.cuda.synchronize()
                allocated = torch.cuda.memory_stats()['allocation.all.allocated']
                uploads = [('token_ids', 0, ids.numpy().tobytes()),
                    ('query_start', 0, torch.tensor([0, live], dtype=torch.int32).numpy().tobytes()),
                    ('num_seqs', 0, torch.tensor([int(live > 0)], dtype=torch.int32).numpy().tobytes()),
                    ('num_tokens', 0, torch.tensor([live], dtype=torch.int32).numpy().tobytes())]
                fence = worker.submit(0, uploads)
                deadline = time.monotonic() + 30
                completion = worker.poll()
                while completion is None and time.monotonic() < deadline:
                    time.sleep(.001); completion = worker.poll()
                assert completion is not None and completion.fence == fence
                assert torch.cuda.memory_stats()['allocation.all.allocated'] == allocated
                hashes = ple_hash_packed_reference(ids[:live], torch.tensor([0, live], dtype=torch.int32),
                    metadata.committed_history.cpu(), eos_token_id=2, multipliers=table.geometry.multipliers.cpu(),
                    prime_sizes=table.geometry.prime_sizes.cpu(), table_offsets=table.geometry.table_offsets.cpu(), heads_per_order=8)
                assert torch.equal(table.hashes[:live].cpu(), hashes)
                assert torch.equal(output[:live].cpu(), cpu_table[hashes].flatten(1))
                assert torch.count_nonzero(output[live:]).item() == 0
                assert lookup.error.item() == 0
    finally:
        if worker is not None: worker.close()
        if graph is not None: graph.reset()
        table.close()
