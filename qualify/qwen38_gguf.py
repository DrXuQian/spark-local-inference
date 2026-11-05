# SPDX-License-Identifier: Apache-2.0

"""Export real GGUF blocks through Flare's existing request and MTP machinery.

This is a qualification fixture, not a smaller language model. Target blocks
44/47 and the complete shared MTP use checkpoint weights. Vocabulary endpoints
and the small IQ4_NL lookup table are deterministic test data. Native scheduling,
sampling, speculative transactions, cache maintenance and graph families remain
the ordinary Qwen implementation.
"""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import gguf
import numpy as np
import torch

from ignition.components.embedding import PreparedEmbedding
from ignition.components.gated_delta import PreparedGDN
from ignition.components.gguf_weights import GgufWeightLoader
from ignition.components.linear import UnquantizedProjection
from ignition.components.ngram_delivery import PreparedPLELookup
from ignition.components.parallel import TensorParallelShard
from ignition.components.sampling import PreparedSampling
from ignition.preparation.gguf import (GgufCheckpoint, GgufShardMetadata, GgufTensorMetadata,
                                      file_identity, inspect_gguf_checkpoint)
from ignition.recipes.qwen38.attention import PreparedAttention
from ignition.recipes.qwen38.gguf_weights import QwenGgufWeightLoader
from ignition.recipes.qwen38.hyperconnection import PreparedHyperconnection
from ignition.recipes.qwen38.moe import PreparedQwenMoe


def component_checkpoint(path, metadata_path, layers):
    path, metadata_path = Path(path).resolve(), Path(metadata_path).resolve()
    reader = gguf.GGUFReader(path)
    headers = gguf.GGUFReader(metadata_path)
    metadata = {key: tuple(value) if isinstance(value := field.contents(), list) else value
                for key, field in headers.fields.items() if key.startswith(('general.', 'qwen4exp.'))}
    inventory = {}
    for tensor in reader.tensors:
        if not any(tensor.name.startswith(f'blk.{layer}.') for layer in layers):
            continue
        shape = tuple(int(n) for n in reversed(tensor.shape))
        block, size = gguf.GGML_QUANT_SIZES[tensor.tensor_type]
        inventory[tensor.name] = GgufTensorMetadata(tensor.name, path.name, tensor.data_offset,
            tensor.n_bytes, shape, tensor.tensor_type.name if block == 1 else 'U8',
            tensor.tensor_type.name, shape if block == 1 else (*shape[:-1], shape[-1] // block, size),
            'decoder_experts' if '_exps.' in tensor.name else 'decoder_other')
    digest = hashlib.sha256(reader.data[:reader.data_offset]).hexdigest()
    return GgufCheckpoint(path.parent, MappingProxyType(inventory),
        (GgufShardMetadata(path.name, path.stat().st_size, reader.data_offset, digest,
                           len(inventory), file_identity(path)),), MappingProxyType(metadata), 248320)


def fixture_rows(directory):
    """Bounded synthetic endpoints; all real block weights come from the shard."""
    path = directory / 'fixture-vocabulary.gguf'
    if not path.exists():
        rng = np.random.default_rng(1337)
        seed = rng.normal(0, .1, (32, 2560)).astype(np.float32)
        raw = np.tile(gguf.quantize(seed, gguf.GGMLQuantizationType.Q8_0), (248320 // 32, 1))
        writer = gguf.GGUFWriter(path, 'qwen4exp')
        writer.add_tensor('token_embd.weight', raw, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
        writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    with GgufWeightLoader(inspect_gguf_checkpoint(path), 'cuda:0', read_mode='pread') as loader:
        embedding = loader.load_quantized('token_embd.weight')
        head = loader.load_dequantized('token_embd.weight')
        # Keep the real vocabulary projection size, with distinct active logits.
        # Repeated head rows would make greedy tie selection obscure parity.
        head[16:32].copy_(-head[:16])
        head[32:].zero_()
        return embedding, head


def fixture_table(directory, tokens, storage, sequence, frame, *, external=False):
    from ignition.components.readiness import ReadinessTimings
    from ignition.preparation.runtime_package import FileSource
    from ignition.recipes.qwen38.gguf_ple import GgufPLECaps, GgufPLEGeometry, GgufPLELayout, GgufPLETable

    if external:
        from actlize_la.serving import ple_geometry

        caps = hash_caps = GgufPLECaps(torch.device('cuda:0'), 248044)
        geometry = ple_geometry(vocab_size=248320, base_size=101)
        geometry_buffers = GgufPLEGeometry.allocate(geometry, 'cuda:0')
    else:
        from b12x.sequence import ple_embedding, ple_hash

        caps = ple_embedding.Caps(device='cuda:0', max_tokens=tokens, max_seqs=1, vocab_size=248320,
            eos_token_id=248044, max_order=3, heads_per_order=8, dense_layer_ordinal=0,
            base_table_size=101, embedding_dim=2560, tp_size=1, tp_rank=0, quant_mode='bf16')
        hash_caps = ple_hash.Caps(**{field: getattr(caps, field) for field in ple_hash.Caps.__dataclass_fields__})
        geometry = ple_hash.compute_geometry(hash_caps)
        geometry_buffers = ple_hash.allocate_geometry(geometry, device='cuda:0')
    path = directory / 'fixture-ple.gguf'
    if not path.exists():
        rng = np.random.default_rng(225)
        raw = rng.integers(0, 256, (geometry.padded_vocab_size, 5, 18), dtype=np.uint8)
        raw[..., :2] = np.full((*raw.shape[:-1], 1), .0001, dtype='<f2').view(np.uint8)
        writer = gguf.GGUFWriter(path, 'qwen4exp')
        writer.add_tensor('per_layer_token_embd.weight', raw.reshape(-1, 90), raw_dtype=gguf.GGMLQuantizationType.IQ4_NL)
        writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    tensor = gguf.GGUFReader(path).tensors[0]
    rows = tokens * hash_caps.head_count
    mailbox = storage.host_allocate('ple_requests', (rows + 1,), torch.int64)
    mailbox.host_view.zero_()
    ready = torch.cuda.Event(external=True); ready.record()
    table = GgufPLETable({'vocab_size': 248320}, GgufPLELayout(hash_caps, geometry),
        geometry_buffers,
        torch.zeros((rows, 90), dtype=torch.uint8, device='cuda'),
        torch.full((tokens, hash_caps.head_count), -1, dtype=torch.int64, device='cuda'), mailbox, ready,
        ((FileSource(path, tensor.data_offset, tensor.n_bytes),),), sequence, ReadinessTimings.allocate(frame, 1))
    return table, caps


class PartialDecoder:
    """Real blocks with fixture endpoints; called by the existing round owner."""
    def __init__(self, weights, scratch, state, cache, rotary, table, workspace, completion, frame, sequence,
                 *, session, name, max_cache_pages, residual_output=None, verification=False,
                 gdn_records=None, gdn_in_place=False, sampling_workspace=None,
                 actlize_library=None, actlize_qsa_library=None, gguf_projection_library=None, **collectives):
        self.state, self.weights = state, weights
        self.scratch = s = replace(scratch, sample_hidden=scratch.hidden) if verification else scratch
        self.residual_output = residual_output
        self.embedding = PreparedEmbedding(weights.embedding, state.ple.token_ids, s.embedding,
            num_rows=state.gdn.num_tokens, session=session, name=name + '.embedding')
        self.ple_output = torch.zeros_like(s.embedding)
        self.lookup = PreparedPLELookup(table, state.ple, self.ple_output, weights.ple_caps,
                                       session=session, name=name + '.ple')
        source = s.hyperconnection.block_input
        self.layers = []
        self.gdn_operations = {}
        for index, module, attention_weights, moe_weights in weights.blocks:
            prefix = f'{name}.layer{index}'
            ah = PreparedHyperconnection(module.attn_hyper_connection, s.hyperconnection,
                priming_state=s.residual, session=session, name=prefix + '.attn_hc')
            mh = PreparedHyperconnection(module.mlp_hyper_connection, s.hyperconnection,
                priming_state=s.residual, session=session, name=prefix + '.mlp_hc')
            if index % 4 == 3:
                attention = PreparedAttention(module.self_attn, attention_weights, source, s.qsa,
                    cache[index], state.qsa, rotary, session=session, name=prefix + '.qsa',
                    workspace=workspace, projection_workspace=workspace, max_cache_pages=max_cache_pages, draft_tokens=7,
                    actlize_library=actlize_qsa_library, live_rows=state.gdn.num_seqs)
            else:
                attention = PreparedGDN(module.linear_attn, attention_weights, source, s.gdn,
                    state.pool.gdn[index], state.gdn, mode=state.mode, session=session, name=prefix + '.gdn',
                    checkpoint_export=False, workspace=workspace, projection_workspace=workspace,
                    records=None if gdn_records is None else gdn_records[index], in_place=gdn_in_place,
                    actlize_library=actlize_library)
                self.gdn_operations[index] = attention
            moe = PreparedQwenMoe(source, moe_weights, session=session, name=prefix + '.moe',
                scratch=s.moe, workspace=s.gguf_experts, projection_workspace=workspace,
                gguf_projection_library=gguf_projection_library)
            ah.prepare_combine_norm(mh, priming_state=s.residual, session=session, name=prefix + '.attn_combine')
            self.layers.append((ah, attention, mh, moe))
        self.final = PreparedHyperconnection(weights.final_mixer, replace(s.hyperconnection, block_input=s.hidden),
            priming_state=s.residual, session=session, name=name + '.final')
        for index, (_, _, mh, _) in enumerate(self.layers[:-1]):
            if index == 0:
                continue  # The lookup contribution precedes the next normalization.
            mh.prepare_combine_norm(self.layers[index + 1][0], priming_state=s.residual,
                                    session=session, name=f'{name}.combine{index}')
        state.prepare_hidden(s.hidden, s.sample_hidden); state.prepare_residual(s.residual)
        self.head = UnquantizedProjection(s.sample_hidden, weights.head, s.logits, session=session, name=name + '.head')
        self.sampler = None
        if not verification:
            inputs = state.inputs
            self.sampler = PreparedSampling(s.logits, state.requests, inputs.row_slot_ids[:state.rows],
                inputs.row_generations[:state.rows], inputs.live_rows, completion, frame, sequence,
                sample_enabled=inputs.sample_enabled[:state.rows], workspace=sampling_workspace)
            self.sampler.prepare()

    def __call__(self):
        s = self.scratch
        self.state.begin(); self.lookup.publish(); self.embedding()
        s.residual.view(-1, 4, 2560).copy_(s.embedding[:, None])
        for index, (ah, attention, mh, moe) in enumerate(self.layers):
            if index == 1:
                self.lookup(s.residual)
                s.residual.view(-1, 4, 2560).add_(self.ple_output[:, None])
            block = attention(ah.mix(s.residual, pre_normalized=index > 1))
            ah.combine_norm(s.residual, block)
            block = moe(mh.mix(s.residual, pre_normalized=True))
            if index > 0 and index + 1 < len(self.layers):
                mh.combine_norm(s.residual, block)
            else:
                mh.combine(s.residual, block)
        if self.residual_output is not None:
            self.residual_output.copy_(s.residual)
        self.final.mix(s.residual); self.state.finish(); self.head(s.sample_hidden)
        if self.sampler is not None:
            self.sampler()

    @property
    def device_errors(self):
        from ignition.components.operator_errors import device_errors

        return device_errors(self.embedding, self.lookup, self.final,
            *(operation for layer in self.layers for operation in layer))


@torch.no_grad()
def prepare(args):
    from transformers import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextDecoderLayer
    from ignition.components.completion import CompletionOutputs
    from ignition.components.sampling import AdmissionBuffers
    from ignition.preparation.checkpoint import CheckpointManifest
    from ignition.preparation.blas import BlasLibrarySpec
    from ignition.preparation.qwen38 import CONFIG_SHA256
    from ignition.preparation.capture import capture_forwards
    from ignition.preparation.contract import history_region
    from ignition.preparation.execution import GraphSignature
    from ignition.preparation.frontend import annotated_template_asset, prepare_frontend
    from ignition.preparation.model import PreparedModel
    from ignition.preparation.module_images import ModuleImages
    from ignition.preparation.package_storage import PackageStorage
    from ignition.preparation.runtime_package import export_model
    from ignition.preparation.scratch import ScratchArena
    from ignition.recipes.qwen38.tuning import external_operators, preparation_session
    from ignition.preparation import runtime_package_pb2 as p
    from ignition.recipes.qwen38.attention import QSAState, RotaryTables
    from ignition.recipes.qwen38.contract import prepare as contract
    from ignition.recipes.qwen38.decoder import DecoderScratch
    from ignition.recipes.qwen38.decoder_state import DecoderInputs, PreparedDecoderState, PreparedResume
    from ignition.recipes.qwen38.maintenance import PreparedMaintenance
    from ignition.recipes.qwen38.mtp_weights import MTPWeights
    from ignition.recipes.qwen38.recipe import RECIPE
    from ignition.recipes.qwen38.sampling import DeviceRequestState
    from ignition.recipes.qwen38.speculative_model import SpeculativePreparation, PreparedSpeculativeRound
    from ignition.recipes.qwen38.state import HybridStatePool, StateTransferMetadata
    from helios.tests.signing import publisher

    output, frontend = args.output.resolve(), args.frontend.resolve()
    if args.layers not in ([44, 47], [44, 45, 46, 47]):
        raise ValueError('qualification supports target blocks [44,47] or [44,45,46,47]')
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'package').exists():
        raise FileExistsError(output / 'package')
    if args.actlize_qsa and args.actlize_library is None:
        raise ValueError('--actlize-qsa requires an explicit --actlize-library')
    if args.operator_target.startswith('ppu'):
        missing = [name for name in ('actlize_library', 'quactlize_abi_library', 'quactlize_helpers_library',
                                     'dense_blas_library', 'flash_attention3_library') if getattr(args, name) is None]
        if missing or not args.actlize_qsa:
            raise ValueError('PPU qualification requires --actlize-qsa and all external libraries; missing '
                             + ', '.join('--' + name.replace('_', '-') for name in missing))
    if (args.dense_blas_library is None) != (args.dense_blas_api is None):
        raise ValueError('--dense-blas-library and --dense-blas-api must be supplied together')
    dense_blas = None if args.dense_blas_library is None else BlasLibrarySpec.read(Path.cwd(),
        dict(library=str(args.dense_blas_library), api=args.dense_blas_api,
             workspace_bytes=args.dense_blas_workspace_bytes))
    raw = (frontend / 'config.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != CONFIG_SHA256:
        raise ValueError('qualification requires the pinned Qwen frontend config')
    config = Qwen4ExpTextConfig.from_dict(json.loads(raw)['text_config'])
    structure = CheckpointManifest(frontend, {}, (), {'config.json': CONFIG_SHA256}, '', {})
    manifest = component_checkpoint(args.target_shard, args.metadata, args.layers)
    with ModuleImages() as images, PackageStorage(device=0) as storage:
        torch.cuda.set_device(0)
        print('Loading fixture vocabulary and actual GGUF blocks', flush=True)
        embedding, head = fixture_rows(output)
        blocks = []
        with QwenGgufWeightLoader(manifest, 'cuda:0', read_mode='pread') as loader:
            for index in args.layers:
                with torch.device('meta'):
                    module = Qwen4ExpTextDecoderLayer(config, index)
                module.eval().requires_grad_(False)
                for label, field in (('attn', 'attn_hyper_connection'), ('ffn', 'mlp_hyper_connection')):
                    loader.load_hyperconnection(getattr(module, field), f'blk.{index}.hc_{label}')
                attention = (loader.load_attention(module.self_attn, index) if index % 4 == 3
                             else loader.load_gdn(module.linear_attn, index))
                blocks.append((index, module, attention, loader.load_moe(index, cache_shared=True)))
                print(f'Loaded complete target block {index}', flush=True)
        mtp = MTPWeights.load_gguf(inspect_gguf_checkpoint(args.mtp), structure, 'cuda:0',
            embedding_weight=embedding, lm_head_weight=head, read_mode='pread')
        print('Loaded shared-Q8 MTP', flush=True)
        frame = torch.zeros(1, dtype=torch.int32, device='cuda')
        sequence = torch.zeros(1, dtype=torch.int64, device='cuda')
        table, ple_caps = fixture_table(output, 64, storage, sequence, frame,
            external=args.actlize_library is not None and args.quactlize_helpers_library is not None)
        from ignition.preparation.actlize import ActlizeLibrarySpec
        from ignition.preparation.quactlize import QuactlizeHelpersSpec, QuactlizeLibrarySpec
        from ignition.preparation.flash_attn3 import FlashAttention3Spec
        from ignition.preparation.qwen38_gguf import GgufInputs

        target = args.operator_target
        quant_target = 'nvidia' if target == 'cuda_sm120' else target
        operator_inputs = GgufInputs(args.target_shard, args.mtp, args.frontend, 'pread',
            None if args.actlize_library is None else ActlizeLibrarySpec.read(args.actlize_library, target=target),
            None if args.quactlize_abi_library is None else QuactlizeLibrarySpec.read(args.quactlize_abi_library, target=quant_target),
            dense_blas=dense_blas,
            flash_attention3=None if args.flash_attention3_library is None else
                FlashAttention3Spec.read(args.flash_attention3_library),
            quactlize_helpers=None if args.quactlize_helpers_library is None else
                QuactlizeHelpersSpec.read(args.quactlize_helpers_library, target=quant_target))
        external = external_operators(operator_inputs)
        if external and not args.actlize_qsa:
            raise ValueError('complete external operator qualification requires --actlize-qsa')
        attention_libraries = operator_inputs.load_operators('cuda:0')
        dense_blas_library = attention_libraries.pop('dense_blas')
        quactlize_helpers = attention_libraries.pop('quactlize_helpers')
        flash_attention3 = attention_libraries.pop('flash_attention3')
        operators = operator_inputs.operator_identity() or {'gdn': 'b12x'}
        kv_dtype = torch.bfloat16 if args.actlize_qsa or flash_attention3 is not None else torch.float8_e4m3fn
        operators.update(qsa='actlize' if args.actlize_qsa else 'b12x', kv_dtype=str(kv_dtype))
        if not args.actlize_qsa:
            attention_libraries['actlize_qsa_library'] = None
        projection_library = attention_libraries['gguf_projection_library']
        weights = SimpleNamespace(blocks=tuple(blocks), embedding=embedding, head=head,
            final_mixer=mtp.module.hyper_connection_mixer, ple_caps=ple_caps, shard=TensorParallelShard(0, 1))
        state = HybridStatePool.allocate(5, 'cuda:0', speculative_tokens=7, storage=storage, kv_dtype=kv_dtype)
        state = replace(state, gdn=MappingProxyType({i: state.gdn[i] for i in args.layers if i % 4 != 3}))
        requests = DeviceRequestState.allocate('cuda:0', capacity=1)
        inputs = DecoderInputs.allocate('cuda:0', context=1024, tokens=64, capacity=1)
        scratch = DecoderScratch.allocate(64, torch.device('cuda:0'), sample_rows=8, gguf_experts=blocks[0][3].experts,
            gguf_projection_library=projection_library)
        workspace = ScratchArena('cuda:0')
        rotary = RotaryTables.allocate(config, 1024, 'cuda:0')
        # Snapshot rows borrow KV backing in native serving. Reserve enough
        # pages for the canonical full-state row even with only two live blocks.
        pages = 512
        cache = {i: QSAState.allocate(pages=1, slots=2, device='cuda:0', draft_tokens=7, kv_dtype=kv_dtype)
                 .resize_pages_for_preparation(pages, storage=storage, prefix=f'kv-{i}-', backed_pages=64)
                 for i in (*range(3, 48, 4), 48)}
        state.memory_capacity[1] = 64
        completion = CompletionOutputs.allocate((2, 1, 24), device='cuda:0')
        admission, transfer = AdmissionBuffers.allocate('cuda:0', vocab_size=248320, capacity=1), StateTransferMetadata.allocate(1, 'cuda:0')
        buffers = {**inputs.buffers(), **table.buffers(), **completion.buffers(),
            **{n: getattr(admission, n) for n in admission.__dataclass_fields__},
            **{f'state_{n}': getattr(transfer, n) for n in transfer.__dataclass_fields__},
            'frame_index': frame, 'submission_sequence': sequence, 'state_pool': state.backing,
            'memory_capacity': state.memory_capacity, 'sampling_grammar_mask': requests.grammar_mask,
            'request_pending_token': requests.pending_token}
        functions, signatures, owners = {}, {}, []
        with preparation_session(device='cuda:0', checkpoint='qwen38-gguf-partial',
                gguf_inputs=operator_inputs,
                namespace={'fixture': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           'layers': args.layers, 'target_header': manifest.shards[0].header_sha256,
                           'operators': operators},
                autotune_effort='off', compile_workers=2) as session:
            session.quactlize_helpers = quactlize_helpers
            session.flash_attention3 = flash_attention3
            session.actlize_helpers = attention_libraries['actlize_library']
            session.dense_blas = None
            if dense_blas_library is not None:
                from ignition.components.blas_projection import PreparedBlas

                session.dense_blas = PreparedBlas(dense_blas_library,
                    torch.empty(dense_blas.workspace_bytes, dtype=torch.uint8, device='cuda:0'))
            speculation = SpeculativePreparation(weights=weights, mtp_weights=mtp, pool=state, requests=requests,
                inputs=inputs, scratch=scratch, workspaces=workspace, rotary=rotary, cache=cache, table=table,
                session=session, completion=completion, frame=frame, sequence=sequence, context=1024,
                prefill_capacity=64, max_cache_pages=pages, verification_width=6, **attention_libraries)
            buffers.update(speculation.buffers())
            fields = (*inputs.buffers(), 'sampling_grammar_mask', *speculation.buffers())
            def add(name, forward, kind, tokens, names, enabled=True):
                functions[name] = forward
                signatures[name] = GraphSignature(kind, 1, tokens, tuple(names), speculation_enabled=enabled)
            for width in (1, 2, 4, 6):
                name = f'speculative1x{width}'
                op = PreparedSpeculativeRound(speculation, 1, width, name, decoder_factory=PartialDecoder)
                op.prepare(); speculation.owners.append(op)
                add(name, op, 'speculative', width, fields)
            metadata = PreparedDecoderState(inputs, requests, state, tokens=64, rows=1, mode='prefill', maintain_draft_state=True)
            decoder = PartialDecoder(weights, scratch.rows(64, 1), metadata, cache, rotary, table, workspace,
                completion, frame, sequence, session=session, name='prefill64', max_cache_pages=pages, **attention_libraries)
            add('prefill64', speculation.prefill(decoder, 'prefill64'), 'prefill', 64, fields)
            plain_fields = (*inputs.buffers(), 'sampling_grammar_mask')
            for mode, tokens in (('prefill', 64), ('decode', 1)):
                metadata = PreparedDecoderState(inputs, requests, state, tokens=tokens, rows=1,
                    mode=mode, maintain_draft_state=False)
                name = f'{mode}{tokens}' + ('-no-mtp' if mode == 'prefill' else '')
                plain = PartialDecoder(weights, scratch.rows(tokens, 1), metadata, cache, rotary, table,
                    workspace, completion, frame, sequence, session=session, name=name, max_cache_pages=pages,
                    **attention_libraries)
                owners.append(plain)
                add(name, plain, mode, tokens, plain_fields, enabled=False)
            resume = PreparedResume(inputs, requests, state, scratch.sample_hidden[:1], scratch.logits[:1], head,
                completion, frame, sequence, session=session, name='resume1')
            add('resume1', resume, 'resume', 1, fields)
            add('resume1-no-mtp', resume, 'resume', 1, plain_fields, enabled=False)
            state_fields = tuple(f'state_{n}' for n in transfer.__dataclass_fields__ if n != 'errors')
            for mode in ('init', 'restore', 'snapshot'):
                op = PreparedMaintenance(mode, requests, state, transfer, completion, frame, sequence,
                    admission_buffers=admission, forward_inputs=inputs, capacity=1, speculation=speculation,
                    qsa_states=tuple(cache[i] for i in range(3, 48, 4)))
                op.prepare(); owners.append(op)
                names = state_fields + (fields if mode == 'snapshot' else ('request_state_rows', *admission.__dataclass_fields__))
                if mode == 'restore': names += ('page_tables', 'draft_page_tables')
                add(mode + '1', op, mode, 1, names)
                plain = PreparedMaintenance(mode, requests, state, transfer, completion, frame, sequence,
                    admission_buffers=admission, forward_inputs=inputs, capacity=1,
                    qsa_states=tuple(cache[i] for i in range(3, 48, 4)))
                plain.prepare(); owners.append(plain)
                names = state_fields + (plain_fields if mode == 'snapshot' else ('request_state_rows', *admission.__dataclass_fields__))
                if mode == 'restore': names += ('page_tables',)
                add(mode + '1-no-mtp', plain, mode, 1, names, enabled=False)
            session.register_scratch(workspace)
            session.tune(show_progress=False)
            print('Compiling complete prefill/MTP/maintenance forwards', flush=True)
            from ignition.components.operator_errors import guard_forwards

            functions = guard_forwards(functions, requests, completion, frame, sequence)
            stream = torch.cuda.Stream()
            graphs, compiled = capture_forwards(functions, signatures, session=session, device=torch.device('cuda:0'),
                stream=stream, pool=storage.graph_pool.id, retain_graphs=True,
                cache_root=output / 'compiled')
            session.freeze(); state.reset_for_preparation()
            regions = [history_region(buffers, f'cache.{i}.{name}', getattr(value, name), p.PAGED_KEY_VALUE,
                tokens_per_record=4 if name == 'compressed_k_cache' else 1)
                for i, value in cache.items() for name in ('main_k_cache', 'main_v_cache', 'compressed_k_cache')]
            identity = hashlib.sha256((manifest.shards[0].header_sha256
                + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
                + json.dumps(operators, sort_keys=True)).encode()).hexdigest()
            settings = dict(model_id='qwen38-gguf-partial-qualification', identity=identity,
                max_sequence_tokens=1024, prefill_chunk_size=64, pages=64, state_rows=4, snapshots=3, page_tokens=64,
                frontend=prepare_frontend(frontend, eos_token_ids=[248046, 248044], image_profile=None),
                sampling_defaults=RECIPE.sampling_defaults.as_dict(), speculative_widths=(1, 2, 4, 6),
                speculative_token_capacity=6,
                model_contract=contract(signatures, buffers, state, page_tokens=64, draft_lag=1, embedding_tables=(0,), regions=regions),
                assets={'tokenizer': dict(path=str(frontend / 'tokenizer.json'), sha256=hashlib.sha256((frontend / 'tokenizer.json').read_bytes()).hexdigest())})
            settings['assets']['chat_template'] = annotated_template_asset((frontend / 'chat_template.jinja').read_bytes(),
                Path(__file__).resolve().parents[1] / 'ignition/recipes/qwen38/assets/chat_template.patch.json')
            from ignition.recipes.qwen38.export import disposable_ranges
            zero = disposable_ranges(cache=cache, scratch=scratch, workspaces=workspace, vision_scratch={},
                decoders=[decoder, *speculation.decoders], owners=owners, speculation=speculation)
            if session.dense_blas is not None:
                from ignition.preparation.disposable import storage_ranges

                zero = (*zero, *storage_ranges((session.dense_blas.workspace,)))
            prepared = PreparedModel(device=0, graphs=graphs, buffers=buffers, signatures=signatures, max_active=1,
                session=session, server_config=settings, memory_report={}, package_data=table, export_events=table.events(),
                export_zero_ranges=(*zero, *completion.zero_ranges()),
                close_resources=() if quactlize_helpers is None else (quactlize_helpers,),
                resources=(weights, mtp, table, state, requests, inputs, scratch, workspace, rotary, cache,
                           completion, admission, transfer, functions, compiled, owners, speculation, stream, storage))
            print('Exporting native qualification package', flush=True)
            export_model(prepared, images, storage, output / 'package', options={}, publisher=publisher())
            print(f'Exported {output / "package"}; target blocks={args.layers}, shared-Q8 MTP, synthetic endpoints', flush=True)
            prepared.close(); table.close()


if __name__ == '__main__':
    import argparse
    from qualify.qwen38_gguf import prepare

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operator-target', choices=('cuda_sm120', 'ppu10', 'ppu15'), default='cuda_sm120',
                        help='actual target of the external QUactlize and actlizeLA libraries')
    parser.add_argument('--quactlize-helpers-library', type=Path,
                        help='explicit QUactlize serving v2/v1 DSO, independent of the matmul DSO')
    parser.add_argument('--flash-attention3-library', type=Path,
                        help='installed flash_attn_3._C extension for BF16 ordinary MTP attention')
    for name in ('target-shard', 'metadata', 'mtp', 'frontend', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--layers', type=int, nargs='+', default=[44, 47])
    parser.add_argument('--actlize-library', type=Path,
                        help='qualify the explicitly selected actlizeLA GDN library inside complete serving graphs')
    parser.add_argument('--actlize-qsa', action='store_true',
                        help='also bind external QSA, with BF16 KV pages and matching snapshots')
    parser.add_argument('--quactlize-abi-library', type=Path,
                        help='qualify the explicit native matmul-record ABI implementation')
    parser.add_argument('--dense-blas-library', type=Path,
                        help='use this cuBLAS-compatible library for unquantized projections')
    parser.add_argument('--dense-blas-api', choices=('cublas', 'acblas'),
                        help='C symbol prefix of the explicitly selected BLAS library')
    parser.add_argument('--dense-blas-workspace-bytes', type=int, default=4 << 20,
                        help='caller-owned shared BLAS scratch, default 4 MiB; must be 256-byte aligned')
    prepare(parser.parse_args())
