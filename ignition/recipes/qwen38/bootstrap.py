# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepare the complete Qwen graph catalog before starting native serving."""

from __future__ import annotations

import contextlib
import hashlib
import json
from pathlib import Path

import torch

from ignition.components.completion import CompletionOutputs, VECTOR_TOKENS_FIELD
from ignition.components.prompt_logprobs import (
    PreparedScoredPrefill, PromptScoring, prefill_graph_count, prefill_modes,
)

from ignition.preparation.tuning import with_autotune_effort

from ignition.components.parallel import TensorParallelShard
from ignition.preparation.collectives import (
    CollectivePreparation,
    create_all_gather_runtime,
    create_all_reduce_runtime,
    create_roce_runtime,
)
from ignition.components.capacity import speculative_token_capacity
from ignition.components.sampling import AdmissionBuffers
from ignition.preparation import inspect_qwen38_checkpoint
from ignition.preparation.capture import capture_forwards
from .contract import prepare as prepare_contract
from .config import PreparationConfig
from ignition.preparation.contract import history_region
from ignition.preparation.rebuild_manifest import declare_weights
from ignition.preparation import runtime_package_pb2 as contract_proto
from ignition.preparation.execution import GraphSignature
from ignition.preparation.host_memory import device_memory_info
from ignition.preparation.deferred import deferred_mode
from ignition.preparation.fixtures.recurrent import register_benchmark_state
from ignition.preparation.frontend import annotated_template_asset, prepare_frontend
from ignition.preparation.model import PreparedModel
from ignition.preparation.rank_group import minimum_across_ranks
from ignition.preparation.qwen38 import (
    FRONTEND_SHA256,
    MODEL_ID,
    REVISION,
)
from ignition.preparation.scratch import GraphBufferPool, ScratchArena, bind_scratch
from ignition.progress import phase

from .recipe import RECIPE
from .attention import QSAState, RotaryTables
from .decoder import DecoderScratch, ModelWeights, PreparedDecoder
from .decoder_state import DecoderInputs, PreparedDecoderState, PreparedResume
from .maintenance import PreparedMaintenance
from .ple import PLETable
from .sampling import DeviceRequestState
from .state import (
    HybridStateLayout,
    HybridStatePool,
    StateTransferMetadata,
)
from .tuning import preparation_session
from .vision import PreparedVision, VisionScratch


def _identity(manifest, world_size=1, *, operators=None):
    sources = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted(Path(__file__).parent.glob("*.py"))}
    components = Path(__file__).parents[2] / "components"
    sources.update({f"components/{path.name}": hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(components.glob("*.py"))})
    value = {"headers": manifest.headers_sha256,
             "config": manifest.metadata_sha256["config.json"], "recipe": sources, "kv_dtype": "float8_e4m3fn", "page_size": 64, "world_size": world_size}
    if operators is not None:
        value.update(operators=operators, kv_dtype=operators['kv_dtype'])
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def prepare_rank(snapshot, config, *, rank, group, progress=None, package_storage=None):
    return prepare_model(snapshot, config, progress=progress, compile_decoder=config.compile_decoder,
                         rank=rank, group=group, package_storage=package_storage)


@torch.no_grad()
def prepare_model(snapshot: Path, config: PreparationConfig | None = None, *, progress=None,
                  compile_decoder: bool = True, rank: int = 0, group=None, package_storage=None) -> PreparedModel:
    config = PreparationConfig() if config is None else config
    transient = contextlib.ExitStack()
    try:
        prepared = _prepare_model(snapshot, config, progress=progress, compile_decoder=compile_decoder,
            rank=rank, group=group, package_storage=package_storage, transient=transient)
    except BaseException as error:
        try:
            transient.close()
        except BaseException as cleanup_error:
            error.add_note(f"transient allocation scope cleanup failed: {cleanup_error!r}")
        raise
    else:
        transient.close()
        return prepared


@with_autotune_effort
def _prepare_model(snapshot: Path, config: PreparationConfig | None = None, *, transient,
                   progress=None, compile_decoder: bool = True, rank: int = 0,
                   group=None, package_storage=None) -> PreparedModel:
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpVisionModel

    from .vision_state import (
        PreparedEmbeddingInjection,
        PreparedVisionGeneration,
        PreparedVisionHandoff,
        VisionFeatureState,
        VisionInputs,
    )

    def report(message):
        if progress is not None:
            progress(message)

    config = PreparationConfig() if config is None else config
    shard = TensorParallelShard(rank, 1 if config.devices is None else len(config.devices))
    if (shard.world_size > 1) != (group is not None):
        raise ValueError("tensor parallel preparation requires an established control group")
    device = torch.device("cuda", config.device if config.devices is None else config.devices[rank])
    torch.cuda.set_device(device)
    with phase(progress, "Inspect checkpoint"):
        if config.gguf_inputs is None:
            manifest = inspect_qwen38_checkpoint(snapshot)
        else:
            from ignition.preparation.qwen38_gguf import inspect_qwen38_gguf

            manifest = inspect_qwen38_gguf(config.gguf_inputs)
    operator_libraries = (dict(actlize_library=None, actlize_qsa_library=None, gguf_projection_library=None)
        if config.gguf_inputs is None else config.gguf_inputs.load_operators(device))
    dense_blas_library = operator_libraries.pop('dense_blas', None)
    quactlize_helpers = operator_libraries.pop('quactlize_helpers', None)
    flash_attention3 = operator_libraries.pop('flash_attention3', None)
    operators = None if config.gguf_inputs is None else config.gguf_inputs.operator_identity()
    if config.actlize is not None:
        report('actlizeLA GDN/QSA · BF16 KV and snapshots · ' + config.actlize.target)
    if operator_libraries['gguf_projection_library'] is not None:
        report('QUactlize matmul records · ' + config.gguf_inputs.quactlize.target)
    if flash_attention3 is not None:
        report('FA3 ordinary MTP attention · BF16 page64/D256')
    if quactlize_helpers is not None:
        report('QUactlize serving helpers · router v2, rounded SwiGLU v2 and raw GGUF rows')
    # Loading, construction and tuning free most of what they allocate; a transient package pool returns
    # that memory to the device once tuning completes, before compiled forwards are bound.
    if package_storage is not None:
        transient.enter_context(package_storage.transient())
    weights = ModelWeights.load(manifest, device, progress=progress, shard=shard,
        shard_hyperconnections=config.shard_hyperconnections)
    if torch.cuda.get_device_properties(device).is_integrated:
        from ignition.preparation.host_memory import release_page_cache

        release_page_cache(manifest.shard_paths)
    mtp_weights = None
    has_mtp = config.mtp
    if has_mtp:
        from .mtp_weights import MTPWeights

        mtp_kwargs = dict(
            embedding_weight=weights.embedding_weight,
            lm_head_weight=weights.tree.lm_head.weight, progress=progress, shard=shard)
        if config.gguf_inputs is None:
            mtp_weights = MTPWeights.load(manifest, device,
                shard_hyperconnections=config.shard_hyperconnections, **mtp_kwargs)
        else:
            mtp_weights = MTPWeights.load_gguf(manifest.draft, manifest, device,
                read_mode=manifest.read_mode, **mtp_kwargs)
    host_storage = {} if package_storage is None else {"host_allocator": package_storage.host_allocate}
    if config.gguf_inputs is None:
        table = PLETable.load(manifest, device, tokens=config.token_capacity, progress=progress, **host_storage)
    else:
        from .gguf_ple import GgufPLETable

        table = GgufPLETable.load(manifest, device, tokens=config.token_capacity, progress=progress, **host_storage)
    sample_rows = max(batch * width for batch, width in config.speculative_shapes) if config.mtp else config.request_capacity
    sample_rows += bool(config.mixed_shapes)
    coordinator = CollectivePreparation(group) if group is not None else None
    ipc = None if package_storage is None else package_storage.peer_storage
    collective = vocabulary_collective = None
    reduction_width = 10240 if config.shard_hyperconnections else 2560
    if coordinator is not None and coordinator.spans_hosts:
        # Ranks on separate Sparks reduce and gather through one RoCEnante runtime: decode rows
        # one-shot, prefill chunks through the pipelined two-shot path.
        collective = vocabulary_collective = create_roce_runtime(group, device,
            max_tokens=max(config.request_capacity, sample_rows), pipe_tokens=config.token_capacity, hidden_size=reduction_width,
            gather_bytes=sample_rows * (248320 // shard.world_size) * 2, package_storage=package_storage, ring=config.roce_ring, ring_open=config.roce_chain)
    elif group is not None:
        collective = create_all_reduce_runtime(group, device, collective_tuning=config.collective_tuning, max_tokens=config.token_capacity, hidden_size=reduction_width, ipc=ipc)
        vocabulary_collective = create_all_gather_runtime(group, device, max_rows=sample_rows, columns=248320, ipc=ipc)
    collectives = dict(runtime=collective, vocabulary_runtime=vocabulary_collective, coordinator=coordinator)
    session = preparation_session(
        compile_workers=config.compile_workers, collective_tuning=config.collective_tuning, collective_ladder=config.collective_ladder, autotune_effort=config.autotune_effort, tuning_replica=config.tuning_replica,
        native_dense_activations=config.native_dense_activations, lock_gpu_clocks=config.lock_gpu_clocks,
        dense_blas_api=None if dense_blas_library is None else dense_blas_library.api,
        device=device, checkpoint=REVISION if config.gguf_inputs is None else manifest.headers_sha256,
        coordinator=coordinator,
        namespace={"config": manifest.metadata_sha256["config.json"], "headers": manifest.headers_sha256, "world_size": shard.world_size,
                   "shard_hyperconnections": config.shard_hyperconnections, "operators": operators},
        headroom_bytes=config.headroom_bytes, benchmark_storage=package_storage)
    session.__enter__()
    graphs = {}
    try:
        session.quactlize_helpers = quactlize_helpers
        session.flash_attention3 = flash_attention3
        session.actlize_helpers = operator_libraries['actlize_library']
        session.dense_blas = None
        if dense_blas_library is not None:
            from ignition.components.blas_projection import PreparedBlas

            with phase(progress, 'Prepare dense BLAS resources'):
                blas_workspace = torch.empty(config.gguf_inputs.dense_blas.workspace_bytes,
                                             dtype=torch.uint8, device=device)
                session.dense_blas = PreparedBlas(dense_blas_library, blas_workspace)
        with phase(progress, "Allocate model state"):
            speculative_tokens = 7 if config.mtp else 0
            layout = HybridStateLayout.build(speculative_tokens=speculative_tokens, world_size=shard.world_size,
                kv_dtype=config.kv_dtype)
            snapshots = layout.snapshot_capacity(config.snapshot_budget_bytes)
            slots = config.request_capacity
            # Verification commits into each request's live row, so a request holds one row.
            state_rows = snapshots + slots
            state = HybridStatePool.allocate(state_rows + 1, device,
                speculative_tokens=speculative_tokens, world_size=shard.world_size, storage=package_storage,
                kv_dtype=config.kv_dtype,
                backed_slots=(min(state_rows + 1, config.preparation_state_rows) if package_storage is not None else None))
            register_benchmark_state(
                session, next(iter(state.gdn.values())).recurrent,
                storage=state.backing, restore=state.reset_for_preparation,
            )
            requests = DeviceRequestState.allocate(device, capacity=slots)
            from ignition.components.sampling import PreparedSampling

            sampling_workspace = bind_scratch(PreparedSampling.scratch_layout(slots, requests.vocab_size), device=device)
            inputs = DecoderInputs.allocate(device, context=config.context, tokens=config.token_capacity, capacity=slots)
            graph_buffers = GraphBufferPool(device, chunk_bytes=2 << 20)
            scratch = DecoderScratch.allocate(config.token_capacity, device, sample_rows=sample_rows,
                world_size=shard.world_size, buffers=graph_buffers.cursor(),
                gguf_experts=weights.layers[0].moe.experts if config.gguf_inputs is not None else None,
                gguf_projection_library=operator_libraries['gguf_projection_library'],
                hyperconnection_lowrank=320 // shard.world_size if config.shard_hyperconnections else 320)
            workspaces = ScratchArena(device, overlap=config.stream_overlap)
            rotary = RotaryTables.allocate(weights.tree.config.text_config, config.context, device)
            cache = {index: QSAState.allocate(pages=1, slots=slots + 1, device=device, draft_tokens=speculative_tokens,
                     kv_heads=max(1, 2 // shard.world_size), kv_dtype=config.kv_dtype)
                     for index in (*range(3, 48, 4), *((48,) if has_mtp else ()))}
            page_bytes = sum(sum(getattr(value, name).nbytes for name in
                ("main_k_cache", "main_v_cache", "compressed_k_cache")) for value in cache.values())
            planned_pages = minimum_across_ranks(torch.cuda.mem_get_info(device)[1] // page_bytes, group)
            admission = AdmissionBuffers.allocate(device, vocab_size=248320, capacity=slots)
            transfer = StateTransferMetadata.allocate(slots, device)
            frame = torch.zeros(1, dtype=torch.int32, device=device)
            sequence = torch.zeros(1, dtype=torch.int64, device=device)
            table.sequence = sequence
            from ignition.components.readiness import ReadinessTimings
            table.timings = ReadinessTimings.allocate(frame, 1)
            completion = CompletionOutputs.allocate((2, slots, VECTOR_TOKENS_FIELD + config.completion_token_capacity), device=device)
            features = (None if weights.vision is None else VisionFeatureState.allocate(device,
                capacity=slots, image_slots=config.image_concurrency, feature_slot=inputs.vision_feature_slot))
            vision_inputs = None if features is None else VisionInputs.allocate(device)
            vision_scratch = {}
            buffers = {**inputs.buffers(), **table.buffers(), **{name: getattr(admission, name) for name in admission.__dataclass_fields__},
                **{f"state_{name}": getattr(transfer, name) for name in transfer.__dataclass_fields__},
                **({} if vision_inputs is None else {name: getattr(vision_inputs, name) for name in vision_inputs.__dataclass_fields__}),
                "frame_index": frame, "submission_sequence": sequence, **completion.buffers(),
                "sampling_grammar_mask": requests.grammar_mask, "state_pool": state.backing, "memory_capacity": state.memory_capacity}
            functions, signatures, decoders, owners = {}, {}, [], []
            speculation = None
            if config.mtp:
                from .speculative_model import SpeculativePreparation

                speculation = SpeculativePreparation(weights=weights, mtp_weights=mtp_weights,
                    pool=state, requests=requests, inputs=inputs, scratch=scratch, workspaces=workspaces,
                    rotary=rotary, cache=cache, table=table, session=session, completion=completion,
                    frame=frame, sequence=sequence, context=config.context,
                    prefill_capacity=config.token_capacity, max_cache_pages=planned_pages,
                    sampling_workspace=sampling_workspace, graph_buffers=graph_buffers,
                    verification_width=max(config.speculative_widths), **collectives, **operator_libraries)
                buffers.update(speculation.buffers())
        scoring = PromptScoring.allocate(config, device, buffers, frame=frame, group=group, ipc=ipc,
            runtime=collective, coordinator=coordinator)
        serving_modes = (True, False) if config.mtp else (False,)
        graph_count = ((4 if features is not None else 0) + len(serving_modes) * (prefill_graph_count(config.prefill_capacities, scoring)
                                              + 4 * len(config.batch_capacities))
                       + len(config.batch_capacities) + (len(config.speculative_shapes) if config.mtp else 0)
                       + len(config.mixed_shapes))
        with phase(progress, "Construct graph catalog", total=graph_count) as stage:
            def add(name, function, kind, batch, tokens, fields, *, enabled=None, prefill_capacity=0):
                if kind == "scored_prefill":
                    function = PreparedScoredPrefill(function, requests, inputs)
                    fields = (*fields, "prompt_target_tokens")
                functions[name] = function
                signatures[name] = GraphSignature(kind, batch, tokens, tuple(fields), speculation_enabled=enabled,
                                                  prefill_capacity=prefill_capacity)
                stage.update(advance=1)

            vision_fields = () if vision_inputs is None else (*vision_inputs.__dataclass_fields__, "vision_feature_slot")
            for capacity in (() if features is None else (16384, 4096, 1024, 256)):
                cursor = graph_buffers.cursor()
                s = vision_scratch[capacity] = VisionScratch.allocate(capacity, device, buffers=cursor)
                handoff = PreparedVisionHandoff(vision_inputs, features, requests, s.output, completion, frame, sequence,
                                               pixels=s.patches, buffers=cursor)
                handoff.prepare()
                with torch.device("meta"):
                    module = Qwen4ExpVisionModel(weights.tree.config.vision_config)
                module.eval().requires_grad_(False)
                encoder = PreparedVision(module, s, handoff.metadata, weights.vision, session=session,
                    name=f"vision{capacity}", projection_workspace=workspaces,
                    attention_workspace=workspaces)
                add(f"vision{capacity}", PreparedVisionGeneration(handoff, encoder), "vision", 1, capacity, vision_fields)
                owners.extend((handoff, encoder))

            for enabled in serving_modes:
                suffix = "-no-mtp" if config.mtp and not enabled else ""
                forward_fields = (*inputs.buffers(), "sampling_grammar_mask",
                                  *(("draft_page_tables",) if enabled and has_mtp else ()))
                modes = prefill_modes(config.prefill_capacities,
                    () if enabled else reversed(config.batch_capacities), scoring)
                for kind, mode, capacities in modes:
                    for capacity in capacities:
                        rows = 1 if mode == "prefill" else capacity
                        name = f"{kind}{capacity}" + (suffix if mode == "prefill" else "")
                        metadata = PreparedDecoderState(inputs, requests, state, tokens=capacity, rows=rows, mode=mode,
                            maintain_draft_state=enabled)
                        s = scratch.rows(capacity, rows)
                        injection = None
                        if mode == "prefill" and features is not None:
                            injection = PreparedEmbeddingInjection(features, requests, inputs.vision_token_indices[:capacity],
                                inputs.row_slot_ids, inputs.row_generations, inputs.query_lengths, inputs.live_rows,
                                s.embedding, rows=rows)
                            injection.prepare()
                        decoder = PreparedDecoder(weights, s, metadata, cache, rotary, table, workspaces,
                            completion, frame, sequence, session=session, name=name,
                            embedding_injection=injection, max_cache_pages=planned_pages,
                            sampling_workspace=sampling_workspace,
                            scoring=scoring if kind == "scored_prefill" else None, **collectives, **operator_libraries)
                        forward = speculation.prefill(decoder, name) if enabled else decoder
                        add(name, forward, kind, rows,
                            capacity if mode == "prefill" else 1,
                            forward_fields, enabled=enabled)
                        decoders.append(decoder)

            if speculation is not None:
                forward_fields = (*inputs.buffers(), "sampling_grammar_mask",
                                  *(("draft_page_tables",) if has_mtp else ()))
                fields = (*forward_fields, *(name for name in speculation.buffers() if name != "draft_page_tables"))
                for batch, width in sorted(config.speculative_shapes, key=lambda shape: shape[0] * shape[1], reverse=True):
                    name = f"speculative{batch}x{width}"
                    operation = speculation.round(batch, width, name)
                    operation.prepare()
                    add(name, operation, "speculative", batch, width, fields, enabled=True)
                decoders.extend(speculation.decoders)
                owners.append(speculation)

            if config.mixed_shapes:
                from types import SimpleNamespace
                from .mixed_model import PreparedMixedRound
                from .mixed_state import MixedScratch

                mixed_scratch = MixedScratch.allocate(config.prefill_chunk_size,
                    max(b * w for _, b, w, _ in config.mixed_shapes), device, world_size=shard.world_size)
                owners.append(mixed_scratch)
                mixed_owner = speculation if speculation is not None else SimpleNamespace(
                    weights=weights, pool=state, requests=requests, inputs=inputs, scratch=scratch,
                    workspaces=workspaces, rotary=rotary, cache=cache, table=table, session=session,
                    completion=completion, frame=frame, sequence=sequence, max_cache_pages=planned_pages,
                    collectives=collectives, sampling_workspace=sampling_workspace)
                for enabled, batch, width, prefill in config.mixed_shapes:
                    name = f"mixed{batch}x{width}p{prefill}" + ("" if enabled else "-no-mtp")
                    operation = PreparedMixedRound(mixed_owner, batch, width, prefill, name,
                        scratch=mixed_scratch, features=features, enabled=enabled)
                    fields = (*inputs.buffers(), "sampling_grammar_mask",
                              *(speculation.buffers() if enabled else ()))
                    add(name, operation, "mixed", batch, width, fields, enabled=enabled, prefill_capacity=prefill)
                    decoders.append(operation.decoder)
                    owners.append(operation)

            state_fields = tuple(f"state_{name}" for name in transfer.__dataclass_fields__ if name != "errors")
            admission_fields = tuple(admission.__dataclass_fields__)
            for enabled in serving_modes:
                suffix = "-no-mtp" if config.mtp and not enabled else ""
                forward_fields = (*inputs.buffers(), "sampling_grammar_mask",
                                  *(("draft_page_tables",) if enabled and has_mtp else ()))
                for capacity in reversed(config.batch_capacities):
                    name = f"resume{capacity}" + suffix
                    resume = PreparedResume(inputs, requests, state, scratch.sample_hidden[:capacity],
                        scratch.logits[:capacity], weights.tree.lm_head.weight, completion, frame, sequence,
                        session=session, name=name, vocabulary_runtime=vocabulary_collective,
                        local_logits=None if scratch.local_logits is None else scratch.local_logits[:capacity],
                        coordinator=coordinator, sampling_workspace=sampling_workspace)
                    forward = speculation.resume(resume) if enabled else resume
                    add(name, forward, "resume", capacity, 1, forward_fields, enabled=enabled)
                    owners.append(resume)
                    for mode in ("init", "restore", "snapshot"):
                        maintenance = PreparedMaintenance(mode, requests, state, transfer, completion, frame, sequence,
                            admission_buffers=admission, forward_inputs=inputs,
                            qsa_states=tuple(value for index, value in cache.items() if index < 48), capacity=capacity,
                            speculation=speculation if enabled else None)
                        maintenance.prepare()
                        fields = state_fields + (forward_fields if mode == "snapshot" else ("request_state_rows", *admission_fields))
                        if mode == "restore":
                            fields += ("page_tables", *(("draft_page_tables",) if enabled and has_mtp else ()))
                        add(f"{mode}{capacity}" + suffix, maintenance, mode, capacity, 1, fields, enabled=enabled)
                        owners.append(maintenance)
        session.register_scratch(workspaces)
        with phase(progress, "Prepare and tune kernels") as stage, transient:
            tuning = session.tune(show_progress=progress is not None, console=stage.console)
        report(f"selected prepared kernels: {tuning['selection_counts']}")
        stream = torch.cuda.Stream(device=device)
        graph_pool = torch.cuda.graph_pool_handle() if package_storage is None else package_storage.graph_pool.id
        compiled_functions = {}
        from ignition.components.operator_errors import guard_forwards

        functions = guard_forwards(functions, requests, completion, frame, sequence)
        def capture_catalog(*, compile_decoder, materialize=None, label="Prepare serving graphs"):
            captured, compiled = capture_forwards(functions, signatures, session=session, device=device,
                stream=stream, pool=graph_pool, compile_decoder=compile_decoder, materialize=materialize,
                progress=progress, label=label, cache_root=config.kernel_cache,
                collective=collective, coordinator=coordinator, retain_graphs=config.retain_graphs,
                warmup_storage=package_storage)
            graphs.update(captured)
            compiled_functions.clear()
            compiled_functions.update(compiled)

        page_capacity = config.preparation_pages

        def rebind_cache(pages):
            for decoder in decoders:
                for index in range(3, 48, 4):
                    decoder.layers[index][1].rebind_pages_for_preparation(pages[index])
                decoder.qsa_states = pages
            if speculation is not None:
                speculation.rebind_pages_for_preparation(pages)

        if compile_decoder:
            from b12x._lib.compile_plan import compile_only_launches

            mode = deferred_mode()
            with compile_only_launches():
                rebind_cache({index: value.resize_pages_for_preparation(planned_pages if package_storage is not None else page_capacity, metadata_mode=mode)
                              for index, value in cache.items()})

        final_cache = None
        def materialize_cache():
            nonlocal final_cache
            with phase(progress, "Allocate preparation KV cache"):
                free = minimum_across_ranks(device_memory_info(device)[0], group)
                if page_capacity * page_bytes + config.headroom_bytes > free:
                    raise MemoryError("insufficient memory for preparation KV pages and configured headroom")
                if package_storage is not None:
                    final_cache = {index: value.resize_pages_for_preparation(planned_pages, storage=package_storage,
                        prefix=f"kv-{index}-", backed_pages=page_capacity) for index, value in cache.items()}
                else:
                    final_cache = {index: value.resize_pages_for_preparation(page_capacity)
                                   for index, value in cache.items()}
                report(f"{page_capacity * page_bytes / (1 << 20):.1f} MiB per GPU · {page_capacity:,} pages")
                rebind_cache(final_cache)
                for owner in owners:
                    if isinstance(owner, PreparedMaintenance):
                        owner.rebind_pages_for_preparation(final_cache)
                state.memory_capacity[1] = page_capacity

        capture_catalog(compile_decoder=compile_decoder, materialize=materialize_cache)
        session.freeze()
        free, total = device_memory_info(device)
        memory = {"weights_bytes": weights.allocated_bytes, "ngram_request_bytes": table.mailbox.host_view.nbytes,
            "hybrid_state_bytes": state.backing.nbytes, "state_rows": state_rows,
            "preparation_state_bytes": state.backed_slots * layout.row_bytes,
            "preparation_state_rows": state.backed_slots,
            "pages": page_capacity, "page_bytes": page_bytes, "kv_bytes": page_capacity * page_bytes,
            "cached_token_capacity": page_capacity * 64, "headroom_bytes": config.headroom_bytes, "page_tokens": 64,
            "free_device_bytes": free, "total_device_bytes": total,
            "allocated_bytes": torch.cuda.memory_allocated(device), "reserved_bytes": torch.cuda.memory_reserved(device),
            "completion_output_bytes": completion.nbytes, "operator_scratch_bytes": workspaces.nbytes, "graphs": len(graphs), "tuning": tuning,
            "prefill_chunk_size": config.prefill_chunk_size, "prefill_capacities": config.prefill_capacities,
            "mixed_shapes": config.mixed_shapes,
            "mixed_prefill": config.mixed_prefill,
            "compiled_decoder": compile_decoder, "rank": rank, "world_size": shard.world_size,
            "shard_hyperconnections": config.shard_hyperconnections}
        if operators is not None:
            memory['operators'] = operators
        if session.dense_blas is not None:
            memory['dense_blas_workspace_bytes'] = session.dense_blas.workspace.nbytes
            memory['operator_scratch_bytes'] += session.dense_blas.workspace.nbytes
        if scoring is not None:
            memory.update(scoring.memory_report())
        if speculation is not None:
            memory.update(mtp_weights_bytes=0 if mtp_weights is None else mtp_weights.allocated_bytes,
                speculative_widths=config.speculative_widths,
            speculative_token_capacity=speculative_token_capacity(config.max_active, max(config.speculative_widths), config.speculative_token_capacity),
                speculative_shapes=config.speculative_shapes,
                speculative_state_columns=8, draft_raw_ring_capacity=layout.draft_ring_capacity if has_mtp else 0,
                speculative_fixed_width=config.speculative_fixed_width)
        hashes = {"tokenizer": ("tokenizer.json", FRONTEND_SHA256["tokenizer.json"])}
        cache_regions = [history_region(buffers, f"cache.{layer}.{name}", getattr(value, name), contract_proto.PAGED_KEY_VALUE,
            tokens_per_record=4 if name == "compressed_k_cache" else 1)
            for layer, value in (final_cache or cache).items()
            for name in ("main_k_cache", "main_v_cache", "compressed_k_cache")]
        buffers["request_pending_token"] = requests.pending_token
        server_config = {"model_id": MODEL_ID if config.gguf_inputs is None else str(snapshot),
            "identity": _identity(manifest, shard.world_size, operators=operators),
            "max_sequence_tokens": config.context, "prefill_chunk_size": config.prefill_chunk_size,
            "pages": page_capacity, "state_rows": state_rows, "snapshots": snapshots, "page_tokens": 64,
            "frontend": prepare_frontend(manifest.metadata_path, eos_token_ids=[248046, 248044],
                                         image_profile="qwen" if features is not None else None),
            "sampling_defaults": RECIPE.sampling_defaults.as_dict(),
            "model_contract": prepare_contract(signatures, buffers, state, image_slots=config.image_concurrency, page_tokens=64, draft_lag=1 if has_mtp else None, embedding_tables=(0,), regions=cache_regions),
            "assets": {name: {"path": str((manifest.metadata_path / filename).resolve()), "sha256": digest}
                       for name, (filename, digest) in hashes.items()}}
        server_config["assets"]["chat_template"] = annotated_template_asset(
            (manifest.metadata_path / "chat_template.jinja").read_bytes(),
            Path(__file__).with_name("assets") / "chat_template.patch.json")
        if speculation is not None:
            server_config.update(speculative_widths=config.speculative_widths,
            speculative_token_capacity=speculative_token_capacity(config.max_active, max(config.speculative_widths), config.speculative_token_capacity),
                speculative_fixed_width=config.speculative_fixed_width)
        resources = (scoring, weights, table, session, state, requests, inputs, scratch, workspaces, rotary,
            cache, final_cache, admission, transfer, features, vision_inputs, vision_scratch, graph_buffers,
            functions, compiled_functions, decoders, owners, stream, graph_pool, speculation, mtp_weights, collective, vocabulary_collective, coordinator)
        report(f"prepared {len(graphs)} graphs with {page_capacity} preparation KV pages")
        from .export import disposable_ranges

        export_zero_ranges = disposable_ranges(cache=final_cache, scratch=scratch, workspaces=workspaces,
            vision_scratch=vision_scratch, decoders=decoders, owners=owners, speculation=speculation)
        from ignition.preparation.disposable import storage_ranges

        export_zero_ranges = (*export_zero_ranges, *storage_ranges(graph_buffers.chunks))
        if session.dense_blas is not None:
            export_zero_ranges = (*export_zero_ranges, *storage_ranges((session.dense_blas.workspace,)))
        return PreparedModel(rebuild_weights=declare_weights({'weights': weights, 'mtp': mtp_weights}, components={
        'tree.model.language_model.embed_tokens': 'embedding',
        'embedding': 'embedding',
        'tree.model.language_model.hyper_connection_mixer': 'hyper-connection',
        'tree.lm_head': 'head',
        'vision': 'vision',
        'mtp': 'mtp',
    }, layers=('layers', 'tree.model.language_model.layers')), device=device.index, max_active=config.max_active, graphs=graphs, buffers=buffers, signatures=signatures,
            export_zero_ranges=(*export_zero_ranges, *(() if scoring is None else scoring.zero_ranges()), *completion.zero_ranges(),
                                *((t.data_ptr(), t.nbytes) for t in table.buffers().values())),
            export_events=table.events(), package_data=table if rank == 0 else None,
            resources=resources, server_config=server_config, memory_report=memory, session=session,
            close_resources=(table, *((quactlize_helpers,) if quactlize_helpers is not None else ()),
                             *((scoring,) if scoring is not None else ()),
                             *((vocabulary_collective, collective) if collective is not None else ())))
    except BaseException:
        torch.cuda.synchronize(device)
        for graph in graphs.values():
            graph.reset()
        if quactlize_helpers is not None:
            quactlize_helpers.close()
        session.close()
        table.close()
        raise
