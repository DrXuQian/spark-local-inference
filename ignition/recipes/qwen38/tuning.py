# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Preparation policy for the Qwen 3.8 operator catalog."""

from functools import wraps

from ignition.preparation.qwen38_gguf import operator_target


def external_operators(inputs):
    if inputs is None:
        return False
    operator_target(inputs)
    required = ('actlize', 'quactlize', 'dense_blas', 'quactlize_helpers', 'flash_attention3')
    missing = [name for name in required if getattr(inputs, name) is None]
    return not missing


def with_autotune_effort(prepare):
    @wraps(prepare)
    def scoped(snapshot, config, *args, **kwargs):
        if external_operators(getattr(config, 'gguf_inputs', None)):
            return prepare(snapshot, config, *args, **kwargs)
        from ignition.preparation.tuning import with_autotune_effort as legacy_scope

        return legacy_scope(prepare)(snapshot, config, *args, **kwargs)
    return scoped


def preparation_session(*, device, checkpoint, namespace=None, coordinator=None, headroom_bytes=0,
                        benchmark_storage=None, compile_workers=None, tuning_replica=None, autotune_effort="default", collective_tuning="static", collective_ladder=None,
                        native_dense_activations=False, lock_gpu_clocks=False, dense_blas_api=None, gguf_inputs=None):
    if external_operators(gguf_inputs):
        from ignition.preparation.external_session import ExternalPreparationSession

        if coordinator is not None or tuning_replica is not None or collective_tuning != 'static' or collective_ladder is not None:
            raise ValueError('external GGUF operators currently require one device without collective tuning')
        if native_dense_activations:
            raise ValueError('external GGUF dense projections require BF16 activations')
        identity = dict(namespace or {}, workload_schema='qwen38-external-operators-v1',
                        numerical_policy='gguf-bf16-fp32-accum')
        return ExternalPreparationSession(device=device, checkpoint=checkpoint, namespace=identity,
            compile_workers=compile_workers, headroom_bytes=headroom_bytes,
            autotune_effort=autotune_effort, lock_gpu_clocks=lock_gpu_clocks)
    from ignition.preparation.tuning import RecordedPreparationSession

    return RecordedPreparationSession(
        device=device, checkpoint=checkpoint, compile_workers=compile_workers, tuning_replica=tuning_replica,
        collective_tuning=collective_tuning, collective_ladder=collective_ladder, autotune_effort=autotune_effort, lock_gpu_clocks=lock_gpu_clocks, native_dense_activations=native_dense_activations,
        namespace=namespace, coordinator=coordinator,
        headroom_bytes=headroom_bytes, benchmark_storage=benchmark_storage,
        workload_schema="qwen38-active-operators-v1",
        numerical_policy="qsa16-gdn-sequential-k1-dense-mxfp8-auto-vision-nvfp4-a16-moe-a4-route-fp32",
        search_limits={
            "bf16_projection": ("all eligible b12x backends and rows_per_tile configurations"
                if dense_blas_api is None else f"{dense_blas_api} GemmEx, FP32 accumulation, caller-owned scratch"),
            "mxfp8_projection": "BF16 or quantized activations with eligible b12x projection configurations",
            "qsa": "fixed attention_splits=16; eligible direct-KV warp configurations",
            "gdn_prefill": "sequential k_split=1; v_split, stages, and window_tiles remain tunable",
            "operator_scratch": "one arena sized to the complete candidate envelope",
        },
    )
