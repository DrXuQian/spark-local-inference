# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixed-address metadata and preparation for recurrent sequence kernels."""

from __future__ import annotations

from .recurrent_state import DeltaMetadata, DeltaState

import torch
import triton
import triton.language as tl
from b12x.preparation import PreparedCall

from ignition.preparation.fixtures import recurrent as recurrent_fixtures
from ignition.preparation.scratch import planned_scratch, prime_owners
from ignition.preparation.tuning import declare, register_finalizer


@triton.jit
def _commit_convolution(Backing, Offsets, Rows, Counts, STRIDE: tl.constexpr,
                        CHANNELS: tl.constexpr, COLUMNS: tl.constexpr, BLOCK: tl.constexpr):
    layer, row, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    count = tl.load(Counts + row)
    state = tl.load(Rows + row).to(tl.int64)
    channel = tile * BLOCK + tl.arange(0, BLOCK)
    live = (count > 1) & (state > 0) & (channel < CHANNELS)
    base = Backing + state * STRIDE + tl.load(Offsets + layer) + channel * COLUMNS
    first = tl.load(base + count - 1, live, other=0)
    second = tl.load(base + count, live, other=0)
    third = tl.load(base + count + 1, live, other=0)
    tl.store(base, first, live)
    tl.store(base + 1, second, live)
    tl.store(base + 2, third, live)






def prepare_sequence_binding(
    plan, args, *, session, name, public_bind, mode, set_binding,
    bind_plan=False, workspace=None, run_options=None, bind_method="bind",
):
    run_options = {} if run_options is None else dict(run_options)

    def selected_scratch(specs):
        return planned_scratch(specs, workspace)

    def candidate_specs(prepared):
        return plan.scratch_specs() if bind_plan else prepared.layout.scratch_specs()

    def prepare_call(prepared):
        scratch = selected_scratch(candidate_specs(prepared))
        binding = getattr(prepared, bind_method)(scratch=scratch, **args, **({"plan": plan} if bind_plan else {}))
        return PreparedCall(run=lambda: prepared.run(binding, **run_options), owners=prime_owners(workspace, scratch, binding))

    def benchmark_call(prepared):
        return recurrent_fixtures.benchmark_call(
            prepared, arguments=args, session=session, name=name, mode=mode,
            scratch_specs=candidate_specs, workspace=workspace,
            run_options=run_options, binding_plan=plan if bind_plan else None, bind_method=bind_method,
        )

    declare(session, (plan.request(name=name, prepare_call=prepare_call,
        benchmark_call=benchmark_call),))
    selected = None

    def finalize():
        nonlocal selected
        scratch = selected_scratch(plan.scratch_specs())
        selected = public_bind(plan, scratch=scratch, **args)
        set_binding(selected)

    register_finalizer(session, finalize, workspace=workspace)
    return selected


