# SPDX-License-Identifier: Apache-2.0

"""Bounded GGUF experts composed from external projections and b12x SwiGLU."""

from dataclasses import dataclass

import torch
from b12x.norm import hyperconnection
from b12x.preparation import PreparedCall

from ignition.preparation.rebuild_manifest import declare_container_fields
from ignition.preparation.tuning import declare
from .gguf_projection import PreparedGgufProjection, projection_workspace_bytes
from .gguf_weights import GgufWeight


@dataclass(frozen=True)
class GgufExperts:
    gate_up: GgufWeight
    down: GgufWeight

    def __post_init__(self):
        if (len(self.gate_up.shape) != 3 or len(self.down.shape) != 3
                or self.gate_up.experts != self.down.experts
                or self.gate_up.in_features != self.down.out_features
                or self.gate_up.out_features != 2 * self.down.in_features
                or self.gate_up.raw.device != self.down.raw.device):
            raise ValueError('GGUF experts require gate/up [E,2I,H] and down [E,H,I] on one device')

    @property
    def num_experts(self):
        return self.gate_up.experts

    @property
    def hidden_size(self):
        return self.gate_up.in_features

    @property
    def intermediate_size(self):
        return self.down.in_features


declare_container_fields('ignition.components.gguf_moe.GgufExperts', ('gate_up', 'down'))


@dataclass(frozen=True)
class GgufMoeWorkspace:
    """One bounded chunk, shared by serial layers and graph capacities.

    The caller selects chunk_rows from its model memory budget. All experts stay
    packed; only this many tokens' selected slots expand into matmul storage.
    Larger prefill forwards record a fixed sequence of chunk calls in one graph.
    """

    matmul: torch.Tensor
    gate_up: torch.Tensor
    intermediate: torch.Tensor
    projected: torch.Tensor

    @classmethod
    def allocate(cls, experts: GgufExperts, *, chunk_rows: int, top_k: int, buffers=None):
        if (type(chunk_rows) is not int or chunk_rows < 1 or type(top_k) is not int
                or not 0 < top_k <= experts.num_experts):
            raise ValueError('GGUF expert workspace requires positive rows and valid top-k')
        device, hidden, width = experts.gate_up.raw.device, experts.hidden_size, experts.intermediate_size
        def empty(shape, dtype):
            return torch.empty(shape, dtype=dtype, device=device) if buffers is None else buffers.zeros(shape, dtype)
        gate_up = empty((chunk_rows, top_k, 2 * width), torch.bfloat16)
        intermediate = empty((chunk_rows, top_k, width), torch.bfloat16)
        projected = empty((chunk_rows, top_k, hidden), torch.bfloat16)
        # Query uses geometry only, including these small temporary token/ID views.
        source = torch.empty((chunk_rows, hidden), dtype=torch.bfloat16, device=device)
        ids = torch.empty((chunk_rows, top_k), dtype=torch.int32, device=device)
        required = max(projection_workspace_bytes(source, experts.gate_up, ids=ids),
                       projection_workspace_bytes(intermediate, experts.down, ids=ids))
        return cls(empty((required,), torch.uint8), gate_up, intermediate, projected)

    @property
    def chunk_rows(self):
        return self.gate_up.shape[0]


class PreparedGgufRoutedExperts:
    """Execute dynamic routes with fixed storage and no host routing decisions."""

    def __init__(self, source, experts: GgufExperts, ids, weights, output, *, session, name,
                 workspace: GgufMoeWorkspace, router_plan=None,
                 a4_prefill=False, routing=None, router=None, route_output=None):
        if a4_prefill or routing is not None or router is not None or route_output is not None:
            raise ValueError('GGUF experts require external routing and 16-bit activations')
        if not isinstance(workspace, GgufMoeWorkspace):
            raise TypeError('GGUF experts require a bounded GgufMoeWorkspace')
        if source.ndim != 2 or source.shape[1] != experts.hidden_size or source.shape[0] < 1:
            raise ValueError('GGUF expert source requires [tokens, hidden]')
        tokens, hidden = source.shape
        if ids.ndim != 2 or ids.shape[0] != tokens or not 0 < ids.shape[1] <= experts.num_experts:
            raise ValueError('GGUF expert IDs require [tokens, topk]')
        topk, width, capacity = ids.shape[1], experts.intermediate_size, workspace.chunk_rows
        if weights.shape != ids.shape or output.shape != source.shape:
            raise ValueError('GGUF expert route weights and output must match their inputs')
        for tensor, dtype in ((source, torch.bfloat16), (ids, torch.int32),
                              (weights, torch.float32), (output, torch.bfloat16)):
            if tensor.dtype != dtype or tensor.device != source.device or not tensor.is_contiguous():
                raise ValueError('GGUF expert buffers require contiguous native dtypes on one device')
        for tensor, shape in ((workspace.gate_up, (capacity, topk, 2 * width)),
                              (workspace.intermediate, (capacity, topk, width)),
                              (workspace.projected, (capacity, topk, hidden))):
            if (tensor.shape != shape or tensor.dtype != torch.bfloat16
                    or tensor.device != source.device or not tensor.is_contiguous()):
                raise ValueError('GGUF expert chunk storage differs from the prepared geometry')
        self.source, self.ids, self.weights, self.output = source, ids, weights, output
        self.workspace, self.num_experts = workspace, experts.num_experts
        self.errors = torch.zeros(((tokens + capacity - 1) // capacity, 2), dtype=torch.int32, device=source.device)
        self.device_errors = (self.errors,)
        self.calls, prepared = [], {}
        for start in range(0, tokens, capacity):
            end = min(start + capacity, tokens)
            count = end - start
            if count not in prepared:
                up = PreparedGgufProjection(source[start:end], experts.gate_up, workspace.gate_up[:count],
                    workspace=workspace.matmul, ids=ids[start:end])
                down = PreparedGgufProjection(workspace.intermediate[:count], experts.down,
                    workspace.projected[:count], workspace=workspace.matmul, ids=ids[start:end])
                prepared[count] = up, down
            self.calls.append((start, end, *prepared[count]))
        self.activation_plan = hyperconnection.plan(hyperconnection.Caps(device=source.device,
            max_tokens=capacity * topk, hidden_size=width),
            invocation={'operation': 'swiglu', 'limit': float('inf'), 'round_silu': True})
        def prepare_activation(state):
            private = torch.zeros_like(workspace.gate_up).flatten(0, 1)
            result = torch.empty_like(workspace.intermediate).flatten(0, 1)
            return PreparedCall(run=lambda: state.launch(private, private, result), owners=(private, result))
        declare(session, (self.activation_plan.request(name=name + '.swiglu', prepare_call=prepare_activation),))

    def __call__(self):
        for index, (start, end, up, down) in enumerate(self.calls):
            count = end - start
            ids = self.ids[start:end]
            up(self.source[start:end], ids=ids, output=self.workspace.gate_up[:count],
               device_error=self.errors[index, 0])
            hyperconnection.run_swiglu(self.workspace.gate_up[:count].flatten(0, 1),
                limit=float('inf'), round_silu=True, out=self.workspace.intermediate[:count].flatten(0, 1),
                plan=self.activation_plan)
            down(self.workspace.intermediate[:count], ids=ids, output=self.workspace.projected[:count],
                 device_error=self.errors[index, 1])
            valid = (ids >= 0) & (ids < self.num_experts)
            weighted = self.workspace.projected[:count].float() * self.weights[start:end, :, None]
            self.output[start:end].copy_(torch.where(valid[..., None], weighted, 0.).sum(dim=1))
        return self.output
