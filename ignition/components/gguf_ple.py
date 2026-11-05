# SPDX-License-Identifier: Apache-2.0

"""Reuse b12x PLE hashes/readiness with external IQ4_NL row decoding."""

import torch
from b12x.preparation import PreparedCall
from b12x.sequence import ple_hash

from ignition.preparation.tuning import declare
from .ngram_delivery import _requests
from .readiness import PreparedReadinessWait, wait_for_readiness


@torch.library.custom_op('flare::staged_gguf_ple', mutates_args=('out', 'error', 'timings', 'barrier_output'))
def _lookup(source: torch.Tensor, weight: torch.Tensor, ids: torch.Tensor,
            count: torch.Tensor, out: torch.Tensor, error: torch.Tensor,
            timings: torch.Tensor, frame: torch.Tensor, barrier_input: torch.Tensor,
            barrier_output: torch.Tensor, barrier: int, event: int, helpers: int = 0) -> None:
    wait_for_readiness(source, event, timings, frame, 0, barrier_input, barrier_output, barrier)
    if helpers:
        from quactlize.serving import gather_rows

        gather_rows(weight, ids, out.view(-1, 160), count, error, 20, weight.shape[0], helpers)
    else:
        from quactlize.cuda_compat import gather_rows

        gather_rows(weight, ids, out.view(-1, 160), count, error, 20, weight.shape[0])


@_lookup.register_fake
def _lookup_fake(source, weight, ids, count, out, error, timings, frame, barrier_input,
                 barrier_output, barrier, event, helpers=0):
    pass


class PreparedGgufPLELookup:
    def __init__(self, table, metadata, output, caps, *, session, name, runtime=None, coordinator=None):
        self.metadata, self.output, self.geometry = metadata, output, table.geometry
        self.helpers = getattr(session, 'quactlize_helpers', None)
        self.mailbox, self.sequence = table.mailbox.device_view, table.sequence
        tokens, self.heads = output.shape[0], table.layout.caps.head_count
        if output.shape != (tokens, self.heads * 160):
            raise ValueError('GGUF PLE output must contain every 160-wide hash head')
        self.ids = table.hashes[:tokens]
        self.requests = torch.empty(tokens, dtype=torch.int32, device=output.device)
        self.rows = torch.arange(tokens * self.heads, dtype=torch.int64, device=output.device)
        self.count = torch.zeros(1, dtype=torch.int32, device=output.device)
        self.error = torch.zeros((), dtype=torch.int32, device=output.device)
        self.device_errors = (self.error,)
        self.weights, self.event = table.weights[:tokens * self.heads], table.ready.cuda_event
        self.wait = PreparedReadinessWait(table.timings, 0, runtime=runtime, session=session,
                                         coordinator=coordinator, name=name + '.readiness')
        hash_caps = ple_hash.Caps(**{field: getattr(caps, field) for field in ple_hash.Caps.__dataclass_fields__})
        g, m = table.geometry, metadata
        kwargs = dict(token_ids=m.token_ids, query_start_loc=m.query_start_loc, committed_history=m.committed_history,
            num_seqs=m.num_seqs, num_tokens=m.num_tokens, prime_sizes=g.prime_sizes,
            table_offsets=g.table_offsets, multipliers=g.multipliers, out=self.ids)
        self.plan = ple_hash.plan(hash_caps, geometry=g.geometry,
            prime_sizes=g.prime_sizes, table_offsets=g.table_offsets, multipliers=g.multipliers,
            invocation=ple_hash.invocation_from_tensors(**kwargs))
        declare(session, (self.plan.request(name=name + '.hash', prepare_call=lambda state: PreparedCall(
            run=lambda: state.run_tensors(m.token_ids, m.query_start_loc, m.committed_history, m.num_seqs,
                m.num_tokens, g.multipliers, g.prime_sizes, g.table_offsets, self.ids, self.requests,
                token_count=tokens), owners=(table, self.requests))),))

    def publish(self):
        m, g = self.metadata, self.geometry
        _requests(m.token_ids, m.query_start_loc, m.committed_history, m.num_seqs, m.num_tokens,
            g.multipliers, g.prime_sizes, g.table_offsets, self.ids, self.requests,
            self.mailbox, self.sequence, self.plan.handle)

    def __call__(self, source):
        self.count.copy_(self.metadata.num_tokens * self.heads)
        _lookup(source, self.weights, self.rows, self.count, self.output, self.error,
            self.wait.timings.timestamps, self.wait.timings.frame, self.wait.input, self.wait.output,
            self.wait.handle, self.event, 0 if self.helpers is None else self.helpers.handle)
