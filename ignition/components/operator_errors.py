# SPDX-License-Identifier: Apache-2.0

"""Consume external operator error flags before publishing a graph completion."""

import torch

from .sampling import ERROR, RUNNING

OPERATOR_FAILURE = 7


def device_errors(*operators):
    """Prepared owners expose explicit error storage, never host callbacks."""
    flags = {}
    for operator in operators:
        for flag in getattr(operator, 'device_errors', ()):
            flags[id(flag)] = flag
    return tuple(flags.values())


def accumulate_errors(flags, output):
    for flag in flags:
        output.logical_or_(flag.ne(0).any())


class ErrorAccumulator:
    """Retain repeated-call errors and expose every flag for round reset."""

    def __init__(self, *operators):
        self.inputs = device_errors(*operators)
        # An ABI may retain its bits until the caller clears them. Exposing
        # only our accumulator would leave those source flags stale next round.
        self.device_errors = ((torch.zeros(1, dtype=torch.int32, device=self.inputs[0].device),
                               *self.inputs) if self.inputs else ())

    def __call__(self):
        if self.inputs:
            accumulate_errors(self.inputs, self.device_errors[0])


class GuardedForward:
    """A batch-global error fails every live completion produced by this frame.

    These ABIs do not identify the failing request. Check after all operations,
    including post-sampling MTP absorption. Request and completion identities
    must match before mutation; another frame or generation stays untouched.
    """

    def __init__(self, forward, requests, completion, frame, sequence):
        self.forward, self.requests, self.completion = forward, requests, completion
        self.frame, self.sequence = frame, sequence
        self.flags = device_errors(forward)
        if not self.flags:
            raise ValueError('operator guard requires declared device error flags')
        for flag in self.flags:
            if (flag.dtype != torch.int32 or flag.device != requests.status.device
                    or not flag.is_contiguous() or not flag.numel()):
                raise ValueError('operator errors require nonempty contiguous device int32 storage')
        self.fields = torch.arange(completion.records.shape[2], device=requests.status.device)
        self.consumed_before = torch.empty(requests.capacity, dtype=torch.int64, device=requests.status.device)
        self.generated_before = torch.empty_like(self.consumed_before)
        self.status_before = torch.empty_like(requests.status)

    def __call__(self):
        for flag in self.flags:
            flag.zero_()
        self.consumed_before.copy_(self.requests.consumed_tokens)
        self.generated_before.copy_(self.requests.generated_tokens)
        self.status_before.copy_(self.requests.status)
        self.forward()
        failed = self.flags[0].ne(0).any()
        for flag in self.flags[1:]:
            failed = failed | flag.ne(0).any()
        index = self.frame.long()
        records = self.completion.records.index_select(0, index)
        active = (records[..., 1] == self.sequence[0]) & (records[..., 0] == self.requests.generation)
        active = active & (self.status_before == RUNNING) & records[..., 5].ne(0) & failed
        errors = torch.where(records[..., 5] == ERROR, records[..., 6], OPERATOR_FAILURE)
        self.requests.status.copy_(torch.where(active[0], ERROR, self.requests.status))
        self.requests.error_code.copy_(torch.where(active[0], errors[0], self.requests.error_code))
        # No token from a failed round reaches the client. Keep its logical
        # progress unchanged so the existing native completion validator and
        # any already-queued round observe the same committed endpoint.
        self.requests.consumed_tokens.copy_(torch.where(active[0], self.consumed_before, self.requests.consumed_tokens))
        self.requests.generated_tokens.copy_(torch.where(active[0], self.generated_before, self.requests.generated_tokens))
        fields = self.fields
        invalid = (fields == 4) | (fields == 12) | (fields == 13) | (fields >= 16)
        zero = ((fields >= 8) & (fields <= 11)) | (fields == 14)
        replacement = torch.where(invalid, -1, torch.where(zero, 0, records))
        replacement = torch.where(fields == 5, ERROR, replacement)
        replacement = torch.where(fields == 6, errors[..., None], replacement)
        replacement = torch.where(fields == 2, self.consumed_before[None, :, None], replacement)
        replacement = torch.where(fields == 3, self.generated_before[None, :, None], replacement)
        self.completion.records.index_copy_(0, index, torch.where(active[..., None], replacement, records))
        if self.completion.speculative_survival is not None:
            survival = self.completion.speculative_survival.index_select(0, index)
            self.completion.speculative_survival.index_copy_(0, index,
                torch.where(active[..., None], -1., survival))


def guard_forwards(forwards, requests, completion, frame, sequence):
    return {name: GuardedForward(forward, requests, completion, frame, sequence)
            if device_errors(forward) else forward for name, forward in forwards.items()}
