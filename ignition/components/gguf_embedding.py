# SPDX-License-Identifier: Apache-2.0

"""Direct lookup of packed GGUF rows using the external QUactlize operator."""

import torch

from ignition.preparation.tensor_metadata import record_numeric_tensor
from .gguf_weights import GgufWeight


class PreparedGgufEmbedding:
    """Decode only requested token rows into fixed caller-owned storage."""

    def __init__(self, weight: GgufWeight, ids, output, *, session, name, num_rows=None):
        if (len(weight.shape) != 2 or ids.ndim != 1 or ids.dtype != torch.int64
                or output.shape != (ids.numel(), weight.in_features)
                or output.dtype not in (torch.bfloat16, torch.float16)
                or output.device.type != 'cuda' or weight.raw.dtype != torch.uint8):
            raise ValueError('GGUF embedding requires a packed table, int64 IDs and 16-bit output [tokens,hidden]')
        if num_rows is not None and (num_rows.numel() != 1 or num_rows.dtype != torch.int32):
            raise ValueError('GGUF embedding live count requires one int32 element')
        for tensor in (weight.raw, ids, output, num_rows):
            if tensor is not None and (tensor.device != output.device or not tensor.is_contiguous()):
                raise ValueError('GGUF embedding operands must be contiguous on the output device')
        import gguf

        block, size = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType(weight.qtype)]
        if weight.in_features % block or weight.raw.nbytes != weight.shape[0] * weight.in_features // block * size:
            raise ValueError('GGUF embedding raw byte size differs from its logical geometry')
        self.raw, self.ids, self.output, self.num_rows = weight.raw, ids, output, num_rows
        self.qtype, self.table_rows = weight.qtype, weight.shape[0]
        self.device_error = torch.zeros((), dtype=torch.int32, device=output.device)
        self.device_errors = (self.device_error,)
        self.helpers = getattr(session, 'quactlize_helpers', None)
        if self.helpers is not None and self.qtype not in (8, 20):
            raise ValueError('QUactlize serving row lookup requires Q8_0 or IQ4_NL')
        record_numeric_tensor(self.raw)

    def __call__(self):
        if self.helpers is None:
            from quactlize.cuda_compat import gather_rows
        else:
            gather_rows = self.helpers.gather_rows

        gather_rows(self.raw, self.ids, self.output, self.num_rows, self.device_error,
                    self.qtype, self.table_rows)
        return self.output
