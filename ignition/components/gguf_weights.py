# SPDX-License-Identifier: Apache-2.0

"""Unchanged GGUF blocks loaded through the shared checkpoint transport."""

from dataclasses import dataclass
from math import prod

from ignition.preparation.gguf import GgufCheckpoint
from ignition.preparation.rebuild_manifest import declare_container_fields
from .weights import DirectWeightLoader, _aligned_range


@dataclass(frozen=True)
class GgufWeight:
    raw: object
    encoding: str
    qtype: int
    shape: tuple[int, ...]

    @property
    def in_features(self):
        return self.shape[-1]

    @property
    def out_features(self):
        return self.shape[-2]

    @property
    def experts(self):
        return self.shape[0] if len(self.shape) == 3 else 1


declare_container_fields('ignition.components.gguf_weights.GgufWeight', ('raw',))


class GgufWeightLoader(DirectWeightLoader):
    """Read selected projections/experts without loading the host PLE table.

    Names are GGUF tensor names. Selection and fusion operate on complete
    quantization blocks; no dequantization, padding or requantization occurs.
    Model recipes own the mapping from these names to their modules.
    """

    def __init__(self, manifest, *args, **kwargs):
        if not isinstance(manifest, GgufCheckpoint):
            raise TypeError("GGUF loading requires an inspected GgufCheckpoint")
        super().__init__(manifest, *args, **kwargs)

    def _selected_names(self):
        return {name for name in super()._selected_names()
                if self.manifest.tensors[name].component != 'ple_tables'}

    def _view(self, name, rows, columns, experts):
        import gguf

        self._require_open()
        metadata = self.manifest.tensors[name]
        if len(metadata.shape) not in (2, 3):
            raise ValueError(f"{name}: projection weights require [N,K] or [E,N,K]")
        qtype = gguf.GGMLQuantizationType[metadata.encoding]
        block, width = gguf.GGML_QUANT_SIZES[qtype]
        if block == 1:
            raise ValueError(f"{name}: load plain weights with load_tensor")
        n, k = metadata.shape[-2:]
        r0, r1 = _aligned_range(rows, n, 1)
        c0, c1 = _aligned_range(columns, k, block)
        if r0 == r1 or c0 == c1:
            raise ValueError(f"{name}: empty projection slice")
        logical = [(r0, r1), (c0, c1)]
        selection = (slice(r0, r1), slice(c0 // block, c1 // block), slice(None))
        shape = (r1 - r0, c1 - c0)
        if len(metadata.shape) == 3:
            e0, e1 = _aligned_range(experts, metadata.shape[0], 1)
            if e0 == e1:
                raise ValueError(f"{name}: empty expert slice")
            selection = (slice(e0, e1), *selection)
            logical.insert(0, (e0, e1))
            shape = (e1 - e0, *shape)
        elif experts is not None:
            raise ValueError(f"{name}: dense weights have no expert axis")
        source = self._sources[name][selection]
        if source.numel() != prod(shape) // block * width:
            raise ValueError(f"{name}: selected storage differs from logical GGUF geometry")
        return source, shape, tuple(logical), qtype

    def load_quantized(self, name, *, rows=None, columns=None, experts=None):
        return self.load_quantized_group((name,), row_ranges=(rows,), columns=columns, experts=experts)

    def load_dequantized(self, name, *, rows=None, columns=None, dtype=None):
        """Expand one dense projection during preparation, never an expert/table set.

        Recipes choose which small matrices to retain in 16-bit storage. The
        external decoder performs one final FP16/BF16 rounding of GGUF values;
        packed staging is released after the asynchronous conversion completes.
        """
        import torch
        from quactlize.cuda_compat import gather_rows
        from ignition.preparation import provenance

        dtype = torch.bfloat16 if dtype is None else dtype
        metadata = self.manifest.tensors[name]
        if (len(metadata.shape) != 2 or metadata.component == 'ple_tables'
                or dtype not in (torch.float16, torch.bfloat16)):
            raise ValueError('dense GGUF expansion requires a projection and FP16/BF16 output')
        weight = self.load_quantized(name, rows=rows, columns=columns)
        sources = provenance.lookup(weight.raw)['sources']
        output = torch.empty(weight.shape, device=self.device, dtype=dtype)
        ids = torch.arange(weight.shape[0], device=self.device, dtype=torch.int64)
        error = torch.empty((), device=self.device, dtype=torch.int32)
        gather_rows(weight.raw, ids, output, None, error, weight.qtype, weight.shape[0])
        if error.item():
            raise RuntimeError(f'{name}: external GGUF decoder rejected dense row geometry')
        provenance.record_sources(output, sources, packing='gguf_dequantized',
            packing_parameters=dict(encoding=weight.encoding, dtype=str(dtype).removeprefix('torch.')))
        return output

    def load_quantized_group(self, names, *, row_ranges=None, columns=None, experts=None):
        """Fuse output rows, separately within each expert when E is present."""
        import torch
        from ignition.preparation import provenance

        names = tuple(names)
        if not names:
            raise ValueError("projection group must be nonempty")
        ranges = (None,) * len(names) if row_ranges is None else tuple(row_ranges)
        parts = [self._view(name, rows, columns, experts)
                 for name, rows in zip(names, ranges, strict=True)]
        source, shape, _, qtype = parts[0]
        if any(kind != qtype or logical[:-2] != shape[:-2] or logical[-1] != shape[-1]
               for _, logical, _, kind in parts):
            raise ValueError("GGUF projection fusion requires equal encoding, K and expert axes")
        shape = (*shape[:-2], sum(logical[-2] for _, logical, _, _ in parts), shape[-1])
        storage = (*source.shape[:-3], shape[-2], *source.shape[-2:])
        raw = torch.empty(storage, dtype=torch.uint8, device=self.device)
        axis = len(shape) - 2
        offset = 0
        for part, logical, _, _ in parts:
            self._copy(raw.narrow(axis, offset, logical[-2]), part)
            offset += logical[-2]
        self._session.flush()
        provenance.record_sources(raw,
            [dict(tensor=name, ranges=logical) for name, (_, _, logical, _) in zip(names, parts, strict=True)],
            packing='gguf_raw', concatenation_axis=axis if len(names) > 1 else None,
            packing_parameters=dict(encoding=qtype.name))
        return GgufWeight(raw, qtype.name, int(qtype), shape)

    def load_linear_group(self, prefixes, *, row_ranges=None, column_range=None, **kwargs):
        if kwargs:
            raise ValueError("GGUF blocks carry their own scales")
        return self.load_quantized_group(tuple(f'{prefix}.weight' for prefix in prefixes),
                                          row_ranges=row_ranges, columns=column_range)
