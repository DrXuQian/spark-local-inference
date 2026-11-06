# SPDX-License-Identifier: Apache-2.0

"""Load the HF-named qwen4exp-mtp GGUF without converting its stored weights."""

import torch

from ignition.components.gguf_moe import GgufExperts
from ignition.components.gguf_weights import GgufWeightLoader
from ignition.components.parallel import TensorParallelShard
from ignition.preparation.qwen38_mtp_gguf import HF_EXPERTS, HF_MTP_SHAPES, validate_mtp_gguf
from .gguf_weights import _parameter
from .hyperconnection import packed_down_projection
from .moe import QwenMoeWeights, SharedExpertWeights


class HFMtpGgufWeightLoader(GgufWeightLoader):
    def __init__(self, manifest, *args, shard=TensorParallelShard(0, 1), **kwargs):
        if validate_mtp_gguf(manifest) != 'qwen4exp-mtp':
            raise ValueError('HF-named MTP loader requires qwen4exp-mtp')
        super().__init__(manifest, *args, **kwargs)
        self.shard = shard

    def load_module(self, module):
        names = tuple(name for name, _ in module.named_parameters())
        expected = {name.removeprefix('mtp.') for name in HF_MTP_SHAPES
                    if not name.startswith('mtp.layers.0.mlp.')}
        if set(names) != expected:
            raise ValueError('canonical MTP module parameters differ from the HF GGUF layout')
        kv = tuple(head * 256 for head in self.shard.replicated_partition(2))
        extents = {
            'layers.0.self_attn.q_proj.weight': slice(*self.shard.partition(12288)),
            'layers.0.self_attn.k_proj.weight': slice(*kv),
            'layers.0.self_attn.v_proj.weight': slice(*kv),
            'layers.0.self_attn.o_proj.weight': (slice(None), slice(*self.shard.partition(6144))),
        }
        for name in names:
            value = self.load_tensor('mtp.' + name, extent=extents.get(name))
            if value.ndim == 1:
                if value.dtype != torch.float32 or not torch.isfinite(value).all().item():
                    raise ValueError(f'mtp.{name}: norm requires finite stored FP32 parameters')
                # Strata stores the original zero-centered w, not 1+w. Only
                # the feedback ABI's BF16 representation needs an exact cast.
                if name.startswith('pre_fc_norm_'):
                    converted = value.bfloat16()
                    if not torch.equal(converted.float(), value):
                        raise ValueError(f'mtp.{name}: norm cannot retain its values in BF16')
                    value = converted
            _parameter(module, name, value)
        for mixer in (module.layers[0].attn_hyper_connection, module.layers[0].mlp_hyper_connection):
            packed_down_projection(mixer)
        attention = module.layers[0].self_attn
        attention.compress_ratio = 0
        qkv = torch.cat([getattr(attention, name + '_proj').weight.detach() for name in ('q', 'k', 'v')])
        widths = (12288 // self.shard.world_size, max(1, 2 // self.shard.world_size) * 256,
                  max(1, 2 // self.shard.world_size) * 256)
        for part, weight in zip(('q', 'k', 'v'), qkv.split(widths), strict=True):
            _parameter(attention, part + '_proj.weight', weight)
        attention.register_buffer('packed_qkv', qkv, persistent=False)
        return {'qkv_proj': qkv, 'o_proj': attention.o_proj.weight,
                'indexer.index_qk_proj': attention.indexer.index_qk_proj.weight}

    def load_moe(self):
        import gguf

        gate_name, down_name = HF_EXPERTS
        block, _ = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType[self.manifest.tensors[down_name].encoding]]
        first, last = self.shard.partition(640, alignment=block)
        if (first, last) == (0, 640):
            gate_up = self.load_quantized(gate_name)
        else:
            # Slice each half inside each expert; the stored order is gate, up.
            gate_up = self.load_quantized_group((gate_name, gate_name),
                row_ranges=((first, last), (640 + first, 640 + last)))
        if self.quactlize_packing is not None:
            gate_up = self.quactlize_packing(gate_up)
        down = self.load_quantized(down_name, columns=(first, last))
        if self.quactlize_packing is not None:
            down = self.quactlize_packing(down)
        prefix = 'mtp.layers.0.mlp'
        extent = slice(first, last)
        shared = SharedExpertWeights(torch.cat(self.load_tensors(
            tuple(f'{prefix}.shared_expert.{part}_proj.weight' for part in ('gate', 'up')),
            extents=(extent, extent))),
            self.load_tensor(f'{prefix}.shared_expert.down_proj.weight', extent=(slice(None), extent)),
            self.load_tensor(f'{prefix}.shared_expert_gate.weight'))
        return QwenMoeWeights(self.load_tensor(f'{prefix}.gate.weight'), GgufExperts(gate_up, down), shared)
