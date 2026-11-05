# SPDX-License-Identifier: Apache-2.0

"""Qwen GGUF tensor-name mappings, retaining the checkpoint's packed formats."""

from ignition.components.gguf_moe import GgufExperts
from ignition.components.gguf_weights import GgufWeightLoader
from ignition.components.parallel import TensorParallelShard
from .moe import QwenMoeWeights, SharedExpertWeights


def _parameter(module, path, value):
    import torch

    parent, _, field = path.rpartition('.')
    setattr(module.get_submodule(parent), field, torch.nn.Parameter(value, requires_grad=False))


def _restore_value_heads(value, axis=0, head_dim=128):
    """Undo llama.cpp's [V-per-K,K-head,D] storage permutation."""
    shape = value.shape
    if shape[axis] != 48 * head_dim:
        raise ValueError('Qwen GDN value-head axis must contain 48 heads')
    split = (*shape[:axis], 3, 16, head_dim, *shape[axis + 1:])
    return value.reshape(split).transpose(axis, axis + 1).reshape(shape).contiguous()


class QwenGgufWeightLoader(GgufWeightLoader):
    def __init__(self, *args, shard=TensorParallelShard(0, 1), **kwargs):
        super().__init__(*args, **kwargs)
        if self.manifest.metadata.get('general.architecture') != 'qwen4exp':
            raise ValueError('Qwen GGUF loader requires qwen4exp checkpoint metadata')
        self.shard = shard

    def load_dense(self, name, *, rows=None, columns=None):
        """Keep stored plain precision; cache quantized dense matrices in BF16."""
        import gguf

        metadata = self.manifest.tensors[name]
        if gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType[metadata.encoding]][0] > 1:
            return self.load_dequantized(name, rows=rows, columns=columns)
        extent = (slice(*(rows or (0, metadata.shape[0]))),
                  slice(*(columns or (0, metadata.shape[1]))))
        return self.load_tensor(name, extent=extent)

    def load_norm_delta(self, name, *, dtype=None):
        """Recover zero-centered parameters only when the stored gain is exact."""
        import torch

        gain = self.load_tensor(name)
        if gain.dtype != torch.float32 or not torch.isfinite(gain).all().item():
            raise ValueError(f'{name}: GGUF norm gain must be finite FP32')
        delta = gain - 1.
        if not torch.equal((delta + 1.).view(torch.int32), gain.view(torch.int32)):
            raise ValueError(f'{name}: stored norm gain is not exactly zero-center reversible')
        if dtype is not None:
            converted = delta.to(dtype)
            if not torch.equal(converted.float(), delta):
                raise ValueError(f'{name}: zero-centered weight cannot be represented exactly in {dtype}')
            delta = converted
        return delta

    def load_hyperconnection(self, module, prefix):
        import torch
        from .hyperconnection import packed_down_projection

        _parameter(module, 'hc_norm.weight', self.load_norm_delta(f'{prefix}_norm.weight'))
        for part in ('down', 'up'):
            _parameter(module, f'input_mix_weight_{part}.weight', self.load_dense(f'{prefix}_{part}.weight'))
        if module.block_inject_weight is not None:
            injection = self.load_dense(f'{prefix}_inject.weight')
            # Packing shares the down projection's dtype; permit only an exact cast.
            cast = injection.to(module.input_mix_weight_down.weight.dtype)
            if not torch.equal(cast.float(), injection.float()):
                raise ValueError(f'{prefix}: residual injection cannot be packed without precision loss')
            _parameter(module, 'block_inject_weight.weight', cast)
        packed_down_projection(module)

    def load_attention(self, module, layer):
        import torch

        prefix = f'blk.{layer}'
        kv = tuple(head * 256 for head in self.shard.replicated_partition(2))
        extents = (self.shard.partition(12288), kv, kv)
        pieces = [self.load_dense(f'{prefix}.attn_{part}.weight', rows=extent)
                  for part, extent in zip(('q', 'k', 'v'), extents, strict=True)]
        for part in ('q', 'k'):
            _parameter(module, f'{part}_norm.weight', self.load_norm_delta(f'{prefix}.attn_{part}_norm.weight'))
        result = dict(qkv_proj=torch.cat(pieces),
                      o_proj=self.load_dense(f'{prefix}.attn_output.weight', columns=self.shard.partition(6144)))
        ratios = self.manifest.metadata['qwen4exp.attention.compress_ratios']
        module.compress_ratio = ratios[layer]
        if module.compress_ratio not in (0, 4):
            raise ValueError('Qwen attention requires full attention or compression ratio four')
        if module.compress_ratio:
            result['indexer.index_qk_proj'] = torch.cat([
                self.load_dense(f'{prefix}.indexer.{part}_proj.weight') for part in ('q', 'k')])
            for part in ('q', 'k'):
                _parameter(module.indexer, f'{part}_layernorm.weight',
                           self.load_norm_delta(f'{prefix}.indexer.{part}_norm.weight'))
        return result

    def load_gdn(self, module, layer):
        import torch

        prefix = f'blk.{layer}'
        qkv = self.load_dense(f'{prefix}.attn_qkv.weight')
        q, k, v = qkv.split((2048, 2048, 6144))
        value = self.shard.partition(6144)
        key = self.shard.partition(2048)
        heads = self.shard.partition(48)
        qkvz = torch.cat((q[slice(*key)], k[slice(*key)], _restore_value_heads(v)[slice(*value)],
                         _restore_value_heads(self.load_dense(f'{prefix}.attn_gate.weight'))[slice(*value)]))
        ba = torch.cat([_restore_value_heads(self.load_dense(f'{prefix}.ssm_{part}.weight'), head_dim=1)[slice(*heads)]
                        for part in ('beta', 'alpha')])
        out = _restore_value_heads(self.load_dense(f'{prefix}.ssm_out.weight'), axis=1)[:, slice(*value)].contiguous()
        convolution = self.load_tensor(f'{prefix}.ssm_conv1d.weight')
        converted = convolution.bfloat16()
        if not torch.equal(converted.float(), convolution):
            raise ValueError(f'{prefix}: convolution cannot retain its checkpoint values in BF16')
        convolution = converted
        cq, ck, cv = convolution.split((2048, 2048, 6144))
        convolution = torch.cat((cq[slice(*key)], ck[slice(*key)], _restore_value_heads(cv)[slice(*value)])).unsqueeze(1)
        _parameter(module, 'conv1d.weight', convolution)
        _parameter(module, 'dt_bias', _restore_value_heads(self.load_tensor(f'{prefix}.ssm_dt.bias'), head_dim=1)[slice(*heads)])
        _parameter(module, 'norm.weight', self.load_tensor(f'{prefix}.ssm_norm.weight'))
        stored = _restore_value_heads(self.load_tensor(f'{prefix}.ssm_a'), head_dim=1)[slice(*heads)]
        # The converter stores -exp(A_log.float()), with the original BF16
        # parameter exactly recoverable. Verify it with the same CPU transform.
        candidate = (-stored.cpu().double()).log().bfloat16().float()
        if not torch.equal(-candidate.exp(), stored.cpu()):
            raise ValueError(f'{prefix}: cannot exactly recover the checkpoint GDN A_log parameter')
        _parameter(module, 'A_log', candidate.to(stored.device))
        module.num_k_heads //= self.shard.world_size
        module.num_v_heads //= self.shard.world_size
        module.conv_dim //= self.shard.world_size
        return dict(in_proj_qkvz=qkvz, in_proj_ba=ba, out_proj=out)

    def load_moe(self, layer: int, *, cache_shared=False):
        """Map target or shared-MTP expert names into the existing Qwen component."""
        if type(layer) is not int or layer < 0:
            raise ValueError('Qwen GGUF layer must be a nonnegative integer')
        if (getattr(getattr(self.manifest, 'quactlize', None), 'target', None) in ('ppu10', 'ppu15')
                and self.quactlize_packing is None):
            raise ValueError('PPU routed experts require the selected QUactlize canonical weight producer')
        prefix = f'blk.{layer}'
        # Down projections constrain the intermediate partition's block boundary.
        import gguf

        down = self.manifest.tensors[f'{prefix}.ffn_down_exps.weight']
        shared_down = self.manifest.tensors[f'{prefix}.ffn_down_shexp.weight']
        if down.shape != (512, 2560, 640) or shared_down.shape != (2560, 640):
            raise ValueError('Qwen GGUF requires 512 experts with hidden=2560 and intermediate=640')
        align = max(gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType[t.encoding]][0]
                    for t in (down, shared_down))
        extent = self.shard.partition(640, alignment=align)
        gate_up = self.load_quantized_group(tuple(f'{prefix}.ffn_{part}_exps.weight' for part in ('gate', 'up')),
                                           row_ranges=(extent, extent))
        if self.quactlize_packing is not None:
            gate_up = self.quactlize_packing(gate_up)
        down_weight = self.load_quantized(f'{prefix}.ffn_down_exps.weight', columns=extent)
        if self.quactlize_packing is not None:
            down_weight = self.quactlize_packing(down_weight)
        experts = GgufExperts(gate_up, down_weight)
        if cache_shared:
            import torch

            gate_up = torch.cat([self.load_dense(f'{prefix}.ffn_{part}_shexp.weight', rows=extent)
                                 for part in ('gate', 'up')])
            shared_down = self.load_dense(f'{prefix}.ffn_down_shexp.weight', columns=extent)
        else:
            gate_up = self.load_quantized_group(tuple(f'{prefix}.ffn_{part}_shexp.weight' for part in ('gate', 'up')),
                                                row_ranges=(extent, extent))
            shared_down = self.load_quantized(f'{prefix}.ffn_down_shexp.weight', columns=extent)
        shared = SharedExpertWeights(gate_up, shared_down,
                                     self.load_tensor(f'{prefix}.ffn_gate_inp_shexp.weight').view(1, 2560))
        return QwenMoeWeights(self.load_tensor(f'{prefix}.ffn_gate_inp.weight'), experts, shared)


def load_model(manifest, device, *, progress=None, shard=TensorParallelShard(0, 1),
               quactlize_helpers=None, quactlize_packing=None):
    import torch
    from .decoder import DecoderLayerWeights, ModelWeights
    from .model import create_model_tree

    tree = create_model_tree(manifest)
    before = torch.cuda.memory_allocated(device)
    layers = []
    language = tree.model.language_model
    with QwenGgufWeightLoader(manifest, device, progress=progress, shard=shard,
            read_mode=manifest.read_mode, quactlize_helpers=quactlize_helpers, quactlize_packing=quactlize_packing) as loader:
        embedding = loader.load_quantized('token_embd.weight')
        _parameter(tree, 'lm_head.weight', loader.load_dense('output.weight', rows=shard.partition(248320)))
        loader.load_hyperconnection(language.hyper_connection_mixer, 'output_hc')
        for index, module in enumerate(language.layers):
            for label, field in (('attn', 'attn_hyper_connection'), ('ffn', 'mlp_hyper_connection')):
                loader.load_hyperconnection(getattr(module, field), f'blk.{index}.hc_{label}')
            if index % 4 == 3:
                attention = loader.load_attention(module.self_attn, index)
            else:
                attention = loader.load_gdn(module.linear_attn, index)
            moe = loader.load_moe(index, cache_shared=True)
            _parameter(module.mlp, 'gate.weight', moe.router)
            _parameter(module.mlp, 'shared_expert_gate.weight', moe.shared.gate)
            layers.append(DecoderLayerWeights(attention, moe))
        ple = language.layers[1].ple
        for part in ('key', 'value'):
            _parameter(ple, part + '_proj.weight', loader.load_dense(f'blk.1.ple_{part}.weight'))
        convolution = loader.load_tensor('blk.1.ple_conv1d.weight')
        if manifest.actlize is not None:
            converted = convolution.bfloat16()
            if not torch.equal(converted.float(), convolution.float()):
                raise ValueError('PLE convolution cannot retain its checkpoint values in BF16')
            convolution = converted
        _parameter(ple, 'conv1d.weight', convolution.unsqueeze(1))
        for part in ('key', 'query', 'conv'):
            _parameter(ple, f'norm_{part}.weight', loader.load_norm_delta(f'blk.1.ple_norm_{part}.weight'))
    torch.cuda.synchronize(device)
    return ModelWeights(tree, tuple(layers), None, torch.cuda.memory_allocated(device) - before, shard, embedding)
