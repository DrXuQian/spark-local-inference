# SPDX-License-Identifier: Apache-2.0

"""Validate the two stored Qwen MTP layouts before model allocation."""

from .checkpoint import CheckpointError


HF_EXPERTS = ('mtp.layers.0.mlp.experts.gate_up_proj', 'mtp.layers.0.mlp.experts.down_proj')

# Strata tools/mtp_pack.py preserves the original HF names and zero-centered
# norm parameters. Its GGUF is not the llama.cpp blk.48 layout with norm gains.
HF_MTP_SHAPES = {
    'mtp.fc_embedding.weight': (2560, 2560),
    'mtp.fc_hidden.weight': (2560, 2560),
    'mtp.pre_fc_norm_embedding.weight': (2560,),
    'mtp.pre_fc_norm_hidden.weight': (10240,),
    HF_EXPERTS[0]: (512, 1280, 2560),
    HF_EXPERTS[1]: (512, 2560, 640),
    'mtp.layers.0.mlp.gate.weight': (512, 2560),
    'mtp.layers.0.mlp.shared_expert_gate.weight': (1, 2560),
    **{f'mtp.layers.0.mlp.shared_expert.{part}_proj.weight': shape for part, shape in
       (('gate', (640, 2560)), ('up', (640, 2560)), ('down', (2560, 640)))},
    **{f'{prefix}.{part}.weight': shape
       for prefix in ('mtp.hyper_connection_mixer', 'mtp.layers.0.attn_hyper_connection',
                      'mtp.layers.0.mlp_hyper_connection')
       for part, shape in (('hc_norm', (10240,)), ('input_mix_weight_down', (320, 10240)),
                           ('input_mix_weight_up', (10240, 320)))},
    **{f'mtp.layers.0.{part}_hyper_connection.block_inject_weight.weight': (4, 10240)
       for part in ('attn', 'mlp')},
    **{f'mtp.layers.0.self_attn.{part}.weight': shape for part, shape in
       (('q_proj', (12288, 2560)), ('k_proj', (512, 2560)), ('v_proj', (512, 2560)),
        ('o_proj', (2560, 6144)), ('q_norm', (256,)), ('k_norm', (256,)),
        ('indexer.index_qk_proj', (640, 2560)), ('indexer.q_layernorm', (128,)),
        ('indexer.k_layernorm', (128,)))},
}


def validate_mtp_gguf(checkpoint):
    """Return the stored architecture without relabeling names, norms or bytes."""
    architecture = checkpoint.metadata.get('general.architecture')
    if architecture == 'qwen4exp-mtp':
        missing = HF_MTP_SHAPES.keys() - checkpoint.tensors.keys()
        extra = checkpoint.tensors.keys() - HF_MTP_SHAPES.keys()
        if missing or extra:
            raise CheckpointError(f'MTP qwen4exp-mtp tensor coverage differs: missing={sorted(missing)}, extra={sorted(extra)}')
        for name, shape in HF_MTP_SHAPES.items():
            tensor = checkpoint.tensors[name]
            if tensor.shape != shape:
                raise CheckpointError(f'MTP {name} requires shape {shape}, found {tensor.shape}')
            if name in HF_EXPERTS:
                if tensor.dtype != 'U8':
                    raise CheckpointError(f'MTP {name} requires quantized blocks')
            else:
                expected = 'F32' if len(shape) == 1 else 'BF16'
                if tensor.encoding != expected:
                    raise CheckpointError(f'MTP {name} requires stored {expected}, found {tensor.encoding}')
        gate, down = (checkpoint.tensors[name] for name in HF_EXPERTS)
        declared = checkpoint.metadata.get('strata.mtp.expert_format')
        if (not isinstance(declared, str) or declared not in ('q2_0', 'q4_0', 'q8_0')
                or gate.encoding != declared.upper() or down.encoding != gate.encoding):
            raise CheckpointError('MTP expert types differ from strata.mtp.expert_format')
        # The complete name set has no target embedding/head and exactly one
        # draft layer. The caller supplies shared endpoints and pinned config.
        return architecture

    expected = {'general.architecture': 'qwen4exp', 'qwen4exp.block_count': 49,
                'qwen4exp.nextn_predict_layers': 1, 'qwen4exp.nextn_shared_target_tensors': True}
    for name, value in expected.items():
        actual = checkpoint.metadata.get(name)
        if actual != value or type(actual) is not type(value):
            raise CheckpointError(f'MTP {name}={actual!r}; the current shared-MTP fixture requires {value!r}')
    if not checkpoint.tensors or any(not name.startswith('blk.48.') for name in checkpoint.tensors):
        raise CheckpointError('the current shared-MTP fixture requires only blk.48.* tensors')
    ratios = checkpoint.metadata.get('qwen4exp.attention.compress_ratios')
    if not isinstance(ratios, tuple) or len(ratios) != 49 or ratios[48] != 0:
        raise CheckpointError('MTP requires an uncompressed layer 48 in attention.compress_ratios')
    for part, shape in (('gate', (512, 640, 2560)), ('up', (512, 640, 2560)),
                        ('down', (512, 2560, 640))):
        name = f'blk.48.ffn_{part}_exps.weight'
        tensor = checkpoint.tensors.get(name)
        if tensor is None or tensor.shape != shape:
            raise CheckpointError(f'MTP {name} requires shape {shape}')
        if tensor.dtype != 'U8':
            raise CheckpointError(f'MTP {name} uses {tensor.encoding}; the current GGUF expert path requires quantized blocks')
    gate, up = (checkpoint.tensors[f'blk.48.ffn_{part}_exps.weight'] for part in ('gate', 'up'))
    if gate.encoding != up.encoding:
        raise CheckpointError('MTP gate/up experts require equal encodings for the existing fused load')
    return architecture
