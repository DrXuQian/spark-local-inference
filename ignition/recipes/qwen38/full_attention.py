# SPDX-License-Identifier: Apache-2.0

"""Qwen's uncompressed MTP attention and cache-only absorption."""

import torch

from ignition.components.linear import prepare_projection


class PreparedFullAttention:
    def __init__(self, module, weights, source, scratch, state, metadata, rotary, *,
                 session, name, draft_tokens=0, workspace=None, projection_workspace=None,
                 max_cache_pages=None, cache_only=False):
        self.module, self.scratch, self.state = module, scratch, state
        self.metadata, self.rotary, self.cache_only = metadata, rotary, cache_only
        self.tokens, self.q_heads, self.kv_heads = source.shape[0], scratch.query.shape[1], scratch.key.shape[1]
        if (source.shape != scratch.output.shape or source.dtype != torch.bfloat16
                or bool((metadata.request_ids != -1).any().item())
                or bool((metadata.slot_mapping != -1).any().item())):
            raise ValueError('full attention preparation requires matching BF16 inputs and inactive metadata')
        self.cache_page_capacity = state.main_k_cache.shape[0] if max_cache_pages is None else max_cache_pages
        if self.cache_page_capacity < state.main_k_cache.shape[0]:
            raise ValueError('planned full-attention pages must cover the bound cache')
        self.qkv = prepare_projection(source, weights['qkv_proj'], scratch.projected_qkv,
            session=session, name=name + '.qkv', workspace=projection_workspace)
        fa3 = getattr(session, 'flash_attention3', None)
        operands = (scratch.key, scratch.projected_v.view(self.tokens, self.kv_heads, 256),
                    state.main_k_cache, state.main_v_cache, metadata.slot_mapping)
        if fa3 is not None:
            from ignition.components.flash_attn3 import PreparedPagedKVWriter

            self.writer = PreparedPagedKVWriter(*operands, workspace=workspace, max_cache_pages=self.cache_page_capacity)
        else:
            from ignition.components.paged_attention import PreparedPagedKVWriter

            self.writer = PreparedPagedKVWriter(*operands, session=session, name=name + '.kv_write')
        if cache_only:
            return
        self.projection = prepare_projection(scratch.gated, weights['o_proj'], scratch.output,
            session=session, name=name + '.output', workspace=projection_workspace)
        batch = metadata.main_block_table.shape[0]
        width = (self.tokens + batch - 1) // batch
        if fa3 is not None:
            from ignition.components.flash_attn3 import PreparedFlashAttention3

            self.attention = PreparedFlashAttention3(fa3, scratch.query, state.main_k_cache, state.main_v_cache,
                metadata.main_block_table, metadata.sequence_lengths, metadata.query_start_loc,
                scratch.attended, width=width)
            return
        from ignition.components.paged_attention import PreparedPagedAttention

        # Every invocation is a suffix of each request's cache. Extend handles
        # absorption/prefill; short suffixes use paged GQA's verification schedule.
        mode = 'decode' if width == 1 else 'verify' if width <= 8 else 'extend'
        self.attention = PreparedPagedAttention(scratch.query, state.main_k_cache, state.main_v_cache,
            metadata.main_block_table, metadata.sequence_lengths, metadata.query_start_loc,
            scratch.attended, mode=mode, width=width, session=session, workspace=workspace,
            name=name + '.full', k_descale=self.writer.arguments['k_scale'],
            v_descale=self.writer.arguments['v_scale'])

    @property
    def device_errors(self):
        return getattr(self.writer, 'device_errors', ())

    def rebind_pages_for_preparation(self, state):
        if state.main_k_cache.shape[0] > self.cache_page_capacity:
            raise ValueError('replacement cache exceeds prepared full-attention page capacity')
        for name in ('main_k_cache', 'main_v_cache'):
            old, new = getattr(self.state, name), getattr(state, name)
            if (old.shape[1:] != new.shape[1:] or old.stride() != new.stride()
                    or old.dtype != new.dtype or old.device != new.device):
                raise ValueError('full-attention rebinding must preserve the page ABI')
        self.writer.arguments.update(k_cache=state.main_k_cache, v_cache=state.main_v_cache)
        if hasattr(self.writer, 'rebind'):
            self.writer.rebind()
        if not self.cache_only:
            self.attention.arguments.update(k_cache=state.main_k_cache, v_cache=state.main_v_cache)
            self.attention.rebind()
        self.state = state

    def __call__(self, source):
        from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb

        s, m = self.scratch, self.metadata
        self.qkv(source)
        projected = s.projected_q.view(self.tokens, self.q_heads, 512)
        query, gate = projected[..., :256], projected[..., 256:]
        key = s.projected_k.view(self.tokens, self.kv_heads, 256)
        cos, sin = self.rotary.gather(m.rope_positions)
        query, key = apply_rotary_pos_emb(self.module.q_norm(query), self.module.k_norm(key), cos, sin,
                                         unsqueeze_dim=1)
        s.query.copy_(query)
        s.key.copy_(key)
        self.writer()
        if self.cache_only:
            return None
        attended = self.attention()
        s.gated.copy_(attended.flatten(1) * torch.sigmoid(gate.reshape(self.tokens, -1)))
        result = self.projection(s.gated)
        result.masked_fill_((m.request_ids < 0)[:, None], 0)
        return result
