# -*- coding: utf-8 -*-
"""
LoganLM · 语言模型主干（从 0 手写的 decoder-only Transformer）
============================================================

结构（LLaMA 同构，代码全部重写）：

    token ids ─► Embedding ─► [RMSNorm → 因果自注意力(GQA+RoPE) → 残差
                               RMSNorm → SwiGLU 前馈 → 残差] × n_layers
              ─► RMSNorm ─► lm_head(共享 embedding 权重) ─► logits

特性：
  * 支持 inputs_embeds（多模态图像软 token 直接拼进 embedding）
  * 支持 KV Cache 增量解码（聊天逐 token 生成时不用重算历史）
  * 支持返回交叉熵 loss（训练用）

作者：罗根（Logan）
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import LLMConfig
from .layers import RMSNorm, TransformerBlock, init_linear, precompute_rope


class LoganLMTransformer(nn.Module):
    """LoganLM 语言模型主干。"""

    def __init__(self, cfg: LLMConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.dim, cfg.n_heads, cfg.n_kv_heads,
                             cfg.ffn_hidden, cfg.max_seq_len, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.norm_f = RMSNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight  # 共享权重

        cos, sin = precompute_rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # 残差投影层用更小的 std，深层更稳
        for block in self.blocks:
            init_linear(block.attn.wo, cfg.n_layers)
            init_linear(block.ffn.w2, cfg.n_layers)

    # ------------------------------------------------------------------
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    def forward(
        self,
        idx: Optional[torch.Tensor] = None,       # [b, s] token id（或为空，用 inputs_embeds）
        inputs_embeds: Optional[torch.Tensor] = None,  # [b, s, dim]（多模态软 token）
        targets: Optional[torch.Tensor] = None,   # [b, s] 训练标签（-100 忽略）
        start_pos: int = 0,                       # KV cache 写入起始位置
        use_cache: bool = False,                  # 推理时增量写入 KV cache
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """返回 (logits, loss)。loss 仅在 targets 给出时计算。"""
        b, s = idx.shape if idx is not None else inputs_embeds.shape[:2]
        if start_pos + s > self.cfg.max_seq_len:
            raise ValueError(
                f"序列长度超出上限：{start_pos}+{s} > {self.cfg.max_seq_len}，"
                f"请截断历史或增大 max_seq_len")

        x = self.tok_emb(idx) if idx is not None else inputs_embeds
        past_len = start_pos  # 注意力里 query 全局位置偏移

        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin, past_len=past_len,
                      use_cache=use_cache)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # 关键：next-token 移位。logits[t] 预测 targets[t+1]，
            # 即"给定前缀 ≤ t，下一个词是什么"——这才是语言模型该学的。
            shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
            shift_labels = targets[:, 1:].reshape(-1)
            loss = F.cross_entropy(shift_logits.float(), shift_labels.long(),
                                   ignore_index=-100)
        return logits, loss

    # ------------------------------------------------------------------
    # KV Cache 管理（推理用）
    # ------------------------------------------------------------------
    def reset_cache(self, batch_size: int = 1) -> None:
        device = self.rope_cos.device
        dtype = self.tok_emb.weight.dtype
        for block in self.blocks:
            block.attn.reset_cache(batch_size, device, dtype)

    def free_cache(self) -> None:
        for block in self.blocks:
            block.attn.cache_k = None
            block.attn.cache_v = None

    # ------------------------------------------------------------------
    def count_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
            if self.cfg.tie_weights:
                n += self.tok_emb.weight.numel()  # 共享权重别扣两次
        return n

    # ------------------------------------------------------------------
    def num_cached_tokens(self) -> int:
        ck = self.blocks[0].attn.cache_k
        return 0 if ck is None else ck.size(2)
