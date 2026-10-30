# -*- coding: utf-8 -*-
"""
LoganLM 多模态桥接 —— 把视觉特征"翻译"进语言模型的世界
========================================================

作者：罗根（Logan）

LLaVA 式思路（从0重写）：
    ViT patch 特征 (B, N, Dv)
      → Perceiver 重采样器：Q 个可学习查询 token 对 patch 做交叉注意力，
        压缩成固定数量的"图像软 token" (B, Q, Dv)
      → MLP 投影到 LLM 的 embedding 维度 (B, Q, D)
      → 作为前缀软 token 拼在文本 embedding 前面喂给语言模型
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, dim * 2, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """queries: (B,Q,D)，context: (B,N,D) → (B,Q,D)。"""
        B, Q, D = queries.shape
        N = context.shape[1]
        q = self.q(self.norm_q(queries)).view(B, Q, self.n_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(self.norm_kv(context)).view(B, N, 2, self.n_heads, self.head_dim)
        k, v = kv.permute(2, 0, 3, 1, 4).unbind(0)
        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        out = (att.softmax(dim=-1) @ v).transpose(1, 2).reshape(B, Q, D)
        return self.proj(out)


class PerceiverResampler(nn.Module):
    def __init__(self, dim: int, depth: int = 2, n_heads: int = 4,
                 num_query_tokens: int = 4) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_query_tokens, dim) * 0.02)
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.ModuleDict({
                "cross": CrossAttention(dim, n_heads),
                "ffn": nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 2),
                                     nn.GELU(), nn.Linear(dim * 2, dim)),
            }))

    def forward(self, patch_feats: torch.Tensor) -> torch.Tensor:
        """(B,N,D) → (B,Q,D)"""
        x = self.queries.expand(patch_feats.shape[0], -1, -1)
        for blk in self.blocks:
            x = x + blk["cross"](x, patch_feats)
            x = x + blk["ffn"](x)
        return x


class VisionBridge(nn.Module):
    """视觉 → LLM 的完整桥梁。"""

    def __init__(self, vision_dim: int, llm_dim: int, hidden_dim: int = 160,
                 num_query_tokens: int = 4, depth: int = 2, n_heads: int = 4) -> None:
        super().__init__()
        self.resampler = PerceiverResampler(vision_dim, depth=depth, n_heads=n_heads,
                                            num_query_tokens=num_query_tokens)
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, llm_dim),
        )
        self.num_query_tokens = num_query_tokens

    def forward(self, patch_feats: torch.Tensor) -> torch.Tensor:
        """(B,N,Dv) → (B,Q,D_llm)：可直接作为 LLM 的前缀软 token。"""
        return self.proj(self.resampler(patch_feats))


if __name__ == "__main__":
    b = VisionBridge(vision_dim=96, llm_dim=192)
    out = b(torch.randn(2, 16, 96))
    print("桥接输出:", tuple(out.shape), "| 参数:", sum(p.numel() for p in b.parameters()))
    print("桥接自测通过 ✅")
