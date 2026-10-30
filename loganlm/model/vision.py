# -*- coding: utf-8 -*-
"""
LoganLM · 视觉编码器（从 0 手写的 Vision Transformer）
======================================================

把一张图片切成 16x16 的小块（patch），每一块当一个"视觉单词"：
  图片 [b,3,128,128] ─ Conv2d 切块 ─ [b, 64, dim]（64 个 patch token）
                      + 可学习位置编码
                      ─ [LN → 多头自注意力 → LN → MLP] × depth
                      ─► patch 特征 + 平均池化特征 + 三个分类头

三个分类头用于视觉预训练（合成的 形状/颜色/位置 数据集，见 data/synthetic.py）：
  * 颜色头：红/橙/黄/绿/蓝/紫/黑/白
  * 形状元：圆形/方形/三角形/星形/爱心/十字
  * 位置头：主体落在 3x3 网格的哪一格

作者：罗根（Logan）
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import VisionConfig


# ---------------------------------------------------------------------------
# 多头自注意力（视觉侧，双向注意力，无因果掩码）
# ---------------------------------------------------------------------------
class VisionAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int) -> None:
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        qkv = self.qkv(x).view(b, s, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]              # [b, h, s, hd]
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, s, d)
        return self.proj(out)


class VisionBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = VisionAttention(dim, n_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# ViT 主干
# ---------------------------------------------------------------------------
class VisionEncoder(nn.Module):
    def __init__(self, cfg: VisionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        n_patches = cfg.num_patches

        # 切块投影：Conv2d(kernel=patch, stride=patch) 一步完成切块+线性映射
        self.patch_embed = nn.Conv2d(
            cfg.in_chans, cfg.dim, kernel_size=cfg.patch_size, stride=cfg.patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, cfg.dim))
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([
            VisionBlock(cfg.dim, cfg.n_heads, cfg.mlp_ratio, cfg.dropout)
            for _ in range(cfg.depth)
        ])
        self.norm = nn.LayerNorm(cfg.dim)

        # 三个预训练分类头
        self.head_color = nn.Linear(cfg.dim, cfg.num_colors)
        self.head_shape = nn.Linear(cfg.dim, cfg.num_shapes)
        self.head_pos = nn.Linear(cfg.dim, cfg.num_positions)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init)

    def _init(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.Conv2d)):
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias)
            if hasattr(m, "weight") and m.weight is not None and m.weight.dim() > 1:
                nn.init.trunc_normal_(m.weight, std=0.02)

    # ------------------------------------------------------------------
    def encode(self, pixels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """像素 [b,3,H,W] → (patch 特征 [b,N,dim], 平均池化特征 [b,dim])"""
        x = self.patch_embed(pixels)                  # [b, dim, H/p, W/p]
        x = x.flatten(2).transpose(1, 2)              # [b, N, dim]
        x = self.drop(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        pooled = x.mean(dim=1)
        return x, pooled

    def forward(self, pixels: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats, pooled = self.encode(pixels)
        return {
            "patch_feats": feats,
            "pooled": pooled,
            "logits_color": self.head_color(pooled),
            "logits_shape": self.head_shape(pooled),
            "logits_pos": self.head_pos(pooled),
        }

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
