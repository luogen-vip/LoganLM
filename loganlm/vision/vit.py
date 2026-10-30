# -*- coding: utf-8 -*-
"""
LoganLM ViT —— 从 0 手写的 Vision Transformer
===============================================

作者：罗根（Logan）

不使用任何预训练权重、torchvision 模型。组件：
  - Patch Embedding：Conv2d 把图片切成 patch 并线性映射
  - 可学习位置嵌入
  - Pre-Norm 自注意力块 + MLP
  - 三个分类头：颜色 / 形状 / 位置（合成数据多任务训练）

前向返回 (logits_color, logits_shape, logits_pos, patch_feats)，
patch_feats 交给多模态桥接（bridge.py）喂给语言模型。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..config import VisionConfig


class Block(nn.Module):
    """ViT 基础块：LayerNorm + 多头自注意力 + MLP（Pre-Norm）。"""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: float, dropout: float = 0.0) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(B, N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # (B,H,N,hd) ×3
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.softmax(dim=-1)
        att = self.drop(att)
        out = (att @ v).transpose(1, 2).reshape(B, N, D)
        x = x + self.drop(self.proj(out))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    def __init__(self, cfg: VisionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        n_patches = cfg.num_patches
        self.patch_embed = nn.Conv2d(cfg.in_chans, cfg.dim,
                                     kernel_size=cfg.patch_size, stride=cfg.patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, cfg.dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList(
            Block(cfg.dim, cfg.n_heads, cfg.mlp_ratio, cfg.dropout) for _ in range(cfg.depth)
        )
        self.norm = nn.LayerNorm(cfg.dim)
        self.head_color = nn.Linear(cfg.dim, cfg.num_colors)
        self.head_shape = nn.Linear(cfg.dim, cfg.num_shapes)
        self.head_pos = nn.Linear(cfg.dim, cfg.num_positions)

    def forward(self, px: torch.Tensor):
        """px: (B,3,H,W) 归一化到 [-1,1]。"""
        B = px.shape[0]
        x = self.patch_embed(px).flatten(2).transpose(1, 2)  # (B,N,dim)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        pooled = x.mean(dim=1)  # (B,dim)
        feats = x  # (B,N,dim) 交给桥接
        return (self.head_color(pooled), self.head_shape(pooled),
                self.head_pos(pooled), feats)

    def predict(self, px: torch.Tensor):
        """推理辅助：返回 (color_idx, shape_idx, pos_idx)。"""
        self.eval()
        with torch.no_grad():
            lc, ls, lp, _ = self.forward(px.unsqueeze(0) if px.dim() == 3 else px)
            return (int(lc[0].argmax()), int(ls[0].argmax()), int(lp[0].argmax()))


if __name__ == "__main__":
    v = ViT(VisionConfig(image_size=64, patch_size=16, dim=96, depth=3, n_heads=4))
    px = torch.randn(2, 3, 64, 64)
    lc, ls, lp, f = v(px)
    print("logits:", lc.shape, ls.shape, lp.shape, "| feats:", f.shape,
          "| 参数:", sum(p.numel() for p in v.parameters()))
    print("ViT 自测通过 ✅")
