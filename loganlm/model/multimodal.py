# -*- coding: utf-8 -*-
"""
LoganLM · 多模态融合（视觉 → 语言的桥，从 0 手写）
==================================================

思路与 LLaVA / Flamingo 一脉相承，但代码是重写的：

  1) Perceiver 重采样器：一组可学习的"查询 token"去 attend 视觉 patch 特征，
     把 任意数量 的 patch 压缩成固定 8 个图像软 token（长度与图片大小无关）；
  2) MLP 投影：把视觉空间特征翻译到语言模型 embedding 空间；
  3) MultimodalLoganLM：把图像软 token 直接拼接在文本 token embedding 前，
     喂给 LoganLMTransformer —— 模型"看图"之后继续"说话"。

作者：罗根（Logan）
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import LLMConfig, MultiModalConfig, VisionConfig
from .transformer import LoganLMTransformer
from .vision import VisionEncoder


class CrossAttention(nn.Module):
    """查询 x attend 到上下文 c（双向，无掩码）。"""

    def __init__(self, dim: int, n_heads: int) -> None:
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, q_x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        b, nq, d = q_x.shape
        nc = ctx.shape[1]
        q = self.q_proj(q_x).view(b, nq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(ctx).view(b, nc, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(ctx).view(b, nc, self.n_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, nq, d)
        return self.out(out)


class PerceiverResampler(nn.Module):
    """可学习查询 + 交叉注意力 + FFN，压 patch 序列为固定长度。"""

    def __init__(self, vision_dim: int, llm_dim: int, hidden_dim: int,
                 n_heads: int, num_queries: int, depth: int = 2) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_queries, llm_dim) * 0.02)
        self.in_proj = nn.Linear(vision_dim, llm_dim)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "norm_q": nn.LayerNorm(llm_dim),
                "norm_c": nn.LayerNorm(llm_dim),
                "attn": CrossAttention(llm_dim, n_heads),
                "norm_ffn": nn.LayerNorm(llm_dim),
                "ffn": nn.Sequential(
                    nn.Linear(llm_dim, hidden_dim), nn.GELU(),
                    nn.Linear(hidden_dim, llm_dim)),
            }) for _ in range(depth)
        ])

    def forward(self, patch_feats: torch.Tensor) -> torch.Tensor:
        x = self.queries.expand(patch_feats.size(0), -1, -1)
        c = self.in_proj(patch_feats)
        for blk in self.blocks:
            q = blk["norm_q"](x)
            kv = blk["norm_c"](c)
            x = x + blk["attn"](q, kv)
            x = x + blk["ffn"](blk["norm_ffn"](x))
        return x  # [b, num_queries, llm_dim]


class MultimodalLoganLM(nn.Module):
    """LoganLM 多模态全家桶：视觉编码器 + 重采样 + 投影 + 语言模型。"""

    def __init__(self, llm_cfg: LLMConfig, vision_cfg: VisionConfig,
                 mm_cfg: MultiModalConfig) -> None:
        super().__init__()
        self.llm_cfg = llm_cfg
        self.vision = VisionEncoder(vision_cfg)
        self.llm = LoganLMTransformer(llm_cfg)
        self.resampler = PerceiverResampler(
            vision_dim=vision_cfg.dim, llm_dim=llm_cfg.dim,
            hidden_dim=mm_cfg.hidden_dim, n_heads=mm_cfg.n_heads,
            num_queries=mm_cfg.num_query_tokens)
        self.projector = nn.Sequential(
            nn.Linear(llm_cfg.dim, mm_cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(mm_cfg.hidden_dim, llm_cfg.dim),
        )
        self.mm_cfg = mm_cfg

    # ------------------------------------------------------------------
    def encode_image(self, pixels: torch.Tensor) -> torch.Tensor:
        """像素 [b,3,H,W] → 图像软 token embedding [b, num_queries, llm_dim]"""
        patch_feats, _ = self.vision.encode(pixels)
        return self.projector(self.resampler(patch_feats))

    # ------------------------------------------------------------------
    def prepare_embeds(self, text_ids: torch.Tensor,
                       image_embeds: Optional[torch.Tensor] = None,
                       image_token_id: Optional[int] = None
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
        """把图像软 token 按 <|img|> 占位符插入文本 embedding。

        约定：text_ids 中每个 image_token_id 的位置 = 一张图的 n_q 个软 token。
        返回 (inputs_embeds, labels)；图像位置 labels=-100（不参与 LM loss）。
        """
        b, s = text_ids.shape
        tok_emb = self.llm.tok_emb(text_ids)
        if image_embeds is None:
            return tok_emb, text_ids.clone()

        n_q = self.mm_cfg.num_query_tokens
        device, dtype = tok_emb.device, tok_emb.dtype
        embeds, labels = [], []
        for i in range(b):
            pieces, lab = [], []
            img_idx = 0  # 第几张图
            for j in range(s):
                tid = int(text_ids[i, j])
                if image_token_id is not None and tid == image_token_id and img_idx < image_embeds.size(0):
                    chunk = image_embeds[img_idx].to(dtype)  # 第 img_idx 张图的 [n_q, dim]
                    pieces.append(chunk)
                    lab.append(torch.full((n_q,), -100, device=device, dtype=torch.long))
                    img_idx += 1
                else:
                    pieces.append(tok_emb[i, j:j + 1])
                    lab.append(text_ids[i, j:j + 1])
            embeds.append(torch.cat(pieces, dim=0))
            labels.append(torch.cat(lab, dim=0))
        return torch.stack(embeds), torch.stack(labels)

    # ------------------------------------------------------------------
    def forward(self, text_ids: torch.Tensor,
                image_pixels: Optional[torch.Tensor] = None,
                image_token_id: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """训练前向。返回 (logits, loss)。"""
        img_emb = self.encode_image(image_pixels) if image_pixels is not None else None
        inputs_embeds, labels = self.prepare_embeds(text_ids, img_emb, image_token_id)
        logits, loss = self.llm(inputs_embeds=inputs_embeds, targets=labels)
        return logits, loss

    def count_params(self) -> dict:
        return {
            "vision": self.vision.count_params(),
            "llm": self.llm.count_params(),
            "bridge": sum(p.numel() for n, p in self.named_parameters()
                          if "resampler" in n or "projector" in n),
        }
