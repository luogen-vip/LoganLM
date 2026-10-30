# -*- coding: utf-8 -*-
"""
LoganLM · Transformer 核心算子（从 0 手写）
==========================================

包含现代开源大模型（LLaMA / Qwen / Mistral 一系）普遍采用的核心组件，
但每一行都是重新实现的：

  * RMSNorm      —— 均方根层归一化（替代 LayerNorm，省去均值中心化）
  * RoPE         —— 旋转位置编码（把"位置"编码进复数平面的旋转里，
                    天然具备相对位置外推能力）
  * CausalSelfAttention —— 因果多头自注意力，支持 GQA（分组查询注意力）
                    与 KV Cache（增量解码，聊天时越聊越快不重算旧 token）
  * SwiGLU       —— 门控前馈层（silu(xW1)⊙xW3 再投影回 d_model）
  * TransformerBlock —— Pre-Norm 残差块

作者：罗根（Logan）
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """y = x / sqrt(mean(x^2) + eps) * weight"""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm * self.weight.float()).to(dtype)


# ---------------------------------------------------------------------------
# RoPE 旋转位置编码
# ---------------------------------------------------------------------------
def precompute_rope(head_dim: int, max_seq_len: int, theta: float = 10000.0):
    """预计算 cos/sin 表，形状 [max_seq_len, head_dim//2]。"""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    pos = torch.arange(max_seq_len).float()
    angles = torch.outer(pos, inv_freq)               # [seq, head_dim//2]
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """对 [b, n_head, seq, head_dim] 的 q/k 施加旋转。

    采用 rotate_half 风格：把 head_dim 拆成前后两半 (x1, x2)，
    输出 = [x1*cos - x2*sin, x1*sin + x2*cos]，等价于复数乘法旋转。
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    # cos/sin: [seq, half] -> [1, 1, seq, half]
    if cos.dim() == 2:
        cos = cos[None, None]
        sin = sin[None, None]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


# ---------------------------------------------------------------------------
# 因果多头自注意力（GQA + KV Cache）
# ---------------------------------------------------------------------------
def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """[b, n_kv, s, d] -> [b, n_kv*n_rep, s, d]（GQA 下让 Q 头共享 KV 头）。"""
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int,
                 max_seq_len: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.n_rep = n_heads // n_kv_heads
        self.dropout = dropout

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        # 推理时的 KV Cache（eval 模式增量解码用）
        self.cache_k: Optional[torch.Tensor] = None
        self.cache_v: Optional[torch.Tensor] = None
        self.max_seq_len = max_seq_len

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,                 # [b, s_q, dim]
        cos: torch.Tensor,               # RoPE cos 表
        sin: torch.Tensor,               # RoPE sin 表
        past_len: int = 0,               # KV cache 中已有的 token 数（用于因果偏移）
        attn_mask: Optional[torch.Tensor] = None,  # 额外掩码：1=可见 0=屏蔽 [b, s_k]
        use_cache: bool = False,         # 推理时把本次 k/v 写入缓存
    ) -> torch.Tensor:
        b, s_q, dim = x.shape

        q = self.wq(x).view(b, s_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, s_q, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, s_q, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE 表按当前全局位置切片（增量解码时 = past_len 偏移）
        cos = cos[past_len:past_len + s_q]
        sin = sin[past_len:past_len + s_q]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # ---- KV cache：本次 k/v 写入缓冲区，并把历史 k/v 拼到最前面 ----
        if use_cache and self.cache_k is not None:
            self.cache_k[:, :, past_len:past_len + s_q] = k
            self.cache_v[:, :, past_len:past_len + s_q] = v
            if past_len > 0:
                k = torch.cat([self.cache_k[:, :, :past_len], k], dim=2)
                v = torch.cat([self.cache_v[:, :, :past_len], v], dim=2)

        k_rep = _repeat_kv(k, self.n_rep)   # [b, n_heads, s_k, d]
        v_rep = _repeat_kv(v, self.n_rep)

        # ---- 缩放点积注意力（手写，不调用 nn.Transformer）----
        s_k = k_rep.shape[2]
        scores = torch.matmul(q, k_rep.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 因果掩码：query 位置 i（全局 = past_len + i）只能看到 key 位置 <= 全局 i
        q_pos = torch.arange(past_len, past_len + s_q, device=x.device)[:, None]
        k_pos = torch.arange(0, s_k, device=x.device)[None, :]
        causal = (k_pos <= q_pos).view(1, 1, s_q, s_k)
        scores = scores.masked_fill(~causal, float("-inf"))

        # padding / 图像块掩码（1 可见，0 屏蔽）
        if attn_mask is not None:
            m = attn_mask[:, None, None, :].bool()  # [b,1,1,s_k]
            scores = scores.masked_fill(~m, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        if self.dropout > 0.0 and self.training:
            attn = F.dropout(attn, p=self.dropout)

        out = torch.matmul(attn, v_rep)                      # [b, h, s_q, d]
        out = out.transpose(1, 2).contiguous().view(b, s_q, -1)
        return self.wo(out)

    # ------------------------------------------------------------------
    def reset_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> None:
        self.cache_k = torch.zeros(
            batch_size, self.n_kv_heads, self.max_seq_len, self.head_dim,
            device=device, dtype=dtype)
        self.cache_v = torch.zeros_like(self.cache_k)

    def update_cache(self, k: torch.Tensor, v: torch.Tensor, start: int) -> None:
        self.cache_k[:, :, start:start + k.size(2)] = k
        self.cache_v[:, :, start:start + v.size(2)] = v


# ---------------------------------------------------------------------------
# SwiGLU 前馈层
# ---------------------------------------------------------------------------
class SwiGLU(nn.Module):
    """FFN(x) = W2( silu(W1 x) ⊙ W3 x )"""

    def __init__(self, dim: int, hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# ---------------------------------------------------------------------------
# Pre-Norm Transformer 块
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, ffn_hidden: int,
                 max_seq_len: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads, n_kv_heads, max_seq_len, dropout)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn_hidden, dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                past_len: int = 0, attn_mask: Optional[torch.Tensor] = None,
                use_cache: bool = False) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, past_len, attn_mask, use_cache)
        x = x + self.ffn(self.ffn_norm(x))
        return x


def init_linear(module: nn.Linear, n_layer: int) -> None:
    """初始化：残差投影层用 0.02/sqrt(2*n_layer) 稳定深层训练。"""
    std = 0.02 / math.sqrt(2 * max(n_layer, 1))
    nn.init.normal_(module.weight, mean=0.0, std=std)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
