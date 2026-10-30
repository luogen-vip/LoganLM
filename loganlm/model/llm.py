# -*- coding: utf-8 -*-
"""
LoganLM 语言模型 —— 从 0 手写的 LLaMA 风格 Transformer
======================================================

作者：罗根（Logan）

全部组件手写实现，不使用任何现成模型权重或第三方模型库：

    RMSNorm     均方根归一化（比 LayerNorm 更稳更快）
    RoPE        旋转位置编码（把相对位置信息旋进 Q/K 里）
    GQA         分组查询注意力（少数 KV 头被多数 Q 头共享，省显存提速）
    SwiGLU      门控前馈网络（SiLU(w_gate·x) * w_up·x）
    权重共享    输入 embedding 与输出 lm_head 共用一张表

对话模板（训练与推理严格一致）：
    <bos><sys>系统提示<user>用户说<assistant>模型答<eos><user>...<assistant>...
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import LLMConfig
from ..generate.sampler import sample_next


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """均方根归一化：x * weight / sqrt(mean(x^2) + eps)"""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


# ---------------------------------------------------------------------------
# RoPE 旋转位置编码
# ---------------------------------------------------------------------------
def precompute_rope(head_dim: int, max_seq_len: int, theta: float = 10000.0):
    """预计算每个位置、每个频率维度的旋转角 cos/sin。"""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    angles = torch.outer(t, freqs)  # (max_seq_len, head_dim/2)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """把旋转位置编码作用到 Q 或 K 上。

    x: (B, H, T, head_dim)，两两一组做平面旋转。
    """
    x1, x2 = x[..., ::2], x[..., 1::2]
    T = x.shape[-2]
    c = cos[:T].unsqueeze(0).unsqueeze(0)  # (1,1,T,D/2)
    s = sin[:T].unsqueeze(0).unsqueeze(0)
    out = torch.empty_like(x)
    out[..., ::2] = x1 * c - x2 * s
    out[..., 1::2] = x1 * s + x2 * c
    return out


# ---------------------------------------------------------------------------
# 注意力（支持 GQA）
# ---------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, cfg: LLMConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.n_rep = cfg.n_rep
        self.head_dim = cfg.head_dim
        self.wq = nn.Linear(cfg.dim, cfg.n_heads * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.dim, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                mask: Optional[torch.Tensor]) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        if self.n_rep > 1:  # GQA：把少量 KV 头广播给所有 Q 头
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # 因果遮罩：看不到未来
        attn = F.softmax(scores.float(), dim=-1).to(v.dtype)
        attn = self.dropout(attn)
        out = attn @ v  # (B, H, T, head_dim)
        return self.wo(out.transpose(1, 2).reshape(B, T, -1))


# ---------------------------------------------------------------------------
# SwiGLU 前馈网络
# ---------------------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, cfg: LLMConfig) -> None:
        super().__init__()
        self.w_gate = nn.Linear(cfg.dim, cfg.ffn_hidden, bias=False)
        self.w_up = nn.Linear(cfg.dim, cfg.ffn_hidden, bias=False)
        self.w_down = nn.Linear(cfg.ffn_hidden, cfg.dim, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


# ---------------------------------------------------------------------------
# Transformer 块（Pre-Norm 残差结构）
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, cfg: LLMConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim)
        self.ffn = FeedForward(cfg)

    def forward(self, x, cos, sin, mask) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ---------------------------------------------------------------------------
# 完整模型
# ---------------------------------------------------------------------------
class LoganLM(nn.Module):
    """LoganLM 语言模型本体。"""

    def __init__(self, cfg: LLMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList(TransformerBlock(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight  # 权重共享

        cos, sin = precompute_rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """idx: (B, T) 的 token id；targets: (B, T)，pad 处填 -100 表示不算 loss。"""
        B, T = idx.shape
        T = min(T, self.cfg.max_seq_len)
        idx = idx[:, -T:]

        x = self.tok_emb(idx)
        mask = None
        if T > 1:
            mask = torch.triu(torch.full((T, T), float("-inf"), device=idx.device), diagonal=1)

        for layer in self.layers:
            x = layer(x, self.rope_cos[:T], self.rope_sin[:T], mask)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            targets = targets[:, -T:]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    # ------------------------------------------------------------------
    # 自回归生成
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 64,
                 temperature: float = 0.85, top_k: int = 40, top_p: float = 0.92,
                 repetition_penalty: float = 1.15,
                 eos_id: Optional[int] = None) -> torch.Tensor:
        """从 idx 开始自回归生成，遇到 eos_id 或达到上限就停。"""
        out = idx
        recent: List[int] = []
        for _ in range(max_new_tokens):
            logits, _ = self(out[:, -self.cfg.max_seq_len:])
            nxt = sample_next(
                logits[0, -1],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                recent=recent,
            )
            if eos_id is not None and nxt == eos_id:
                break
            recent.append(nxt)
            out = torch.cat([out, torch.tensor([[nxt]], dtype=out.dtype)], dim=1)
        return out


if __name__ == "__main__":
    cfg = LLMConfig(vocab_size=512, dim=64, n_layers=2, n_heads=4, n_kv_heads=2,
                    ffn_hidden=176, max_seq_len=64)
    m = LoganLM(cfg)
    x = torch.randint(0, 512, (2, 32))
    y = torch.randint(0, 512, (2, 32))
    logits, loss = m(x, y)
    print("参数量:", m.num_params(), "| logits:", tuple(logits.shape), "| loss:", float(loss))
    out = m.generate(x[:1, :8], max_new_tokens=8)
    print("生成形状:", tuple(out.shape))
    print("模型自测通过 ✅")
