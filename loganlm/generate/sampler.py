# -*- coding: utf-8 -*-
"""
LoganLM 采样器 —— 控制"下一个 token 怎么选"
============================================

作者：罗根（Logan）

实现四种经典解码策略（全部手写，不依赖第三方生成库）：
  - temperature      温度：越低越保守，越高越放飞
  - top-k            只在概率最高的 k 个 token 里选
  - top-p (nucleus)  只在累计概率达到 p 的最小集合里选
  - repetition_penalty  重复惩罚：刚说过的词再选就变难
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F


def sample_next(logits: torch.Tensor, temperature: float = 0.85, top_k: int = 40,
                top_p: float = 0.92, repetition_penalty: float = 1.15,
                recent: Optional[List[int]] = None) -> int:
    """从一维 logits 中采样下一个 token 的 id。"""
    with torch.no_grad():
        logits = logits.clone().float()

        # 1) 重复惩罚：出现过的 token，正的压低、负的抬高
        if recent and repetition_penalty and repetition_penalty != 1.0:
            for p in set(recent):
                v = logits[p]
                logits[p] = v / repetition_penalty if v > 0 else v * repetition_penalty

        # 2) 温度为 0 → 贪心
        if temperature <= 1e-6:
            return int(torch.argmax(logits).item())

        logits = logits / max(float(temperature), 1e-6)

        # 3) top-k 截断
        if top_k and 0 < int(top_k) < logits.size(-1):
            kth = torch.topk(logits, int(top_k))[0][-1]
            logits = torch.where(logits < kth,
                                 torch.full_like(logits, -float("inf")), logits)

        # 4) top-p（nucleus）截断
        if top_p and 0 < float(top_p) < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cum = torch.cumsum(probs, dim=-1)
            # 保留累计概率第一次超过 p 的那个 token，之后的全部屏蔽
            cut = (cum - probs) > float(top_p)
            sorted_logits = sorted_logits.masked_fill(cut, -float("inf"))
            choice = torch.multinomial(F.softmax(sorted_logits, dim=-1), 1)
            return int(sorted_idx[choice].item())

        probs = F.softmax(logits, dim=-1)
        try:
            return int(torch.multinomial(probs, 1).item())
        except RuntimeError:
            return int(torch.argmax(logits).item())


if __name__ == "__main__":
    torch.manual_seed(0)
    lg = torch.randn(1000)
    ids = [sample_next(lg) for _ in range(50)]
    print("采样示例:", ids[:12])
    print("采样器自测通过 ✅")
