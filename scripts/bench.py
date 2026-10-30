#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测一测这台机器训练一个 step 要多久（决定训练步数用）。"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from loganlm.config import LLMConfig
from loganlm.model.llm import LoganLM

torch.set_num_threads(os.cpu_count() or 1)
cfg = LLMConfig(vocab_size=4096, dim=192, n_layers=4, n_heads=6, n_kv_heads=2,
                ffn_hidden=512, max_seq_len=256)
m = LoganLM(cfg)
opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
print(f"参数量 {m.num_params() / 1e6:.2f}M | torch 线程 {torch.get_num_threads()}")
for bs in (4, 8):
    x = torch.randint(0, 4096, (bs, 256))
    y = torch.randint(0, 4096, (bs, 256))
    ts = []
    for _ in range(3):
        t0 = time.time()
        _, loss = m(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        ts.append(time.time() - t0)
    print(f"batch={bs} seq=256（最坏情况）: {min(ts):.2f} 秒/step")
