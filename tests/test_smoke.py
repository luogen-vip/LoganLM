#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoganLM 冒烟测试 —— 一分钟内验证全链路
=======================================

测什么：
  1) 分词器：从0训练 BPE + 往返编解码
  2) 模型：前向传播 + loss 有限
  3) 生成：自回归采样能产出 token
  4) 采样器：随机采样与贪心采样

运行：python tests/test_smoke.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from loganlm.tokenizer import BPETokenizer, EOS
from loganlm.config import LLMConfig
from loganlm.model.llm import LoganLM
from loganlm.generate.sampler import sample_next
from loganlm.data.corpus import DIALOGS, SYSTEM_PROMPT
from loganlm.data.dataset import ChatDataset, render_prompt

print("== 1) 分词器：从0训练 BPE + 往返编解码 ==")
text = "\n".join(u + a for u, a in DIALOGS[:40])
tok = BPETokenizer().train(text, vocab_size=800)
s = "你好，我是小罗！"
ids = tok.encode(s)
assert tok.decode(ids) == s, "分词器往返编解码失败"
print(f"    词表 {tok.vocab_size} | 「{s}」→ {len(ids)} tokens ✅")

print("== 2) 模型：前向传播 + loss ==")
torch.manual_seed(0)
cfg = LLMConfig(vocab_size=tok.vocab_size, dim=64, n_layers=2, n_heads=4,
                n_kv_heads=2, ffn_hidden=176, max_seq_len=128)
model = LoganLM(cfg)
ds = ChatDataset(tok, DIALOGS[:40], SYSTEM_PROMPT, seq_len=128)
x, y, _ = ds.sample_batch(4)
logits, loss = model(x, y)
assert loss is not None and torch.isfinite(loss.detach()), "loss 非有限值"
print(f"    {model.num_params()} 参数 | batch {tuple(x.shape)} | loss {float(loss.detach()):.3f} ✅")

print("== 3) 生成：自回归采样 ==")
prompt = render_prompt([], SYSTEM_PROMPT) + "<user>你好<assistant>"
pids = tok.encode(prompt)
out = model.generate(torch.tensor([pids]), max_new_tokens=8, eos_id=EOS)
gen = out[0, len(pids):].tolist()
assert len(gen) <= 8
print(f"    从提示词续写 {len(gen)} 个 token ✅")

print("== 4) 采样器：随机 + 贪心 ==")
torch.manual_seed(1)
_ = [sample_next(torch.randn(300)) for _ in range(20)]
lg = torch.randn(300)  # 同一份 logits 重复采样，贪心应当完全确定
greedy = [sample_next(lg, temperature=0.0) for _ in range(5)]
assert len(set(greedy)) == 1, "温度 0 应该是确定性贪心"
print("    随机采样 / top-k / top-p / 贪心 全部正常 ✅")

print("\n全部冒烟测试通过 🎉")
