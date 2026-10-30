#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoganLM 演示模型训练脚本
========================

作者：罗根（Logan）

用法：
    python scripts/train_demo.py                # 默认超参训练（CPU 可跑）
    python scripts/train_demo.py --steps 1500   # 自定义训练步数
    python scripts/train_demo.py --dim 256 --layers 6   # 更大的模型

产物（ checkpoints/demo/ ）：
    model.pt        模型权重
    tokenizer.json  从0训练的 BPE 分词器
    config.json     模型配置

训练完运行 `python chat.py` 即可和小罗聊天！
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from loganlm.tokenizer import BPETokenizer
from loganlm.config import LLMConfig, TrainConfig
from loganlm.model.llm import LoganLM
from loganlm.data.corpus import DIALOGS, SYSTEM_PROMPT, EMOTION_PROMPTS
from loganlm.data.dataset import ChatDataset
from loganlm.train.trainer import Trainer


def main() -> None:
    ap = argparse.ArgumentParser(description="训练 LoganLM 演示模型（CPU 可跑）")
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--out", default="checkpoints/demo")
    args = ap.parse_args()

    # ─── 1. 从0训练分词器 ───────────────────────────────────────────
    corpus_text = "\n".join(f"用户：{u}\n小罗：{a}" for u, a in DIALOGS)
    n_bytes = len(corpus_text.encode("utf-8"))
    print(f"[1/3] 训练分词器：语料 {n_bytes} 字节 / {len(DIALOGS)} 段对话，目标词表 {args.vocab}")
    t0 = time.time()
    tok = BPETokenizer().train(corpus_text, vocab_size=args.vocab)
    sample = tok.encode("你好，我是小罗！")
    print(f"      实际词表 {tok.vocab_size} | 示例「你好，我是小罗！」编码成 {len(sample)} tokens"
          f" | 用时 {time.time() - t0:.1f}s")
    os.makedirs(args.out, exist_ok=True)
    tok.save(os.path.join(args.out, "tokenizer.json"))

    # ─── 2. 构建模型并训练 ─────────────────────────────────────────
    cfg = LLMConfig(vocab_size=tok.vocab_size, dim=args.dim, n_layers=args.layers,
                    n_heads=6, n_kv_heads=2, ffn_hidden=512, max_seq_len=args.seq_len)
    model = LoganLM(cfg)
    print(f"[2/3] 构建模型：{model.num_params() / 1e6:.2f}M 参数"
          f"（dim={cfg.dim} layers={cfg.n_layers} heads={cfg.n_heads} kv_heads={cfg.n_kv_heads}，启用 GQA）")
    # 数据增强：4 种情绪系统提示 × 全部对话（切情绪不降智）
    system_prompts = [SYSTEM_PROMPT + e for e in EMOTION_PROMPTS.values()]
    ds = ChatDataset(tok, DIALOGS, system_prompts, seq_len=cfg.max_seq_len)
    print(f"      训练样本 {len(ds)} 条")
    tcfg = TrainConfig(lr=args.lr, batch_size=args.batch, steps=args.steps,
                       warmup_steps=max(50, args.steps // 10), out_dir=args.out)
    Trainer(model, tcfg).train(ds)

    # ─── 3. 保存 ───────────────────────────────────────────────────
    print(f"[3/3] 保存到 {args.out}/ ...")
    torch.save({"model": model.state_dict(), "config": dataclasses.asdict(cfg),
                "format": "LoganLM-ckpt-v1"}, os.path.join(args.out, "model.pt"))
    with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(cfg), f, ensure_ascii=False, indent=2)
    print("完成！运行  python chat.py  即可和小罗聊天 🎉")


if __name__ == "__main__":
    main()
