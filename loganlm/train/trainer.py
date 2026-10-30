# -*- coding: utf-8 -*-
"""
LoganLM 训练器 —— 让模型"上学"的地方
=====================================

作者：罗根（Logan）

手写训练循环（不用 HF Trainer / lightning 等第三方训练框架）：
  - AdamW 优化器（解耦权重衰减）
  - 预热 + 余弦退火学习率
  - 梯度裁剪（防爆炸）
  - 动态 batch 采样
"""
from __future__ import annotations

import math
import os
import time
from typing import Optional

import torch

from ..config import TrainConfig
from ..model.llm import LoganLM


class Trainer:
    def __init__(self, model: LoganLM, cfg: Optional[TrainConfig] = None,
                 device: Optional[str] = None) -> None:
        self.cfg = cfg or TrainConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.cfg.seed)
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        self.model = model.to(self.device)
        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            betas=(0.9, 0.95),
        )

    # ------------------------------------------------------------------
    def get_lr(self, step: int) -> float:
        """先线性预热，再余弦退火到 min_lr_ratio * lr。"""
        cfg = self.cfg
        if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
            return cfg.lr * (step + 1) / cfg.warmup_steps
        p = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
        p = min(1.0, max(0.0, p))
        return cfg.lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio)
                         * 0.5 * (1.0 + math.cos(math.pi * p)))

    def train(self, dataset, log_every: Optional[int] = None) -> float:
        """主训练循环，返回最后一步的 loss。"""
        log_every = log_every or self.cfg.log_every
        self.model.train()
        t0 = time.time()
        last_loss = float("nan")
        print(f"开始训练：steps={self.cfg.steps} batch={self.cfg.batch_size} "
              f"lr={self.cfg.lr} device={self.device}")
        for step in range(self.cfg.steps):
            lr = self.get_lr(step)
            for g in self.opt.param_groups:
                g["lr"] = lr
            x, y = dataset.sample_batch(self.cfg.batch_size)
            x, y = x.to(self.device), y.to(self.device)
            _, loss = self.model(x, y)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.opt.step()
            last_loss = float(loss.item())
            if step % log_every == 0 or step == self.cfg.steps - 1:
                speed = (step + 1) / (time.time() - t0)
                print(f"  step {step + 1:>5}/{self.cfg.steps} | loss {last_loss:.4f} "
                      f"| lr {lr:.2e} | {speed:.2f} step/s", flush=True)
        print(f"训练结束：最终 loss {last_loss:.4f}，用时 {time.time() - t0:.0f} 秒")
        return last_loss

    def save(self, path: str, tokenizer=None) -> None:
        """保存 checkpoint：模型权重 + 配置。"""
        import dataclasses
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            "model": self.model.state_dict(),
            "config": dataclasses.asdict(self.model.cfg),
            "format": "LoganLM-ckpt-v1",
        }, path)
        if tokenizer is not None:
            tokenizer.save(os.path.join(os.path.dirname(path), "tokenizer.json"))
