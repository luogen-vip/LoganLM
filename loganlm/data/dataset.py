# -*- coding: utf-8 -*-
"""
LoganLM 数据管线 —— 把对话语料变成模型能吃的张量
=================================================

作者：罗根（Logan）

对话模板（与 chat.py 推理时严格一致）：
    <bos><sys>系统提示<user>用户说<assistant>小罗答<eos><user>...<assistant>...

训练技巧：只在 <assistant> 之后的"回答部分"计算 loss，
这样模型专心学"怎么回答"，而不是学"怎么提问"。
"""
from __future__ import annotations

import random
from typing import List, Tuple

import torch

from ..tokenizer import BPETokenizer, ASSISTANT, PAD, EOS
from .corpus import SYSTEM_PROMPT, EMOTION_PROMPTS


def render_example(system: str, user: str, assistant: str) -> str:
    """构造一条训练文本。"""
    return f"<bos><sys>{system}<user>{user}<assistant>{assistant}<eos>"


def render_prompt(history: List[Tuple[str, str]], system: str = SYSTEM_PROMPT) -> str:
    """构造推理时的提示词：系统提示 + 历史对话。"""
    s = f"<bos><sys>{system}"
    for u, a in history:
        s += f"<user>{u}<assistant>{a}<eos>"
    return s


class ChatDataset:
    """对话训练集：每条对话 = 一个变长样本，训练时动态填充到批内最长。

    system_prompts 传列表时，同一对话会以每种系统提示各生成一条样本
    （情绪模式数据增强：切情绪不降智）。

    image_pairs: 看图说话样本列表，每项 {"text": 用户问（含<img></img>标记），
    "answer": 描述，"img": (3,H,W) 张量} —— 用于多模态混合训练。
    """

    def __init__(self, tokenizer: BPETokenizer, dialog_pairs: List[Tuple[str, str]],
                 system_prompts=SYSTEM_PROMPT, seq_len: int = 256,
                 image_pairs=None) -> None:
        if isinstance(system_prompts, str):
            system_prompts = [system_prompts]
        self.seq_len = seq_len
        self.text_examples: List[Tuple[List[int], List[int]]] = []
        self.image_examples: List[Tuple[List[int], List[int], "torch.Tensor"]] = []
        skipped = 0
        for system in system_prompts:
            for user, assistant in dialog_pairs:
                text = render_example(system, user, assistant)
                ids = tokenizer.encode(text)
                if len(ids) > seq_len:  # 太长就截断（demo 语料一般不会）
                    ids = ids[:seq_len]
                if len(ids) < 8:
                    skipped += 1
                    continue
                self.text_examples.append(self._make(ids))
        if image_pairs:
            system = system_prompts[0]
            for p in image_pairs:
                text = render_example(system, p["text"], p["answer"])
                ids = tokenizer.encode(text)
                if len(ids) > seq_len:
                    ids = ids[:seq_len]
                if len(ids) < 8:
                    skipped += 1
                    continue
                ids, labels = self._make(ids)
                self.image_examples.append((ids, labels, p["img"]))
        if skipped:
            print(f"  数据集：跳过 {skipped} 条异常样本")

    @staticmethod
    def _make(ids: List[int]) -> Tuple[List[int], List[int]]:
        """input_ids + labels。

        关键：语言模型学的是"下一个词"——位置 t 的标签 = 原序列 t+1 处的词。
        <assistant>（含）之前的词不算 loss，让模型专心学"怎么回答"。
        """
        orig = list(ids)
        if ASSISTANT in ids:
            pos = len(ids) - 1 - ids[::-1].index(ASSISTANT)
            orig[:pos + 1] = [-100] * (pos + 1)
        labels = orig[1:] + [-100]  # 右移一位：t 处预测 t+1
        return ids, labels

    def __len__(self) -> int:
        return len(self.text_examples) + len(self.image_examples)

    def sample_batch(self, batch_size: int, image: bool = False):
        """随机抽一个 batch，动态 padding 到批内最长（省算力）。

        image=True 时从看图样本池采样，返回 (x, y, imgs)；
        否则返回 (x, y, None)。imgs: (B,3,H,W) 或 None。
        """
        use_img = image and len(self.image_examples) > 0
        pool = self.image_examples if use_img else self.text_examples
        n = len(pool)
        idxs = random.sample(range(n), min(batch_size, n))
        batch = [pool[i] for i in idxs]
        L = max(len(ids) for ids, _ in batch)
        x = torch.full((len(batch), L), PAD, dtype=torch.long)
        y = torch.full((len(batch), L), -100, dtype=torch.long)
        for b, item in enumerate(batch):
            ids, labels = item[0], item[1]
            m = len(ids)
            x[b, :m] = torch.tensor(ids, dtype=torch.long)
            y[b, :m] = torch.tensor(labels, dtype=torch.long)
        if use_img:
            imgs = torch.stack([item[2] for item in batch])
            return x, y, imgs
        return x, y, None


if __name__ == "__main__":
    from ..tokenizer import BPETokenizer
    from .corpus import DIALOGS

    tok = BPETokenizer().train("\n".join(u + a for u, a in DIALOGS), vocab_size=1024)
    ds = ChatDataset(tok, DIALOGS, seq_len=256)
    print("样本数:", len(ds), "| 分词器词表:", tok.vocab_size)
    x, y, _ = ds.sample_batch(4)
    print("batch 形状:", tuple(x.shape),
          "| 可训练位置:", int((y != -100).sum()), "/", int(y.numel()))
    print("数据管线自测通过 ✅")
