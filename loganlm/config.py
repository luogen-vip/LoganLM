# -*- coding: utf-8 -*-
"""
LoganLM（罗根大模型）· 全局配置
================================

本文件是整个项目的"中心契约"：所有模块（分词器、语言模型、视觉模型、
多模态融合、训练器、聊天系统）都从这里读取配置，保证彼此对齐。

作者：罗根（Logan）
开源协议：MIT
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# 1. 语言模型（LLM）配置 —— LLaMA 风格 decoder-only Transformer
# ---------------------------------------------------------------------------
@dataclass
class LLMConfig:
    """从0手写的 Transformer 语言模型超参数。

    vocab_size   : 词表大小（= 256 字节 + 特殊 token + BPE 合并数）
    dim          : 隐藏层维度 d_model
    n_layers     : Transformer 层数
    n_heads      : 注意力头数（query 头）
    n_kv_heads   : KV 头数（< n_heads 时启用 GQA 分组查询注意力）
    ffn_hidden   : SwiGLU 前馈层隐藏维度（建议 ≈ 8/3 * dim 再取整到倍数）
    max_seq_len  : 最大序列长度（RoPE 预计算的长度上限）
    rope_theta   : RoPE 基频率
    dropout      : 残差 dropout（demo 模型默认 0，避免小数据下欠拟合）
    tie_weights  : 输入 embedding 与输出 lm_head 共享权重
    """

    vocab_size: int = 4096
    dim: int = 256
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int = 8
    ffn_hidden: int = 768
    max_seq_len: int = 512
    rope_theta: float = 10000.0
    dropout: float = 0.0
    tie_weights: bool = True

    def __post_init__(self) -> None:
        assert self.dim % self.n_heads == 0, "dim 必须能被 n_heads 整除"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads 必须能被 n_kv_heads 整除"

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def n_rep(self) -> int:
        """每个 KV 头被多少个 Q 头共享（GQA）。"""
        return self.n_heads // self.n_kv_heads


# ---------------------------------------------------------------------------
# 2. 视觉编码器（ViT）配置 —— 从0手写的 Vision Transformer
# ---------------------------------------------------------------------------
@dataclass
class VisionConfig:
    """视觉编码器超参数。

    image_size    : 输入图片边长（正方形）
    patch_size    : 每个 patch 的边长
    in_chans      : 输入通道数（RGB=3）
    dim           : 编码器隐藏维度
    depth         : 编码器层数
    n_heads       : 注意力头数
    mlp_ratio     : FFN 隐藏维度 = dim * mlp_ratio
    num_colors    : 颜色分类头类别数
    num_shapes    : 形状分类头类别数
    num_positions : 位置分类头类别数（3x3 网格 = 9）
    """

    image_size: int = 128
    patch_size: int = 16
    in_chans: int = 3
    dim: int = 192
    depth: int = 6
    n_heads: int = 6
    mlp_ratio: float = 2.0
    dropout: float = 0.0
    num_colors: int = 8
    num_shapes: int = 6
    num_positions: int = 9

    def __post_init__(self) -> None:
        assert self.image_size % self.patch_size == 0, "image_size 必须能被 patch_size 整除"
        assert self.dim % self.n_heads == 0, "dim 必须能被 n_heads 整除"

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads


# ---------------------------------------------------------------------------
# 3. 多模态融合配置 —— 把视觉特征"翻译"进语言模型 embedding 空间
# ---------------------------------------------------------------------------
@dataclass
class MultiModalConfig:
    """LLaVA 式视觉-语言桥接。

    视觉编码器输出的 patch 特征先经过一个"感知器重采样器"（Perceiver
    Resampler，含 num_query_tokens 个可学习查询 token），压缩成固定数量的
    图像软 token，再经 MLP 投影到 LLM 的 embedding 空间，直接拼在文本
    token 前面喂给语言模型。
    """

    vision_dim: int = 192
    llm_dim: int = 256
    hidden_dim: int = 384
    num_query_tokens: int = 8
    n_heads: int = 4


# ---------------------------------------------------------------------------
# 4. 训练配置
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    lr: float = 3e-3
    batch_size: int = 8
    steps: int = 3000
    warmup_steps: int = 100
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    seed: int = 42
    log_every: int = 25
    out_dir: str = "checkpoints"
    device: str = "cpu"

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# 5. 聊天系统配置
# ---------------------------------------------------------------------------
@dataclass
class ChatConfig:
    """聊天编排器配置。

    persona_name      : 助手名字
    creator           : 作者（罗根）
    memory_dir        : 上下文记忆存储目录（长期保存，直到用户手动清空）
    max_history_turns : 每次喂给 LLM 的最近对话轮数
    default_emotion   : 默认情绪风格
    enable_web        : 是否允许联网搜索
    search_topk       : 联网搜索取前几个网页摘要
    """

    persona_name: str = "小罗"
    creator: str = "罗根"
    memory_dir: str = "~/.loganlm"
    max_history_turns: int = 12
    default_emotion: str = "自然"
    enable_web: bool = True
    search_topk: int = 4
    temperature: float = 0.9
    top_p: float = 0.92
    top_k: int = 40
    repetition_penalty: float = 1.25
    max_new_tokens: int = 160


# ---------------------------------------------------------------------------
# 6. 顶层聚合配置
# ---------------------------------------------------------------------------
@dataclass
class LoganLMConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    mm: MultiModalConfig = field(default_factory=MultiModalConfig)
    chat: ChatConfig = field(default_factory=ChatConfig)

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "llm": asdict(self.llm),
            "vision": asdict(self.vision),
            "mm": asdict(self.mm),
            "chat": asdict(self.chat),
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "LoganLMConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return cls(
            llm=LLMConfig(**raw.get("llm", {})),
            vision=VisionConfig(**raw.get("vision", {})),
            mm=MultiModalConfig(**raw.get("mm", {})),
            chat=ChatConfig(**raw.get("chat", {})),
        )


# ---------------------------------------------------------------------------
# 7. 快捷构造器
# ---------------------------------------------------------------------------
def tiny_config() -> LoganLMConfig:
    """超小配置：单元测试 / 快速冒烟（CPU 数秒可训）。"""
    return LoganLMConfig(
        llm=LLMConfig(vocab_size=512, dim=64, n_layers=2, n_heads=4,
                      n_kv_heads=2, ffn_hidden=176, max_seq_len=128),
        vision=VisionConfig(image_size=64, patch_size=16, dim=96,
                            depth=2, n_heads=3, mlp_ratio=2.0),
        mm=MultiModalConfig(vision_dim=96, llm_dim=64, hidden_dim=128,
                            num_query_tokens=4),
    )


def demo_config() -> LoganLMConfig:
    """演示配置：本仓库 checkpoints 里提交的 demo 权重就是用它训练的。"""
    return LoganLMConfig(
        llm=LLMConfig(vocab_size=4096, dim=256, n_layers=6, n_heads=8,
                      n_kv_heads=8, ffn_hidden=768, max_seq_len=512),
        vision=VisionConfig(image_size=128, patch_size=16, dim=192,
                            depth=6, n_heads=6),
        mm=MultiModalConfig(vision_dim=192, llm_dim=256, hidden_dim=384,
                            num_query_tokens=8),
    )


if __name__ == "__main__":
    cfg = demo_config()
    print(json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2))
