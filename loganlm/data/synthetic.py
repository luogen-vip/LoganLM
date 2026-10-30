# -*- coding: utf-8 -*-
"""
LoganLM · 合成视觉数据集（程序自动生成，无需下载任何外部数据）
==============================================================

视觉编码器预训练数据：在随机背景上随机位置画一个 形状+颜色 的主体，
同时生成三种类别标签（颜色 / 形状 / 位置九宫格）。整个数据集由代码
即时合成，仓库里不用携带任何图片文件，训练时可无限生成。

作者：罗根（Logan）
"""
from __future__ import annotations

import math
import random
from typing import List, Tuple

import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# 类别定义
# ---------------------------------------------------------------------------
COLORS: List[Tuple[str, str, Tuple[int, int, int]]] = [
    # (中文名, 英文名, RGB)
    ("红色", "red", (220, 40, 40)),
    ("橙色", "orange", (245, 140, 30)),
    ("黄色", "yellow", (240, 220, 40)),
    ("绿色", "green", (40, 180, 70)),
    ("蓝色", "blue", (40, 90, 230)),
    ("紫色", "purple", (150, 60, 220)),
    ("黑色", "black", (20, 20, 20)),
    ("白色", "white", (245, 245, 245)),
]

SHAPES: List[str] = ["圆形", "方形", "三角形", "星形", "爱心", "十字"]

POS_NAMES: List[str] = [
    "左上", "上中", "右上",
    "左中", "正中", "右中",
    "左下", "下中", "右下",
]

_BACKGROUND_PALETTES: List[Tuple[int, int, int]] = [
    (250, 250, 250), (235, 240, 248), (248, 240, 235),
    (230, 245, 235), (245, 235, 245), (240, 240, 240),
]


def draw_synthetic(rng: random.Random, image_size: int = 128) -> Tuple[Image.Image, int, int, int]:
    """画一张合成图，返回 (PIL图, 颜色idx, 形状idx, 位置idx)。"""
    bg = rng.choice(_BACKGROUND_PALETTES)
    img = Image.new("RGB", (image_size, image_size), bg)
    draw = ImageDraw.Draw(img)

    color_idx = rng.randrange(len(COLORS))
    shape_idx = rng.randrange(len(SHAPES))
    color = COLORS[color_idx][2]

    # 主体中心（留边距，保证形状完整）
    margin = image_size // 5
    cx = rng.randint(margin, image_size - margin)
    cy = rng.randint(margin, image_size - margin)
    r = rng.randint(image_size // 8, image_size // 5)

    # 抗锯齿：先在大图上画，再缩小
    scale = 4
    big = Image.new("RGB", (image_size * scale, image_size * scale), bg)
    bd = ImageDraw.Draw(big)
    R = r * scale
    CX, CY = cx * scale, cy * scale
    shape = SHAPES[shape_idx]

    if shape == "圆形":
        bd.ellipse([CX - R, CY - R, CX + R, CY + R], fill=color)
    elif shape == "方形":
        bd.rectangle([CX - R, CY - R, CX + R, CY + R], fill=color)
    elif shape == "三角形":
        bd.polygon([(CX, CY - R), (CX - R, CY + R * 0.9), (CX + R, CY + R * 0.9)], fill=color)
    elif shape == "星形":
        pts = []
        for i in range(10):
            ang = -math.pi / 2 + i * math.pi / 5
            rad = R if i % 2 == 0 else R * 0.45
            pts.append((CX + rad * math.cos(ang), CY + rad * math.sin(ang)))
        bd.polygon(pts, fill=color)
    elif shape == "爱心":
        k = R / 16.0
        bd.polygon([
            (CX, CY + 14 * k), (CX - 15 * k, CY - 2 * k), (CX - 8 * k, CY - 12 * k),
            (CX, CY - 4 * k), (CX + 8 * k, CY - 12 * k), (CX + 15 * k, CY - 2 * k),
        ], fill=color)
    else:  # 十字
        t = R * 0.42
        bd.rectangle([CX - t, CY - R, CX + t, CY + R], fill=color)
        bd.rectangle([CX - R, CY - t, CX + R, CY + t], fill=color)

    img = big.resize((image_size, image_size), Image.LANCZOS)

    # 位置标签：主体中心映射到 3x3 九宫格
    col = min(2, int(cx / (image_size / 3)))
    row = min(2, int(cy / (image_size / 3)))
    pos_idx = row * 3 + col
    return img, color_idx, shape_idx, pos_idx


class SyntheticVisionDataset(Dataset):
    """即时生成的无限合成视觉数据集。

    size 固定时预生成一遍（评测可复现）；size=None 时每次 __getitem__
    现场随机生成（训练可视为无限数据）。
    """

    def __init__(self, length: int = 4096, image_size: int = 128, seed: int = 0,
                 regenerate_each_epoch: bool = True) -> None:
        super().__init__()
        self.length = length
        self.image_size = image_size
        self.seed = seed
        self.regenerate = regenerate_each_epoch
        self._epoch = 0
        self._cache: List = []

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        return self.length

    def _make(self, idx: int) -> Tuple[torch.Tensor, int, int, int]:
        rng = random.Random(hash((self.seed, self._epoch, idx)) & 0xFFFFFFFF)
        img, c, s, p = draw_synthetic(rng, self.image_size)
        import numpy as np
        arr = torch.from_numpy(np.array(img)).float() / 255.0
        pixels = arr.permute(2, 0, 1)  # [3, H, W]
        return pixels, c, s, p

    def __getitem__(self, idx: int):
        return self._make(idx)


def label_meanings(color_idx: int, shape_idx: int, pos_idx: int) -> str:
    """三个类别 idx -> 中文描述句子。"""
    return (f"我看到了一个{COLORS[color_idx][0]}的{SHAPES[shape_idx]}，"
            f"位于画面{POS_NAMES[pos_idx]}。")


if __name__ == "__main__":
    ds = SyntheticVisionDataset(length=8, image_size=128, regenerate_each_epoch=False)
    px, c, s, p = ds[0]
    print("像素张量:", px.shape, "| 标签:", COLORS[c][0], SHAPES[s], POS_NAMES[p])
    img, c, s, p = draw_synthetic(random.Random(1), 128)
    img.save("/tmp/loganlm_demo_image.png")
    print("示例图已保存 /tmp/loganlm_demo_image.png ->", label_meanings(c, s, p))
