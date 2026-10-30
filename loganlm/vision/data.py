# -*- coding: utf-8 -*-
"""
LoganLM 合成图像数据 —— 颜色 × 形状 × 位置的"看图说话"教材
============================================================

作者：罗根（Logan）

用 Pillow 从0画出合成图片（8 色 × 6 形状 × 9 宫格位置），
并给出对应中文描述（描述模板有多个变体，模型学到的是"看图说话"，
而不是背固定句子——因为同一张图可以有多种问法/答法）。
"""
from __future__ import annotations

import random
from typing import List, Tuple

import torch
from PIL import Image, ImageDraw

IMAGE_SIZE = 64
COLORS = ["红色", "蓝色", "绿色", "黄色", "紫色", "橙色", "黑色", "白色"]
COLOR_RGB = [
    (214, 64, 69), (64, 100, 220), (58, 178, 84), (232, 206, 62),
    (152, 74, 202), (236, 140, 52), (28, 28, 32), (242, 242, 245),
]
SHAPES = ["圆形", "三角形", "方形", "星形", "心形", "十字"]
POSITIONS = ["左上", "中上", "右上", "左中", "中间", "右中", "左下", "中下", "右下"]
NUM_COLORS, NUM_SHAPES, NUM_POSITIONS = 8, 6, 9

ASKS = ["这张图里有什么？", "看看这张图，描述一下", "图里画的是什么？",
        "帮我看下这张图", "这张图片的内容是什么？"]

_CAPS = [
    "这张图片里有一个{c}的{s}，它在{p}。",
    "图里是一个{c}的{s}，位置在{p}。",
    "我看到一个{c}{s}，它位于{p}。",
    "这是一个{c}的{s}，出现在{p}。",
]

_BG = (44, 47, 61)


def _cell_center(pos_idx: int) -> Tuple[int, int]:
    """9 宫格中心坐标。"""
    r, c = divmod(pos_idx, 3)
    cell = IMAGE_SIZE // 3
    return c * cell + cell // 2, r * cell + cell // 2


def draw_image(color_idx: int, shape_idx: int, pos_idx: int) -> Image.Image:
    """画一张合成图：深灰背景 + 纯色形状。"""
    img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), _BG)
    d = ImageDraw.Draw(img)
    cx, cy = _cell_center(pos_idx)
    r = IMAGE_SIZE // 6  # 形状半径（大一点更好认）
    color = COLOR_RGB[color_idx]
    s = shape_idx
    if s == 0:      # 圆形
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif s == 1:    # 三角形
        d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=color)
    elif s == 2:    # 方形
        d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif s == 3:    # 星形（五角星）
        import math as _m
        pts = []
        for i in range(10):
            ang = -_m.pi / 2 + i * _m.pi / 5
            rad = r if i % 2 == 0 else r * 0.45
            pts.append((cx + rad * _m.cos(ang), cy + rad * _m.sin(ang)))
        d.polygon(pts, fill=color)
    elif s == 4:    # 心形（两圆+三角近似）
        d.ellipse([cx - r, cy - r * 0.9, cx, cy + r * 0.1], fill=color)
        d.ellipse([cx, cy - r * 0.9, cx + r, cy + r * 0.1], fill=color)
        d.polygon([(cx - r, cy - r * 0.35), (cx + r, cy - r * 0.35), (cx, cy + r)], fill=color)
    else:           # 十字
        w = max(3, r // 2)
        d.rectangle([cx - w, cy - r, cx + w, cy + r], fill=color)
        d.rectangle([cx - r, cy - w, cx + r, cy + w], fill=color)
    return img


def to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL → (3,H,W) 归一化到 [-1,1]（不依赖 torchvision）。"""
    import numpy as np
    arr = np.asarray(img, dtype="float32") / 255.0          # (H,W,3)
    arr = (arr - 0.5) / 0.5                                  # [-1,1]
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3,H,W)
    return t


def make_caption(color_idx: int, shape_idx: int, pos_idx: int, rng: random.Random) -> str:
    return rng.choice(_CAPS).format(c=COLORS[color_idx], s=SHAPES[shape_idx],
                                    p=POSITIONS[pos_idx])


def sample_batch_images(batch: int, rng: random.Random):
    """随机采样一批 (图像张量, color, shape, pos)。"""
    imgs, cs = [], []
    for _ in range(batch):
        c, s, p = rng.randrange(8), rng.randrange(6), rng.randrange(9)
        imgs.append(to_tensor(draw_image(c, s, p)))
        cs.append((c, s, p))
    return torch.stack(imgs), cs


def caption_pairs(n: int, rng: random.Random) -> List[dict]:
    """生成 n 条带图训练样本：{img, ask, answer, c, s, p}。"""
    out = []
    for _ in range(n):
        c, s, p = rng.randrange(8), rng.randrange(6), rng.randrange(9)
        out.append({
            "img": to_tensor(draw_image(c, s, p)),
            "ask": rng.choice(ASKS),
            "answer": make_caption(c, s, p, rng),
            "c": c, "s": s, "p": p,
        })
    return out


if __name__ == "__main__":
    rng = random.Random(0)
    im = draw_image(0, 0, 4)  # 红色圆形·中间
    im.save("/tmp/loganlm_vit_test.png")
    t = to_tensor(im)
    print("图像张量:", tuple(t.shape), "范围", float(t.min()), "~", float(t.max()))
    pairs = caption_pairs(3, rng)
    for p in pairs:
        print(" 样本:", p["ask"], "->", p["answer"])
    print("合成数据自测通过 ✅（图已存 /tmp/loganlm_vit_test.png）")
