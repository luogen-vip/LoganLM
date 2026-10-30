#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoganLM 视觉训练 —— 两阶段从0训练"看图说话"
=============================================

作者：罗根（Logan）

阶段一：训练 ViT 分类器（颜色/形状/位置，合成图片，多任务）
阶段二：混合训练 —— 聊天对话 + 看图说话样本联合微调
        （LLM 在已有 demo 权重上继续训，视觉特征经 Perceiver 桥接
         变成前缀软 token，模型学会"看着图说话"）

用法：
    python scripts/train_vision.py                    # 默认
    python scripts/train_vision.py --vit-steps 450 --fusion-steps 800

产物（checkpoints/demo/ 追加）：
    vit.pt / bridge.pt / vision_meta.json；model.pt 更新为多模态版
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from loganlm.config import LLMConfig, VisionConfig
from loganlm.tokenizer import BPETokenizer, PAD
from loganlm.model.llm import LoganLM
from loganlm.vision import ViT, VisionBridge
from loganlm.vision import data as vdata
from loganlm.vision.data import caption_pairs, COLORS, SHAPES, POSITIONS
from loganlm.data.corpus import DIALOGS, SYSTEM_PROMPT, EMOTION_PROMPTS
from loganlm.data.dataset import ChatDataset, render_example

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "checkpoints", "demo")


def collate(items):
    """[(ids, labels), ...] 或 [(ids, labels, img), ...] → 张量批。"""
    L = max(len(it[0]) for it in items)
    x = torch.full((len(items), L), PAD, dtype=torch.long)
    y = torch.full((len(items), L), -100, dtype=torch.long)
    for b, it in enumerate(items):
        m = len(it[0])
        x[b, :m] = torch.tensor(it[0], dtype=torch.long)
        y[b, :m] = torch.tensor(it[1], dtype=torch.long)
    if len(it) >= 3:
        return x, y, torch.stack([it[2] for it in items])
    return x, y, None


def train_vit(steps: int, batch: int, lr: float, rng: random.Random) -> ViT:
    cfg = VisionConfig(image_size=vdata.IMAGE_SIZE, patch_size=16, dim=96,
                       depth=3, n_heads=4, mlp_ratio=2.0)
    vit = ViT(cfg)
    opt = torch.optim.AdamW(vit.parameters(), lr=lr, weight_decay=0.05)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    t0 = time.time()
    for step in range(steps):
        imgs, labels = vdata.sample_batch_images(batch, rng)
        c = torch.tensor([x[0] for x in labels])
        s = torch.tensor([x[1] for x in labels])
        p = torch.tensor([x[2] for x in labels])
        lc, ls, lp, _ = vit(imgs)
        loss = (F.cross_entropy(lc, c) + F.cross_entropy(ls, s)
                + F.cross_entropy(lp, p)) / 3.0
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vit.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            accc = float((lc.argmax(1) == c).float().mean())
            accs = float((ls.argmax(1) == s).float().mean())
            accp = float((lp.argmax(1) == p).float().mean())
            print(f"  [ViT {step+1}/{steps}] loss {float(loss):.3f} "
                  f"色{accc:.2f} 形{accs:.2f} 位{accp:.2f} ({time.time()-t0:.0f}s)",
                  flush=True)
    return vit


def eval_vit(vit: ViT, rng: random.Random, n: int = 96) -> float:
    imgs, labels = vdata.sample_batch_images(n, rng)
    c = torch.tensor([x[0] for x in labels])
    s = torch.tensor([x[1] for x in labels])
    p = torch.tensor([x[2] for x in labels])
    with torch.no_grad():
        lc, ls, lp, _ = vit(imgs)
    acc = (float((lc.argmax(1) == c).float().mean())
           + float((ls.argmax(1) == s).float().mean())
           + float((lp.argmax(1) == p).float().mean())) / 3.0
    return acc


def main() -> None:
    ap = argparse.ArgumentParser(description="训练 LoganLM 视觉能力（看图说话）")
    ap.add_argument("--vit-steps", type=int, default=450)
    ap.add_argument("--fusion-steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--vit-batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1.2e-3)
    ap.add_argument("--n-images", type=int, default=480)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    rng = random.Random(42)
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(42)

    # ─── 阶段一：ViT 分类器 ────────────────────────────────────────
    print(f"[1/2] 训练 ViT（合成图 {vdata.IMAGE_SIZE}px，{len(COLORS)}色 × "
          f"{len(SHAPES)}形 × 9位置）")
    vit = train_vit(args.vit_steps, args.vit_batch, 2.5e-3, rng)
    acc = eval_vit(vit, rng)
    print(f"      验证平均准确率：{acc:.2%}")

    # ─── 阶段二：混合训练（聊天 + 看图说话）─────────────────────────
    print("[2/2] 混合训练：聊天对话 + 看图说话（桥接 + LLM 联合微调）")
    tok = BPETokenizer.load(os.path.join(args.out, "tokenizer.json"))
    with open(os.path.join(args.out, "config.json"), encoding="utf-8") as f:
        llm_cfg = LLMConfig(**json.load(f))
    model = LoganLM(llm_cfg)
    pt = os.path.join(args.out, "model.pt")
    if os.path.exists(pt):  # 在已有 demo 权重上继续训，保留聊天能力
        model.load_state_dict(torch.load(pt, map_location="cpu", weights_only=False)["model"])
        print("      已加载现有 demo 权重，在其上继续训练")

    bridge = VisionBridge(vision_dim=vit.cfg.dim, llm_dim=llm_cfg.dim,
                          hidden_dim=160, num_query_tokens=4)

    # 文本样本池（聊天，含情绪增强）
    ds = ChatDataset(tok, DIALOGS, [SYSTEM_PROMPT + e for e in EMOTION_PROMPTS.values()],
                     seq_len=llm_cfg.max_seq_len)
    text_pool = ds.text_examples

    # 图像样本池（看图说话，带分类标签）
    img_pool = []
    for p in caption_pairs(args.n_images, rng):
        text = render_example(SYSTEM_PROMPT, f"<img></img>{p['ask']}", p["answer"])
        ids = tok.encode(text)[:llm_cfg.max_seq_len]
        if len(ids) < 8:
            continue
        ids, labels = ChatDataset._make(ids)
        img_pool.append((ids, labels, p["img"], p["c"], p["s"], p["p"]))
    print(f"      样本：聊天 {len(text_pool)} 条 + 看图 {len(img_pool)} 条")

    params = list(model.parameters()) + list(bridge.parameters()) + list(vit.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05, betas=(0.9, 0.95))
    steps, warm = args.fusion_steps, max(40, args.fusion_steps // 12)
    t0 = time.time()
    last_ce = float("nan")
    for step in range(steps):
        prog = step / max(1, steps - 1)
        lr_now = args.lr * min(1.0, (step + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))
        for g in opt.param_groups:
            g["lr"] = lr_now

        want_img = rng.random() < 0.4 and len(img_pool) > 0
        if want_img:
            idxs = rng.sample(range(len(img_pool)), min(args.batch, len(img_pool)))
            items = [img_pool[i] for i in idxs]
            x, y, imgs = collate([(it[0], it[1], it[2]) for it in items])
            c = torch.tensor([it[3] for it in items])
            s = torch.tensor([it[4] for it in items])
            p = torch.tensor([it[5] for it in items])
            lc, ls, lp, feats = vit(imgs)
            prefix = bridge(feats)
            cls = (F.cross_entropy(lc, c) + F.cross_entropy(ls, s)
                   + F.cross_entropy(lp, p)) / 3.0
        else:
            idxs = rng.sample(range(len(text_pool)), min(args.batch, len(text_pool)))
            x, y, _ = collate([text_pool[i] for i in idxs])
            imgs, prefix, cls = None, None, 0.0

        logits, ce = model(x, y, prefix_embeds=prefix)
        loss = ce + (0.25 * cls if imgs is not None else 0.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        last_ce = float(ce)
        if step % 50 == 0 or step == steps - 1:
            tag = "图" if imgs is not None else "聊"
            print(f"  [融合 {step+1}/{steps}] {tag} ce {last_ce:.3f} "
                  f"lr {lr_now:.1e} ({time.time()-t0:.0f}s)", flush=True)

    # ─── 保存 ──────────────────────────────────────────────────────
    print("[保存] vit.pt / bridge.pt / vision_meta.json，并更新 model.pt")
    torch.save({"vit": vit.state_dict(),
                "config": {"image_size": vit.cfg.image_size, "patch_size": vit.cfg.patch_size,
                           "dim": vit.cfg.dim, "depth": vit.cfg.depth, "n_heads": vit.cfg.n_heads,
                           "mlp_ratio": vit.cfg.mlp_ratio}},
               os.path.join(args.out, "vit.pt"))
    torch.save({"bridge": bridge.state_dict(),
                "num_query_tokens": bridge.num_query_tokens},
               os.path.join(args.out, "bridge.pt"))
    with open(os.path.join(args.out, "vision_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"colors": COLORS, "shapes": SHAPES, "positions": POSITIONS,
                   "vit_acc": round(acc, 4), "final_caption_ce": round(last_ce, 4)},
                  f, ensure_ascii=False, indent=2)
    torch.save({"model": model.state_dict(), "config": llm_cfg.__dict__,
                "format": "LoganLM-ckpt-v1-vision"}, os.path.join(args.out, "model.pt"))

    # ─── 冒烟：给一张没见过的图，看它说什么 ─────────────────────────
    print("[冒烟] 生成一张『绿色星形·右下』测试图……")
    test_img = vdata.draw_image(2, 3, 8)
    px = vdata.to_tensor(test_img).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        lc, ls, lp, feats = vit(px)
        pre = bridge(feats)
    c, s, p = int(lc[0].argmax()), int(ls[0].argmax()), int(lp[0].argmax())
    print(f"      ViT 识别：{COLORS[c]}的{SHAPES[s]}，位置{POSITIONS[p]}（真实：绿色星形 右下）")
    ids = tok.encode(f"<bos><sys>{SYSTEM_PROMPT}<user><img></img>这张图里有什么？<assistant>")
    out = model.generate(torch.tensor([ids]), max_new_tokens=40, temperature=0.7,
                         top_k=20, top_p=0.9, repetition_penalty=1.15,
                         eos_id=258, prefix_embeds=pre)
    print("      看图说话：", tok.decode(out[0, len(ids):].tolist()).strip())
    print("完成！运行 python chat.py 试试：把图片路径发给我，或网页上传图片 🎉")


if __name__ == "__main__":
    main()
