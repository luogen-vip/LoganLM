# -*- coding: utf-8 -*-
"""视觉能力冒烟：加载训好的 ViT+桥接+LLM，对多张合成图做识别和看图说话。"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from loganlm.config import LLMConfig, VisionConfig
from loganlm.tokenizer import BPETokenizer, EOS
from loganlm.model.llm import LoganLM
from loganlm.vision import ViT, VisionBridge
from loganlm.vision import data as vdata
from loganlm.vision.data import COLORS, SHAPES, POSITIONS, to_tensor, draw_image

D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "demo")
tok = BPETokenizer.load(os.path.join(D, "tokenizer.json"))
with open(os.path.join(D, "config.json"), encoding="utf-8") as f:
    llm_cfg = LLMConfig(**json.load(f))
model = LoganLM(llm_cfg)
model.load_state_dict(torch.load(os.path.join(D, "model.pt"), map_location="cpu", weights_only=False)["model"])
model.eval()

vcfg = torch.load(os.path.join(D, "vit.pt"), map_location="cpu", weights_only=False)
vit = ViT(VisionConfig(**vcfg["config"]))
vit.load_state_dict(vcfg["vit"])
vit.eval()

b = torch.load(os.path.join(D, "bridge.pt"), map_location="cpu", weights_only=False)
bridge = VisionBridge(vision_dim=vcfg["config"]["dim"], llm_dim=llm_cfg.dim,
                      num_query_tokens=b["num_query_tokens"])
bridge.load_state_dict(b["bridge"])
bridge.eval()

CASES = [(0, 0, 4), (1, 2, 0), (3, 4, 2), (5, 1, 8), (6, 5, 6), (2, 3, 8)]
right = 0
for (c, s, p) in CASES:
    px = to_tensor(draw_image(c, s, p)).unsqueeze(0)
    with torch.no_grad():
        lc, ls, lp, feats = vit(px)
        pre = bridge(feats)
    pc, ps, pp = int(lc[0].argmax()), int(ls[0].argmax()), int(lp[0].argmax())
    ok = (pc, ps, pp) == (c, s, p)
    right += ok
    ids = tok.encode(f"<bos><sys>{'你是小罗，由罗根从零手写打造的 AI 助手。回答要简短、自然、有温度。'}"
                     f"<user><img></img>这张图里有什么？<assistant>")
    out = model.generate(torch.tensor([ids]), max_new_tokens=36, temperature=0.65,
                         top_k=15, top_p=0.85, repetition_penalty=1.1,
                         eos_id=EOS, prefix_embeds=pre)
    say = tok.decode(out[0, len(ids):].tolist()).strip()
    print(f"真值: {COLORS[c]}{SHAPES[s]}·{POSITIONS[p]} | ViT: {COLORS[pc]}{SHAPES[ps]}·{POSITIONS[pp]}"
          f" {'✅' if ok else '❌'} | 小罗说: {say}")
print(f"\n识别全对率: {right}/{len(CASES)}")
assert right >= len(CASES) - 2, "识别准确率不达标"
print("视觉冒烟测试通过 🎉")
