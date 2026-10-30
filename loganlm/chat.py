# -*- coding: utf-8 -*-
"""
LoganLM 聊天编排器 —— 小罗的大脑指挥中心
==========================================

作者：罗根（Logan）

一条消息的完整旅程：
    用户输入
      → 意图识别（nlu.intent）
          ├ 时间/天气/算术  → 工具直接答（快、准）
          ├ 搜索           → 联网搜索（tools.web）→ 资料拼进提示词 → 模型总结
          ├ 切情绪/清记忆   → 直接处理
          └ 普通聊天        → 模型生成（带上历史记忆 + 当前情绪）
      → 回复写入持久记忆（memory.MemoryStore）
"""
from __future__ import annotations

import json
import os
from typing import List, Optional

import torch

from .config import ChatConfig, LLMConfig
from .tokenizer import BPETokenizer, EOS
from .model.llm import LoganLM
from .data.corpus import SYSTEM_PROMPT, EMOTION_PROMPTS
from .data.dataset import render_prompt
from .memory import MemoryStore
from .nlu import detect_intent, try_eval, normalize_emotion
from .nlu.emotions import detect_emotion_request
from .tools.web import get_now_str, get_weather, web_search

DEFAULT_CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "checkpoints", "demo")


class ChatSession:
    """一个会话 = 模型 + 分词器 + 持久记忆 + 情绪。线程安全级别：单线程假设。"""

    def __init__(self, ckpt_dir: str = DEFAULT_CKPT, session_id: str = "default",
                 memory_dir: Optional[str] = None) -> None:
        self.session_id = session_id
        self.memory = MemoryStore(memory_dir)
        self.ccfg = ChatConfig()

        tok_path = os.path.join(ckpt_dir, "tokenizer.json")
        cfg_path = os.path.join(ckpt_dir, "config.json")
        pt_path = os.path.join(ckpt_dir, "model.pt")
        if not os.path.exists(pt_path):
            raise FileNotFoundError(
                f"没找到模型 {pt_path}。请先运行: python scripts/train_demo.py")
        self.tok = BPETokenizer.load(tok_path)
        with open(cfg_path, encoding="utf-8") as f:
            cfg = LLMConfig(**json.load(f))
        self.model = LoganLM(cfg)
        sd = torch.load(pt_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(sd["model"])
        self.model.eval()
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        self.emotion = self.memory.get_emotion(session_id)

    # ------------------------------------------------------------------
    def _generate(self, user: str, extra_context: str = "") -> str:
        history = self.memory.get_history(self.session_id,
                                          limit=self.ccfg.max_history_turns)
        system = SYSTEM_PROMPT + EMOTION_PROMPTS.get(self.emotion, "")
        if extra_context:
            system += f"\n[参考资料]\n{extra_context}"
        prompt = render_prompt(history, system) + f"<user>{user}<assistant>"
        ids = self.tok.encode(prompt)
        x = torch.tensor([ids], dtype=torch.long)
        out = self.model.generate(
            x,
            max_new_tokens=self.ccfg.max_new_tokens,
            temperature=self.ccfg.temperature,
            top_k=self.ccfg.top_k,
            top_p=self.ccfg.top_p,
            repetition_penalty=self.ccfg.repetition_penalty,
            eos_id=EOS,
        )
        reply = self.tok.decode(out[0, len(ids):].tolist()).strip()
        return reply or "（我走神了，再说一遍好不好？）"

    # ------------------------------------------------------------------
    def reply(self, text: str) -> dict:
        """处理一条用户消息，返回 {reply, tool, sources}。"""
        text = (text or "").strip()
        if not text:
            return {"reply": "想说什么呀？", "tool": "chat", "sources": []}

        intent = detect_intent(text)

        # 1) 清空记忆
        if intent.kind == "clear":
            self.memory.clear(self.session_id)
            return {"reply": "好的，记忆清空完毕！我们重新认识一次吧：你好，我是小罗！",
                    "tool": "clear", "sources": []}

        # 2) 切换情绪（显式指令 或 自然语言请求）
        new_emo = intent.emotion or detect_emotion_request(text)
        if new_emo:
            self.emotion = normalize_emotion(new_emo)
            self.memory.set_emotion(self.session_id, self.emotion)
            return {"reply": f"情绪切换到「{self.emotion}」啦！说吧，想聊什么？",
                    "tool": "emotion", "sources": []}

        # 3) 时间
        if intent.kind == "time":
            reply = f"现在是 {get_now_str()}（以设备时间为准哦）。"
            self.memory.add_turn(self.session_id, text, reply)
            return {"reply": reply, "tool": "time", "sources": []}

        # 4) 天气
        if intent.kind == "weather":
            if intent.city:
                reply, tool = get_weather(intent.city), "weather"
            else:
                reply, tool = "告诉我在哪个城市，比如「查北京的天气」，我马上帮你看天！", "weather"
            self.memory.add_turn(self.session_id, text, reply)
            return {"reply": reply, "tool": tool, "sources": []}

        # 5) 算术
        if intent.kind == "math":
            r = try_eval(text)
            if r:
                pretty, val = r
                reply = f"{pretty} = {val}！这可难不倒我～"
                self.memory.add_turn(self.session_id, text, reply)
                return {"reply": reply, "tool": "math", "sources": []}

        # 6) 联网搜索：抓资料 → 让模型用资料回答
        sources = []
        if intent.kind == "search" and intent.query:
            items = web_search(intent.query, topk=self.ccfg.search_topk)
            sources = [{"title": it["title"], "url": it["url"]} for it in items]
            context = "\n".join(f"{i}. {it['title']}：{it['snippet']}"
                                for i, it in enumerate(items, 1)) or "（没搜到资料）"
            reply = self._generate(text, extra_context=context)
            self.memory.add_turn(self.session_id, text, reply)
            return {"reply": reply, "tool": "search", "sources": sources}

        # 7) 普通聊天：模型本人回答
        reply = self._generate(text)
        self.memory.add_turn(self.session_id, text, reply)
        return {"reply": reply, "tool": "model", "sources": []}

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.memory.clear(self.session_id)
        self.emotion = "自然"

    def history(self, limit: Optional[int] = None) -> List[tuple]:
        return self.memory.get_history(self.session_id, limit)


if __name__ == "__main__":
    s = ChatSession(session_id="selftest")
    for q in ["你好", "你是谁", "切换毒舌模式", "你是谁", "23*7+15等于几", "清空记忆"]:
        r = s.reply(q)
        print(f"你: {q}\n小罗({r['tool']}): {r['reply']}\n")
    s.reset()
    print("编排器自测通过 ✅")
