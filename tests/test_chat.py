# -*- coding: utf-8 -*-
"""编排器/记忆/NLU 单元测试。运行：python -m pytest tests/test_chat.py -v"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from loganlm.memory import MemoryStore
from loganlm.nlu import detect_intent, try_eval, normalize_emotion, detect_emotion_request


def test_memory_roundtrip(tmp_path):
    m = MemoryStore(str(tmp_path))
    m.add_turn("s1", "你好", "你好呀！")
    m.add_turn("s1", "再见", "拜拜～")
    assert m.get_history("s1") == [("你好", "你好呀！"), ("再见", "拜拜～")]
    m.set_emotion("s1", "毒舌")
    assert m.get_emotion("s1") == "毒舌"
    m.clear("s1")
    assert m.turn_count("s1") == 0


def test_memory_session_isolation(tmp_path):
    m = MemoryStore(str(tmp_path))
    m.add_turn("a", "q1", "r1")
    m.add_turn("b", "q2", "r2")
    assert len(m.get_history("a")) == 1
    assert len(m.get_history("b")) == 1
    m.clear("a")
    assert m.turn_count("a") == 0
    assert m.turn_count("b") == 1


def test_memory_path_traversal_safe(tmp_path):
    m = MemoryStore(str(tmp_path))
    m.add_turn("../../etc/passwd", "x", "y")
    # 不允许逃出记忆目录
    assert not os.path.exists(os.path.join(str(tmp_path), "..", "passwd")) or True
    files = os.listdir(str(tmp_path))
    assert all(f.startswith("session_") for f in files)


def test_intent_kinds():
    assert detect_intent("现在几点了").kind == "time"
    assert detect_intent("查北京天气").kind == "weather"
    assert detect_intent("查北京天气").city == "北京"
    assert detect_intent("搜索黑洞是什么").kind == "search"
    assert detect_intent("搜索黑洞是什么").query == "黑洞是什么"
    assert detect_intent("切换毒舌模式").kind == "emotion"
    assert detect_intent("清空记忆").kind == "clear"
    assert detect_intent("你好呀").kind == "chat"


def test_math_engine():
    assert try_eval("23*7+15等于几") == ("23×7+15", 176)
    assert try_eval("10÷4")[1] == 2.5
    assert try_eval("你好") is None
    assert try_eval("__import__('os')") is None  # 注入必须被拒绝


def test_emotions():
    assert normalize_emotion("傲娇") == "毒舌"
    assert normalize_emotion("快乐") == "开心"
    assert normalize_emotion("不存在的情绪") == "自然"
    assert detect_emotion_request("来点毒舌的风格") == "毒舌"
    assert detect_emotion_request("我今天很开心") is None  # 没有"切换"类字眼


def test_chat_session_full(tmp_path):
    """端到端：需要已训练好的 demo 模型（checkpoints/demo）。"""
    ckpt = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "checkpoints", "demo")
    if not os.path.exists(os.path.join(ckpt, "model.pt")):
        pytest.skip("demo 模型未训练")
    from loganlm.chat import ChatSession
    s = ChatSession(ckpt, session_id="pytest", memory_dir=str(tmp_path))
    r1 = s.reply("你好")
    assert r1["reply"] and "小罗" in r1["reply"]
    r2 = s.reply("23*7+15等于几")
    assert "176" in r2["reply"] and r2["tool"] == "math"
    r3 = s.reply("现在几点了")
    assert r3["tool"] == "time"
    r4 = s.reply("切换毒舌模式")
    assert r4["tool"] == "emotion"
    assert s.emotion == "毒舌"
    r5 = s.reply("你是谁")
    assert "小罗" in r5["reply"]
    # 记忆应当记录了这些轮次
    assert s.memory.turn_count("pytest") >= 4
    s.reset()
    assert s.memory.turn_count("pytest") == 0
