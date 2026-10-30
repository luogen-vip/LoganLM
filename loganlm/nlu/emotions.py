# -*- coding: utf-8 -*-
"""
情绪引擎 —— 让同一个大脑说出不同的"味道"
==========================================

用户需求："可以给出不一样的情绪回答。"

实现：情绪名 -> 系统提示附加段（训练时四种模式都学过，见 data/corpus.py 的
EMOTION_PROMPTS 与数据增强）。模型本体不变，"性格"靠提示切换。
"""
from __future__ import annotations

import re

AVAILABLE_EMOTIONS = ["自然", "开心", "毒舌", "温柔"]

# 口语别名 -> 标准情绪名
_ALIAS = {
    "快乐": "开心", "高兴": "开心", "元气": "开心",
    "傲娇": "毒舌", "高冷": "毒舌", "嘴臭": "毒舌",
    "治愈": "温柔", "贴贴": "温柔", "暖": "温柔",
}

_DETECT = re.compile(r"(自然|开心|快乐|高兴|元气|毒舌|傲娇|高冷|温柔|治愈)")


def normalize_emotion(name: str) -> str:
    name = _ALIAS.get(name, name)
    return name if name in AVAILABLE_EMOTIONS else "自然"


def detect_emotion_request(text: str):
    """从用户输入里识别情绪切换请求，返回标准情绪名或 None。"""
    m = _DETECT.search(text)
    if not m:
        return None
    emo = normalize_emotion(m.group(1))
    # 有明确的"切换/模式"类字眼才算切换请求，避免聊天里随便提到"开心"就切
    if re.search(r"(模式|风格|语气|切换|开启|进入|启动|来点|用|变成|换)", text):
        return emo
    return None
