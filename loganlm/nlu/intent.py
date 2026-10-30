# -*- coding: utf-8 -*-
"""
意图识别 —— 在调用神经网络之前，先用轻量规则判断用户想干什么
==============================================================

能确定性回答的（时间/天气/算术/搜索/切情绪/清记忆）就别麻烦模型了，
模型专注做它最擅长的：像人一样聊天。

作者：罗根（Logan）
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class Intent:
    kind: str                      # time/weather/math/search/emotion/clear/chat
    text: str = ""                 # 原始输入
    query: str = ""                # 搜索关键词
    city: Optional[str] = None     # 天气城市
    expr: Optional[str] = None     # 算术表达式
    emotion: Optional[str] = None  # 目标情绪


CITY_BLACKLIST = {"今天", "明天", "后天", "现在", "当前", "外面", "这里",
                  "哪", "哪里", "什么", "那边", "你们", "我", "你"}

_TIME_PAT = re.compile(r"(现在几点|几点了|几点啦|什么时间|现在时间|今天几号|几月几号|星期几|今天星期|今天是星期)")
_WEATHER_PAT = re.compile(r"(天气|气温|温度|下不下雨|下雨吗|下雪|穿什么)")
_MATH_PAT = re.compile(r"\d[\d\(\)\.\s]*(?:[\+\-\*\/×÷^][\d\(\)\.\s*]+)+")
_SEARCH_TRIGGER = re.compile(r"(搜索|搜一下|搜一搜|搜个|查一下|查查|查个|帮我查|上网查|查资料|联网查|百度|谷歌)")
_EMOTION_PAT = re.compile(r"(切换|开启|进入|启动|来点|用|变成|换).{0,6}(自然|开心|快乐|毒舌|傲娇|温柔|治愈)")
_CLEAR_PAT = re.compile(r"^(清空|清除|删除|忘记)(所有)?(记忆|对话|上下文|历史)|^忘了我?吧$")


def _extract_search_query(text: str) -> str:
    q = _SEARCH_TRIGGER.sub("", text, count=1)
    q = re.sub(r"[？?!！。～~\s]+$", "", q)
    q = re.sub(r"^(一下|个|下)", "", q.strip())
    return q.strip() or text.strip()


def _extract_city(text: str) -> Optional[str]:
    m = re.search(r"(?:查|看看|问下)?([\u4e00-\u9fa5]{2,8})(?:的)?(?:天气|气温|温度|下雨|下雪)", text)
    city = m.group(1) if m else None
    m2 = re.search(r"(?:在|看看|查)([\u4e00-\u9fa5]{2,8})(?:现在)?(?:天气|天气怎么样)", text)
    if m2:
        city = m2.group(1)
    if not city or city in CITY_BLACKLIST:
        return None
    return city


def detect_intent(text: str) -> Intent:
    t = text.strip()
    if _CLEAR_PAT.match(t):
        return Intent("clear", t)
    if re.match(r"^/?(退出|quit|exit)$", t):
        return Intent("chat", t)

    m = _EMOTION_PAT.search(t)
    if m:
        return Intent("emotion", t, emotion=m.group(2))

    if _TIME_PAT.search(t):
        return Intent("time", t)

    if _WEATHER_PAT.search(t):
        return Intent("weather", t, city=_extract_city(t))

    m = _MATH_PAT.search(t)
    if m and re.search(r"[\+\-\*\/×÷^]", m.group(0)):
        return Intent("math", t, expr=m.group(0).strip())

    if _SEARCH_TRIGGER.search(t):
        return Intent("search", t, query=_extract_search_query(t))

    return Intent("chat", t)


if __name__ == "__main__":
    cases = ["现在几点了", "查北京天气", "天气怎么样", "23*7+15等于几",
             "搜索黑洞是什么", "帮我查一下怎么做蛋糕", "切换毒舌模式", "清空记忆",
             "你好呀", "忘了我吧"]
    for c in cases:
        i = detect_intent(c)
        print(f"{c!r:24} -> {i.kind:8} {i.query or i.city or i.expr or i.emotion or ''}")
    print("意图识别自测通过 ✅")
