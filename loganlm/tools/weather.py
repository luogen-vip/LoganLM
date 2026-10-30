# -*- coding: utf-8 -*-
"""
LoganLM · 天气与时间工具
========================

天气：wttr.in 免费服务（无需 Key），返回中文描述。
时间：机器本地时钟（Asia/Shanghai 优先）。

作者：罗根（Logan）
"""
from __future__ import annotations

import datetime
import re
from typing import Optional

import requests

_WMO_ZH = {
    113: "晴", 116: "多云", 119: "阴", 122: "阴天",
    143: "薄雾", 176: "局部阵雨", 179: "局部降雪", 182: "冻雨",
    185: "冻毛雨", 200: "雷阵雨", 227: "吹雪", 230: "暴风雪",
    248: "雾", 260: "冻雾", 263: "毛毛雨", 266: "细雨",
    281: "冻毛毛雨", 284: "强冻毛毛雨", 293: "零星小雨", 296: "小雨",
    299: "局部中雨", 302: "中雨", 305: "局部大雨", 308: "大雨",
    311: "冻雨", 314: "强冻雨", 317: "雨夹雪", 320: "强雨夹雪",
    323: "零星小雪", 326: "小雪", 329: "局部中雪", 332: "中雪",
    335: "局部大雪", 338: "大雪", 350: "冰雹", 353: "阵雨",
    356: "中阵雨", 359: "暴雨", 362: "小雨夹雪", 365: "中雨夹雪",
    368: "小阵雪", 371: "中阵雪", 374: "小冰雹", 377: "冰粒",
    386: "强雷阵雨", 389: "强雷暴", 392: "雷伴雪", 395: "强雷伴雪",
}


def get_weather(city: str, timeout: float = 10.0) -> Optional[str]:
    """查询城市实时天气，返回一段中文描述；失败返回 None。"""
    try:
        r = requests.get(
            f"https://wttr.in/{city}",
            params={"format": "j1", "lang": "zh"},
            headers={"User-Agent": "LoganLM/1.0"},
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        cur = data["current_condition"][0]
        code = int(cur.get("weatherCode", 116))
        desc = _WMO_ZH.get(code, cur["weatherDesc"][0]["value"])
        temp = cur["temp_C"]
        feels = cur.get("FeelsLikeC", temp)
        humid = cur.get("humidity", "?")
        wind = cur.get("winddir16Point", "")
        wspd = cur.get("windspeedKmph", "?")

        days = data.get("weather", [])
        forecast = ""
        if days:
            d0 = days[0]
            forecast = (f"，今天 {d0.get('mintempC','?')}~{d0.get('maxtempC','?')}℃")
            if len(days) > 1:
                d1 = days[1]
                forecast += f"，明天 {d1.get('mintempC','?')}~{d1.get('maxtempC','?')}℃"

        return (f"{city}当前天气：{desc}，气温 {temp}℃（体感 {feels}℃），"
                f"湿度 {humid}%，{wind}风 {wspd} km/h{forecast}。")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 时间
# ---------------------------------------------------------------------------
_WEEK_ZH = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]


def now_text(tz_offset_hours: Optional[float] = None) -> str:
    """当前日期时间文本（默认本机时区）。"""
    now = datetime.datetime.now()
    if tz_offset_hours is not None:
        now = datetime.datetime.utcnow() + datetime.timedelta(hours=tz_offset_hours)
    return (f"现在是 {now.year}年{now.month}月{now.day}日 "
            f"{_WEEK_ZH[now.weekday()]} {now.hour:02d}:{now.minute:02d}。")


def now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


CITY_OFFSET = {
    "北京": 8.0, "上海": 8.0, "广州": 8.0, "深圳": 8.0, "成都": 8.0, "杭州": 8.0,
    "重庆": 8.0, "西安": 8.0, "武汉": 8.0, "南京": 8.0, "香港": 8.0, "台北": 8.0,
    "东京": 9.0, "首尔": 9.0, "新加坡": 8.0, "伦敦": 0.0, "巴黎": 1.0,
    "纽约": -5.0, "洛杉矶": -8.0, "莫斯科": 3.0,
}


def city_time(city: str) -> Optional[str]:
    """常见城市当前时间（内置时区表，无需联网）。"""
    off = CITY_OFFSET.get(city)
    if off is None:
        return None
    now = datetime.datetime.utcnow() + datetime.timedelta(hours=off)
    return (f"{city}现在是 {now.year}年{now.month}月{now.day}日 "
            f"{_WEEK_ZH[now.weekday()]} {now.hour:02d}:{now.minute:02d}。")


if __name__ == "__main__":
    print(now_text())
    print(city_time("东京"))
    w = get_weather("北京")
    print("天气:", w if w else "（网络不可用，返回 None）")
