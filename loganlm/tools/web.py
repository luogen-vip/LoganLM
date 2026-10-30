# -*- coding: utf-8 -*-
"""
LoganLM 联网工具 —— 搜索 / 天气 / 时间
=======================================

作者：罗根（Logan）

只用 Python 标准库（urllib / json / re / html），不依赖任何第三方
HTTP 或搜索库。模型负责"说话"，本模块负责"查事实"。
"""
from __future__ import annotations

import datetime
import json
import re
import urllib.parse
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (compatible; LoganLM/0.1; +https://github.com/)"}


def _http_get(url: str, timeout: int = 10) -> str:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def _strip_tags(s: str) -> str:
    s = re.sub(r"<[^>]+>", "", s)
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------------------
# 联网搜索（DuckDuckGo HTML 版，无需 API key）
# ---------------------------------------------------------------------------
def web_search(query: str, topk: int = 4):
    """网页搜索，返回 [{title, url, snippet}]，失败返回 []。"""
    results = []
    try:
        url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query)
        page = _http_get(url, timeout=10)
        blocks = re.findall(
            r'<a[^>]+class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            page, re.S)
        snippets = re.findall(
            r'<a[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>',
            page, re.S)
        for i, (href, title) in enumerate(blocks[:topk]):
            # DuckDuckGo 的跳转链接还原成真实 URL
            if "uddg=" in href:
                q = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                href = q.get("uddg", [href])[0]
            snippet = _strip_tags(snippets[i]) if i < len(snippets) else ""
            results.append({
                "title": _strip_tags(title),
                "url": href,
                "snippet": snippet,
            })
    except Exception as e:
        print(f"[tools] 搜索失败：{e}")
    return results


def summarize_search(query: str, topk: int = 4) -> str:
    """搜索并拼成一段可以直接塞进 prompt 的资料文本。"""
    items = web_search(query, topk)
    if not items:
        return "（没有搜索到相关资料）"
    lines = [f"「{query}」的搜索结果："]
    for i, it in enumerate(items, 1):
        lines.append(f"{i}. {it['title']} —— {it['snippet']}（{it['url']}）")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 天气（wttr.in，免费无 key）
# ---------------------------------------------------------------------------
WEATHER_ZH = {
    "Sunny": "晴", "Clear": "晴", "Partly cloudy": "多云", "Cloudy": "多云",
    "Overcast": "阴", "Mist": "薄雾", "Fog": "雾", "Patchy rain possible": "可能有雨",
    "Light drizzle": "毛毛雨", "Light rain": "小雨", "Moderate rain": "中雨",
    "Heavy rain": "大雨", "Light rain shower": "阵雨", "Thunderstorm": "雷雨",
    "Light snow": "小雪", "Moderate snow": "中雪", "Heavy snow": "大雪",
    "Patchy snow possible": "可能有雪", "Light snow showers": "阵雪",
}


def get_weather(city: str) -> str:
    """实时天气一句话。失败时返回带说明的提示。"""
    try:
        url = f"https://wttr.in/{urllib.parse.quote(city)}?format=j1"
        data = json.loads(_http_get(url, timeout=8))
        cur = data["current_condition"][0]
        desc_en = cur["weatherDesc"][0]["value"]
        desc = WEATHER_ZH.get(desc_en, desc_en)
        return (f"{city} 现在{desc}，气温 {cur['temp_C']}°C，"
                f"体感 {cur['FeelsLikeC']}°C，湿度 {cur['humidity']}%。")
    except Exception as e:
        return f"（天气查询失败：{e}）"


# ---------------------------------------------------------------------------
# 时间
# ---------------------------------------------------------------------------
def get_now_str() -> str:
    now = datetime.datetime.now()
    wd = "一二三四五六日"[now.weekday()]
    return (f"{now.year}年{now.month}月{now.day}日 星期{wd} "
            f"{now.strftime('%H:%M:%S')}")


if __name__ == "__main__":
    print("当前时间:", get_now_str())
    print("天气测试:", get_weather("北京"))
    print("搜索测试:", summarize_search("什么是大语言模型", topk=2))
