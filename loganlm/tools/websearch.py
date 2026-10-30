# -*- coding: utf-8 -*-
"""
LoganLM · 联网搜索工具（不依赖付费 API）
========================================

流程：DuckDuckGo HTML 版搜索 -> 抽取标题/链接/摘要 -> （可选）抓取网页正文
     -> 抽取式总结（句子按与查询的相关度排序）

这是 LoganLM "超越训练数据、获取实时信息" 的关键模块。
无需任何 API Key，网络不可用时所有函数安全返回空结果。

作者：罗根（Logan）
"""
from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional
from urllib.parse import parse_qs, quote, unquote, urlparse

import requests

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str = ""
    content: str = ""          # 抓取到的网页正文（可选）
    rank: int = 0

    def brief(self, n: int = 120) -> str:
        s = re.sub(r"\s+", " ", self.snippet or self.content or "").strip()
        return s[:n]


# ---------------------------------------------------------------------------
# 搜索
# ---------------------------------------------------------------------------
def search(query: str, topk: int = 4, timeout: float = 12.0) -> List[SearchResult]:
    """DuckDuckGo HTML 版搜索，返回前 topk 条结果。失败返回空列表。"""
    try:
        resp = requests.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            headers={"User-Agent": _UA, "Referer": "https://duckduckgo.com/"},
            timeout=timeout,
        )
        resp.raise_for_status()
        return _parse_ddg(resp.text, topk)
    except Exception:
        # 备用入口：lite 版
        try:
            resp = requests.get(
                "https://lite.duckduckgo.com/lite/",
                params={"q": query},
                headers={"User-Agent": _UA},
                timeout=timeout,
            )
            resp.raise_for_status()
            return _parse_ddg_lite(resp.text, topk)
        except Exception:
            return []


def _clean_url(raw: str) -> str:
    raw = html.unescape(raw)
    if raw.startswith("//"):
        raw = "https:" + raw
    # DDG 跳转链接：/l/?uddg=<encoded>
    if "duckduckgo.com/l/" in raw:
        try:
            qs = parse_qs(urlparse(raw).query)
            if "uddg" in qs:
                return unquote(qs["uddg"][0])
        except Exception:
            pass
    return raw


_TAG_RE = re.compile(r"<[^>]+>")


def _strip_tags(s: str) -> str:
    return re.sub(r"\s+", " ", _TAG_RE.sub("", s)).strip()


def _parse_ddg(page: str, topk: int) -> List[SearchResult]:
    results: List[SearchResult] = []
    # 标题链接
    link_iter = re.finditer(
        r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', page, re.S)
    # 摘要
    snippets = [m.group(1) for m in re.finditer(
        r'<a[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>', page, re.S)]
    for i, m in enumerate(link_iter):
        if i >= topk:
            break
        title = _strip_tags(m.group(2))
        if not title:
            continue
        snip = _strip_tags(html.unescape(snippets[i])) if i < len(snippets) else ""
        results.append(SearchResult(
            title=html.unescape(title), url=_clean_url(m.group(1)),
            snippet=snip, rank=i))
    return results


def _parse_ddg_lite(page: str, topk: int) -> List[SearchResult]:
    results: List[SearchResult] = []
    links = re.findall(r'<a[^>]+class="result-link"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', page, re.S)
    snippets = re.findall(r'<td[^>]*class="result-snippet"[^>]*>(.*?)</td>', page, re.S)
    for i, (href, title) in enumerate(links[:topk]):
        snip = _strip_tags(html.unescape(snippets[i])) if i < len(snippets) else ""
        results.append(SearchResult(
            title=_strip_tags(title), url=_clean_url(href), snippet=snip, rank=i))
    return results


# ---------------------------------------------------------------------------
# 网页抓取与正文抽取
# ---------------------------------------------------------------------------
_SCRIPT_RE = re.compile(
    r"<(script|style|noscript|svg|header|footer|nav|aside)[^>]*>.*?</\1>", re.S | re.I)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)


def fetch_page(url: str, timeout: float = 10.0, max_chars: int = 5000) -> str:
    """抓取网页并抽取可读正文（纯标准库级解析）。失败返回空串。"""
    try:
        resp = requests.get(url, headers={"User-Agent": _UA}, timeout=timeout, stream=True)
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "")
        if "image" in ctype or "pdf" in ctype:
            return ""
        raw = resp.content[:200_000]
        for enc in (resp.encoding, "utf-8", "gb18030"):
            if not enc:
                continue
            try:
                text = raw.decode(enc, errors="ignore")
                break
            except Exception:
                continue
        else:
            text = raw.decode("utf-8", errors="ignore")

        text = _COMMENT_RE.sub("", text)
        text = _SCRIPT_RE.sub("", text)
        # 优先取 <title> 与 <p> 段落
        title = re.search(r"<title[^>]*>(.*?)</title>", text, re.S | re.I)
        paras = re.findall(r"<p[^>]*>(.*?)</p>", text, re.S | re.I)
        body = "\n".join(_strip_tags(p) for p in paras)
        body = re.sub(r"\n{2,}", "\n", body)
        body = unicodedata.normalize("NFKC", body).strip()
        head = _strip_tags(html.unescape(title.group(1))) if title else ""
        out = (head + "\n" + body).strip()
        return out[:max_chars]
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# 抽取式总结（TF 相关度句子排序，纯手写无外部依赖）
# ---------------------------------------------------------------------------
_SENT_SPLIT_RE = re.compile(r"[。！？；!?;\n]")


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT_RE.split(text) if len(s.strip()) >= 6]


def _relevance(sentence: str, query: str) -> float:
    """查询词覆盖度 × 位置衰减（越靠前越重要）。"""
    q_terms = [w for w in re.findall(r"[\w\u4e00-\u9fff]+", query.lower()) if len(w) >= 2]
    if not q_terms:
        return 0.0
    s_low = sentence.lower()
    cover = sum(1 for w in q_terms if w in s_low) / len(q_terms)
    return cover * len(sentence) ** 0.25


def summarize_text(text: str, query: str = "", max_sentences: int = 3) -> str:
    """抽取式总结：按与 query 的相关度选出最重要的句子。"""
    sents = _sentences(text)
    if not sents:
        return ""
    scored = [( _relevance(s, query or text[:50]) + (1.0 if i < 3 else 0.0), i, s)
              for i, s in enumerate(sents[:120])]
    scored.sort(reverse=True)
    picked = sorted((i, s) for score, i, s in scored[:max_sentences])
    return "。".join(s for _, s in picked).rstrip("。") + "。"


def search_and_summarize(query: str, topk: int = 4, fetch_first: int = 2,
                         max_sentences: int = 4) -> dict:
    """搜索 + 抓取 + 总结 的一站式入口。返回 dict 供 reasoner 使用。"""
    results = search(query, topk=topk)
    for r in results[:fetch_first]:
        if r.url:
            r.content = fetch_page(r.url, max_chars=4000)
    bullets, sources = [], []
    for r in results:
        text = r.content or r.snippet
        s = summarize_text(text, query, max_sentences=2) if r.content else r.brief()
        if s:
            bullets.append(s)
            sources.append(r)
    answer = summarize_text("。".join(bullets) or "", query, max_sentences=max_sentences)
    return {
        "query": query,
        "answer": answer,
        "results": results,
        "sources": [(r.title, r.url) for r in sources],
        "ok": bool(bullets),
    }


if __name__ == "__main__":
    import json
    out = search_and_summarize("Python 是什么语言")
    print(json.dumps(out["sources"], ensure_ascii=False, indent=2))
    print("总结:", out["answer"][:300])
