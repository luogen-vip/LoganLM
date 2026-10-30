# -*- coding: utf-8 -*-
"""
LoganLM 分词器 —— 从 0 手写的 Byte-level BPE
=============================================

作者：罗根（Logan）

不依赖任何第三方分词库（没有 sentencepiece / tokenizers / tiktoken）。

核心思想：
  1. 万物皆字节：任何文本（中文、英文、表情、代码……）先变成 UTF-8 字节序列。
     基础词表就是 256 个字节，天然覆盖一切文字，永远不会出现无法表示的字符。
  2. BPE 合并：训练时统计语料中出现最频繁的相邻 token 对，把它们合并成一个
     新 token，重复几千次。高频字词（比如"的"、"你好"的 UTF-8 字节）会
     合并成单个 token，压缩率越来越高。
  3. 特殊 token：<bos>/<eos>/<user>/<assistant>/<img> 等，用于对话控制和
     多模态占位，训练和推理时都不会被拆开。
"""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import Dict, List, Tuple

# 特殊 token 的固定编号（紧跟在 256 个字节之后）
SPECIAL_TOKENS = [
    "<pad>",        # 256 填充
    "<bos>",        # 257 一段对话的开始
    "<eos>",        # 258 一句话的结束（生成到它就停）
    "<unk>",        # 259 兜底
    "<sys>",        # 260 系统提示
    "<user>",       # 261 用户发言开始
    "<assistant>",  # 262 模型发言开始
    "<img>",        # 263 图像特征开始
    "</img>",       # 264 图像特征结束
    "<search>",     # 265 搜索结果开始
    "</search>",    # 266 搜索结果结束
]
PAD, BOS, EOS, UNK = 256, 257, 258, 259
SYS, USER, ASSISTANT, IMG, IMG_END, SEARCH, SEARCH_END = 260, 261, 262, 263, 264, 265, 266


class BPETokenizer:
    """Byte-level BPE 分词器。

    - id 0~255        : 256 个原始字节
    - id 256~(256+S-1): 特殊 token
    - id 256+S 之后   : 训练学到的合并 token（越靠前 = 越常出现）
    """

    def __init__(self) -> None:
        self.merges: Dict[Tuple[int, int], int] = {}   # (id,id) -> 新id
        self.vocab: Dict[int, bytes] = {}              # id -> 字节串
        self.special_to_id: Dict[str, int] = {}
        self._build_base()

    # ------------------------------------------------------------------
    # 基础词表
    # ------------------------------------------------------------------
    def _build_base(self) -> None:
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, t in enumerate(SPECIAL_TOKENS):
            tid = 256 + i
            self.special_to_id[t] = tid
            self.vocab[tid] = t.encode("utf-8")

    @property
    def base_vocab_size(self) -> int:
        return 256 + len(SPECIAL_TOKENS)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # ------------------------------------------------------------------
    # 训练：学习合并规则
    # ------------------------------------------------------------------
    def train(self, text: str, vocab_size: int = 4096, verbose: bool = False) -> "BPETokenizer":
        """在语料上学习 BPE 合并，直到词表达到 vocab_size（或语料不够频繁为止）。"""
        ids: List[int] = list(text.encode("utf-8"))
        n_merges = max(0, vocab_size - self.base_vocab_size)
        for step in range(n_merges):
            stats = Counter(zip(ids, ids[1:]))
            if not stats:
                break
            pair, count = stats.most_common(1)[0]
            if count < 2:  # 没有出现两次以上的字节对了，提前收敛
                break
            new_id = self.base_vocab_size + step
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            ids = self._merge_ids(ids, pair, new_id)
            if verbose and step % 200 == 0:
                print(f"  BPE 合并 {step + 1}/{n_merges}: {pair} -> {new_id}（出现 {count} 次）")
        return self

    @staticmethod
    def _merge_ids(ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        """把序列中所有出现的 pair 替换成 new_id（一遍线性扫描）。"""
        out, i = [], 0
        a, b = pair
        n = len(ids)
        while i < n:
            if i < n - 1 and ids[i] == a and ids[i + 1] == b:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    # ------------------------------------------------------------------
    # 编码 / 解码
    # ------------------------------------------------------------------
    def _encode_chunk(self, chunk: List[int]) -> List[int]:
        """把一段原始字节 id 序列，按学到的合并规则压缩。"""
        while len(chunk) >= 2:
            best_pair, best_rank = None, None
            for i in range(len(chunk) - 1):
                pair = (chunk[i], chunk[i + 1])
                rank = self.merges.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_pair, best_rank = pair, rank
            if best_pair is None:
                break
            chunk = self._merge_ids(chunk, best_pair, self.merges[best_pair])
        return chunk

    def _special_pattern(self) -> "re.Pattern":
        alternation = "|".join(
            re.escape(t) for t in sorted(self.special_to_id, key=len, reverse=True)
        )
        return re.compile("(" + alternation + ")")

    def encode(self, text: str) -> List[int]:
        """文本 -> token id 序列。特殊 token 会被整体识别，不会被拆成字节。"""
        ids: List[int] = []
        for part in self._special_pattern().split(text):
            if not part:
                continue
            if part in self.special_to_id:
                ids.append(self.special_to_id[part])
            else:
                ids.extend(self._encode_chunk(list(part.encode("utf-8"))))
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """token id 序列 -> 文本。非法/未知 id 直接跳过。"""
        buf = bytearray()
        special_ids = set(self.special_to_id.values())
        for i in ids:
            b = self.vocab.get(i)
            if b is None:
                continue
            if skip_special and i in special_ids:
                continue
            buf.extend(b)
        return buf.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # 保存 / 加载
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        data = {
            "type": "LoganLM-BPE-v1",
            "special_tokens": SPECIAL_TOKENS,
            "vocab": {str(k): v.hex() for k, v in self.vocab.items()},
            "merges": [[a, b, new] for (a, b), new in self.merges.items()],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        tok = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data.get("type") == "LoganLM-BPE-v1", "分词器文件格式不对"
        tok.merges = {(a, b): new for a, b, new in data["merges"]}
        # 完整恢复词表（含 256 字节 + 特殊 token + 全部合并 token），
        # 千万不能再调 _build_base()，否则会把合并 token 的解码映射抹掉
        tok.vocab = {int(k): bytes.fromhex(v) for k, v in data["vocab"].items()}
        tok.special_to_id = {t: 256 + i for i, t in enumerate(SPECIAL_TOKENS)}
        # 校验：基础部分必须和文件里的一致
        for i in range(256):
            assert tok.vocab.get(i) == bytes([i]), "分词器文件损坏（基础字节缺失）"
        return tok


if __name__ == "__main__":
    # 自测：训练一个玩具词表并做往返编解码
    t = BPETokenizer().train("你好，我是小罗！你好呀，小罗。" * 20, vocab_size=512)
    s = "你好，我是小罗！"
    ids = t.encode(s)
    print("词表:", t.vocab_size, "| 编码:", ids[:10], "| 还原:", t.decode(ids))
    assert t.decode(ids) == s
    print("分词器自测通过 ✅")
