# -*- coding: utf-8 -*-
"""
LoganLM 持久记忆
=================

用户需求："上下文要有记忆，只要不手动清空，所有内容都记住。"

实现：每个会话一个 JSON 文件，存放在 ~/.loganlm/ 目录：
    session_<id>.json = {
        "history": [["用户说", "小罗答"], ...],   # 全部对话历史
        "emotion": "自然",                        # 当前情绪模式
        "created": 时间戳, "updated": 时间戳
    }

进程重启后记忆仍在；只有用户说"清空记忆/忘了我吧"（或点网页上的按钮）才会删除。
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import List, Optional, Tuple


def _safe_session(session_id: str) -> str:
    """会话 ID 只允许字母数字下划线连字符，防止路径穿越。"""
    sid = re.sub(r"[^A-Za-z0-9_-]", "_", str(session_id))[:64]
    return sid or "default"


class MemoryStore:
    def __init__(self, memory_dir: Optional[str] = None) -> None:
        self.dir = os.path.expanduser(memory_dir or "~/.loganlm")
        os.makedirs(self.dir, exist_ok=True)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def _path(self, session_id: str) -> str:
        return os.path.join(self.dir, f"session_{_safe_session(session_id)}.json")

    def _load(self, session_id: str) -> dict:
        p = self._path(session_id)
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"history": [], "emotion": "自然",
                "created": time.time(), "updated": time.time()}

    def _save(self, session_id: str, data: dict) -> None:
        data["updated"] = time.time()
        tmp = self._path(session_id) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, self._path(session_id))  # 原子写入

    # ------------------------------------------------------------------
    def add_turn(self, session_id: str, user: str, assistant: str) -> None:
        with self._lock:
            data = self._load(session_id)
            data["history"].append([user, assistant])
            self._save(session_id, data)

    def get_history(self, session_id: str, limit: Optional[int] = None) -> List[Tuple[str, str]]:
        hist = self._load(session_id).get("history", [])
        if limit:
            hist = hist[-limit:]
        return [(u, a) for u, a in hist]

    def set_emotion(self, session_id: str, emotion: str) -> None:
        with self._lock:
            data = self._load(session_id)
            data["emotion"] = emotion
            self._save(session_id, data)

    def get_emotion(self, session_id: str) -> str:
        return self._load(session_id).get("emotion", "自然")

    def clear(self, session_id: str) -> None:
        """手动清空：删除这个会话的记忆文件。"""
        with self._lock:
            p = self._path(session_id)
            if os.path.exists(p):
                os.remove(p)

    def turn_count(self, session_id: str) -> int:
        return len(self._load(session_id).get("history", []))


if __name__ == "__main__":
    m = MemoryStore("/tmp/loganlm_test_mem")
    m.add_turn("t1", "你好", "你好呀！")
    m.add_turn("t1", "我是谁", "你告诉我嘛～")
    assert m.get_history("t1") == [("你好", "你好呀！"), ("我是谁", "你告诉我嘛～")]
    m.set_emotion("t1", "毒舌")
    assert m.get_emotion("t1") == "毒舌"
    m.clear("t1")
    assert m.turn_count("t1") == 0
    print("记忆模块自测通过 ✅")
