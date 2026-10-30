# -*- coding: utf-8 -*-
"""
LoganLM JSON API —— 给网站 / 程序调用的命令行接口
==================================================

作者：罗根（Logan）

用法：
    python -m loganlm.api --message "你好" --session web1
    python -m loganlm.api --message "切换毒舌" --session web1
    python -m loganlm.api --reset --session web1
    python -m loganlm.api --history --session web1

输出一行 JSON：
    {"ok": true, "reply": "...", "tool": "model", "sources": []}

每次调用独立加载模型（约 3~5 秒），换来的是零状态、崩溃自动恢复，
配合 memory 的持久化，网站侧天然支持多会话。
"""
from __future__ import annotations

import argparse
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description="LoganLM JSON API")
    ap.add_argument("--message", "-m", default="", help="用户消息")
    ap.add_argument("--session", default="default", help="会话 ID（同名会话共享记忆）")
    ap.add_argument("--ckpt", default=None, help="模型目录")
    ap.add_argument("--emotion", default=None, help="直接设置情绪")
    ap.add_argument("--reset", action="store_true", help="清空该会话记忆")
    ap.add_argument("--history", action="store_true", help="返回历史对话")
    args = ap.parse_args()

    try:
        from .chat import ChatSession
        kwargs = {"session_id": args.session}
        if args.ckpt:
            kwargs["ckpt_dir"] = args.ckpt
        s = ChatSession(**kwargs)

        if args.reset:
            s.reset()
            print(json.dumps({"ok": True, "reply": "记忆已清空", "tool": "clear",
                              "sources": []}, ensure_ascii=False))
            return 0
        if args.history:
            print(json.dumps({"ok": True, "history": s.history(),
                              "emotion": s.emotion}, ensure_ascii=False))
            return 0
        if args.emotion:
            s.emotion = s.emotion  # noqa 占位，真实设置在 reply 流程里
        if not args.message:
            print(json.dumps({"ok": False, "error": "缺少 --message"},
                             ensure_ascii=False))
            return 1
        r = s.reply(args.message)
        print(json.dumps({"ok": True, **r}, ensure_ascii=False))
        return 0
    except Exception as e:  # 网站侧需要稳定 JSON，永不抛裸异常
        print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}",
                          "reply": f"呜，内部出了点小差错：{e}", "tool": "error",
                          "sources": []}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
