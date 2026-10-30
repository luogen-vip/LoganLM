#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoganLM · 小罗命令行聊天
========================

作者：罗根（Logan）

用法：
    python chat.py                      # 和小罗聊天（默认读 checkpoints/demo）
    python chat.py --session 我的会话    # 换一个记忆空间
    python chat.py --ckpt checkpoints/demo

聊天中可以直接说：
    切换毒舌模式 / 开心模式 / 温柔模式 / 自然模式     切情绪
    现在几点了 / 查北京天气 / 23*7+15等于几           工具直接答
    搜索 大语言模型是什么                             联网搜索再总结
    清空记忆 / 忘了我吧                               清空本会话记忆
    /退出                                             结束（Ctrl+C 也行）
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from loganlm.chat import ChatSession, DEFAULT_CKPT


def main() -> None:
    ap = argparse.ArgumentParser(description="小罗命令行聊天")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT, help="模型目录")
    ap.add_argument("--session", default="cli", help="会话 ID（同名会话共享记忆）")
    ap.add_argument("--memory-dir", default=None, help="记忆存储目录（默认 ~/.loganlm）")
    args = ap.parse_args()

    try:
        s = ChatSession(args.ckpt, args.session, args.memory_dir)
    except FileNotFoundError as e:
        print(e)
        return

    n = s.model.num_params() / 1e6
    print(f"小罗已上线 🎉（LoganLM 演示模型 {n:.2f}M 参数 | 会话「{args.session}」| "
          f"记忆已存 {s.memory.turn_count(args.session)} 轮）")
    print("试试：切换开心模式 | 查北京天气 | 搜索 最新新闻 | 23*7+15 | 清空记忆 | /退出\n")

    if sys.stdin.isatty():
        while True:
            try:
                q = input("你> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n小罗> 再见啦，想我随时回来～")
                break
            if not q:
                continue
            if q in ("/退出", "/quit", "/exit", "quit", "exit"):
                print("小罗> 再见啦，想我随时回来～")
                break
            if q == "/help":
                print("直接说话即可。可用：切情绪 / 时间 / 天气 / 算术 / 搜索 / 清空记忆")
                continue
            r = s.reply(q)
            print(f"小罗> {r['reply']}")
            for it in r.get("sources", []):
                print(f"       · {it['title']} {it['url']}")
    else:
        # 管道模式：一行一个问题，方便自动化测试
        for line in sys.stdin:
            q = line.strip()
            if not q:
                continue
            if q in ("/退出", "/quit", "/exit", "quit", "exit"):
                break
            r = s.reply(q)
            print(f"小罗> {r['reply']}", flush=True)


if __name__ == "__main__":
    main()
