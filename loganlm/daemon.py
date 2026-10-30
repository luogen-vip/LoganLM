#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoganLM 常驻守护进程 —— 给网站后端提供低延迟 JSON 服务
========================================================

作者：罗根（Logan）

用法：
    python -m loganlm.daemon --port 3031

接口（仅本机可访问）：
    GET  /health   → {"ok": true, "params_m": 1.81}
    GET  /history?session_id=x → {"ok":true,"history":[...],"emotion":"自然"}
    POST /chat     {"message","session_id","emotion"} → {"reply","tool","sources"}
    POST /reset    {"session_id"}

模型只在启动时加载一次（约 4 秒），之后每条回复 1~2 秒。
被 Next.js 后端按需拉起；崩溃后由下一次请求自动重启。
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .chat import ChatSession, DEFAULT_CKPT

LOCK = threading.Lock()
SESSION: ChatSession = None  # type: ignore
VERSION = "loganlm-daemon-1.1"


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length", 0) or 0)
        try:
            return json.loads(self.rfile.read(n).decode("utf-8") or "{}")
        except Exception:
            return {}

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/health"):
            self._send(200, {"ok": True, "version": VERSION,
                             "params_m": round(SESSION.model.num_params() / 1e6, 2)})
        elif self.path.startswith("/history"):
            q = dict(p.split("=", 1) for p in self.path.split("?", 1)[-1].split("&") if "=" in p)
            sid = q.get("session_id", "web")
            with LOCK:
                SESSION.session_id = sid
                hist = SESSION.history()
                emo = SESSION.emotion
            self._send(200, {"ok": True, "history": hist, "emotion": emo})
        else:
            self._send(404, {"ok": False, "error": "unknown endpoint"})

    def do_POST(self) -> None:  # noqa: N802
        req = self._body()
        sid = str(req.get("session_id", "web"))
        if self.path == "/chat":
            # 可选图片：base64 / dataURL → 临时文件 → 识图
            img_path = None
            img_b64 = req.get("image")
            if img_b64:
                try:
                    if isinstance(img_b64, str) and "," in img_b64:
                        img_b64 = img_b64.split(",", 1)[1]
                    img_path = os.path.join(
                        tempfile.gettempdir(), f"loganlm_web_{int(time.time() * 1000)}.png")
                    with open(img_path, "wb") as fh:
                        fh.write(base64.b64decode(img_b64))
                except Exception:
                    img_path = None
            with LOCK:
                SESSION.session_id = sid
                emo = str(req.get("emotion") or "")
                if emo and emo != SESSION.emotion:
                    SESSION.emotion = emo
                    SESSION.memory.set_emotion(sid, emo)
                r = SESSION.reply(str(req.get("message", "")), image_path=img_path)
            self._send(200, r)
        elif self.path == "/reset":
            with LOCK:
                SESSION.memory.clear(sid)
            self._send(200, {"ok": True, "reply": "记忆已清空"})
        else:
            self._send(404, {"ok": False, "error": "unknown endpoint"})

    def log_message(self, fmt, *args) -> None:
        pass


def main() -> None:
    global SESSION
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=3031)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    SESSION = ChatSession(DEFAULT_CKPT, session_id="web")
    print(f"[loganlm-daemon] listening on {args.host}:{args.port} "
          f"(model {SESSION.model.num_params() / 1e6:.2f}M)", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
