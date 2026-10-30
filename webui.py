#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoganLM · 网页聊天（零依赖，标准库实现）
========================================

作者：罗根（Logan）

用法：
    python webui.py              # 打开 http://localhost:7860 即可和小罗聊天
    python webui.py --port 8000

特点：
  - 不需要 Flask / FastAPI / npm，纯 Python 标准库（http.server）
  - 模型常驻内存，回复快
  - 多会话：网页自动生成 session id，记忆持久化在 ~/.loganlm
  - 同一个编排器：工具（时间/天气/算术）、联网搜索、情绪、记忆全部可用
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from loganlm.chat import ChatSession, DEFAULT_CKPT

# 模型全局只加载一次
SESSION = ChatSession(DEFAULT_CKPT, session_id="web")

PAGE = """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>小罗 · LoganLM 聊天</title>
<style>
  :root { --bg:#0f1220; --card:#171b2e; --me:#4f6ef7; --ro:#232842; --tx:#e8eaf6; --sub:#8b91b0; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:linear-gradient(160deg,#0f1220 0%,#141a33 60%,#1a1440 100%);
         color:var(--tx); font-family:"PingFang SC","Microsoft YaHei",system-ui,sans-serif;
         min-height:100vh; display:flex; flex-direction:column; }
  header { padding:14px 20px; display:flex; align-items:center; gap:10px;
           border-bottom:1px solid #232849; }
  header .logo { width:38px; height:38px; border-radius:12px; font-size:20px;
           background:linear-gradient(135deg,#4f6ef7,#9b5cf6);
           display:flex; align-items:center; justify-content:center; }
  header h1 { font-size:17px; } header p { font-size:12px; color:var(--sub); }
  header .btn { margin-left:auto; background:#232849; border:none; color:#aab2e8;
           padding:7px 14px; border-radius:10px; cursor:pointer; font-size:13px; }
  header .btn:hover { background:#2d3459; }
  #chat { flex:1; overflow-y:auto; padding:20px; display:flex; flex-direction:column; gap:14px; }
  .row { display:flex; gap:10px; align-items:flex-start; max-width:760px; width:100%; margin:0 auto; }
  .row.me { flex-direction:row-reverse; }
  .avatar { width:34px; height:34px; border-radius:10px; flex:none; font-size:16px;
            display:flex; align-items:center; justify-content:center; }
  .row.ro .avatar { background:linear-gradient(135deg,#4f6ef7,#9b5cf6); }
  .row.me .avatar { background:#2a3054; }
  .bubble { padding:11px 14px; border-radius:14px; line-height:1.7; font-size:14.5px;
            white-space:pre-wrap; word-break:break-word; }
  .row.ro .bubble { background:var(--ro); border-top-left-radius:4px; }
  .row.me .bubble { background:var(--me); border-top-right-radius:4px; }
  .src { margin-top:8px; font-size:12px; color:#9fb0ff; }
  .src a { color:#9fb0ff; text-decoration:none; margin-right:10px; }
  .tag { display:inline-block; font-size:11px; color:#8fd3a8; margin-bottom:4px; }
  footer { padding:14px 20px 22px; }
  .emotions { display:flex; gap:8px; justify-content:center; margin-bottom:10px; flex-wrap:wrap; }
  .emotions button { background:#1d2340; border:1px solid #2c3358; color:#aab2e8;
            padding:5px 13px; border-radius:999px; font-size:12.5px; cursor:pointer; }
  .emotions button.on { background:var(--me); color:#fff; border-color:var(--me); }
  .inputbar { display:flex; gap:10px; max-width:760px; margin:0 auto; width:100%; }
  .inputbar input { flex:1; background:var(--card); border:1px solid #2c3358; color:var(--tx);
            padding:13px 16px; border-radius:14px; font-size:14.5px; outline:none; }
  .inputbar input:focus { border-color:#4f6ef7; }
  .inputbar button { background:linear-gradient(135deg,#4f6ef7,#9b5cf6); color:#fff; border:none;
            padding:0 22px; border-radius:14px; font-size:15px; cursor:pointer; }
  .inputbar button:disabled { opacity:.5; }
  .hint { text-align:center; color:var(--sub); font-size:11.5px; margin-top:9px; }
  .dot { display:inline-block; animation:blink 1.2s infinite; }
  .dot:nth-child(2){animation-delay:.2s} .dot:nth-child(3){animation-delay:.4s}
  @keyframes blink { 0%,80%,100%{opacity:.2} 40%{opacity:1} }
</style>
</head>
<body>
<header>
  <div class="logo">🤖</div>
  <div><h1>小罗 · LoganLM</h1><p>从 0 手写的 AI 大模型 · 作者 罗根</p></div>
  <button class="btn" onclick="resetMem()">🧹 清空记忆</button>
</header>
<div id="chat"></div>
<footer>
  <div class="emotions" id="emos"></div>
  <div class="inputbar">
    <input id="q" placeholder="和小罗说点什么…（试试：查北京天气 / 搜索 最新新闻 / 23*7+15）" autofocus>
    <button id="send" onclick="send()">发送</button>
  </div>
  <p class="hint">记忆保存在服务器 ~/.loganlm，不点"清空"就一直记得 · 切情绪会让回答变风格</p>
</footer>
<script>
const SID = Math.random().toString(36).slice(2, 10);
let EMO = "自然";
const EMOS = ["自然","开心","毒舌","温柔"];
const chat = document.getElementById("chat");

function esc(s){ const d=document.createElement("div"); d.textContent=s; return d.innerHTML; }
function add(role, html){
  const row=document.createElement("div"); row.className="row "+role;
  row.innerHTML = `<div class="avatar">${role==="me"?"🙂":"🤖"}</div><div class="bubble">${html}</div>`;
  chat.appendChild(row); chat.scrollTop=chat.scrollHeight; return row;
}
function renderEmos(){
  document.getElementById("emos").innerHTML = EMOS.map(e=>
    `<button class="${e===EMO?"on":""}" onclick="setEmo('${e}')">${e==="自然"?"🌿":e==="开心"?"😄":e==="毒舌"?"😼":"🌸"} ${e}</button>`).join("");
}
async function setEmo(e){ EMO=e; renderEmos(); add("me", esc("切换"+e+"模式"));
  const r=await fetch("/api/chat",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({message:"切换"+e+"模式",session_id:SID})});
  const d=await r.json(); add("ro", esc(d.reply||"")); }
function thinking(){ return add("ro", '小罗正在思考<span class="dot">•</span><span class="dot">•</span><span class="dot">•</span>'); }

async function send(){
  const inp=document.getElementById("q"); const q=inp.value.trim(); if(!q) return;
  inp.value=""; add("me", esc(q));
  const btn=document.getElementById("send"); btn.disabled=true;
  const t=thinking();
  try{
    const r=await fetch("/api/chat",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({message:q,session_id:SID,emotion:EMO})});
    const d=await r.json();
    let html=esc(d.reply||"(空)");
    if(d.tool && d.tool!=="model" && d.tool!=="chat") html=`<span class="tag">⚡ ${d.tool} 工具</span><br>`+html;
    if(d.sources && d.sources.length) html+=`<div class="src">🔎 来源：`+
      d.sources.map(s=>`<a href="${s.url}" target="_blank" rel="noopener">${esc(s.title||s.url)}</a>`).join("")+`</div>`;
    t.querySelector(".bubble").innerHTML=html;
  }catch(e){ t.querySelector(".bubble").textContent="连接出错了："+e; }
  btn.disabled=false; inp.focus(); chat.scrollTop=chat.scrollHeight;
}
async function resetMem(){
  await fetch("/api/reset",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({session_id:SID})});
  add("ro","🧹 记忆已清空，我们重新认识吧！你好，我是小罗～");
}
document.getElementById("q").addEventListener("keydown",e=>{ if(e.key==="Enter") send(); });
add("ro","你好呀！我是小罗，一个由罗根从零手写出来的 AI 助手 🎉\\n\\n可以跟我聊天、让我查天气/时间/算数，说「搜索+关键词」我还能上网查资料。我会一直记得我们聊过的话哦！");
renderEmos();
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/api/health":
            self._json({"ok": True, "model": "LoganLM demo",
                        "params_m": round(SESSION.model.num_params() / 1e6, 2)})
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        n = int(self.headers.get("Content-Length", 0) or 0)
        try:
            req = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
        except Exception:
            self._json({"ok": False, "error": "bad json"}, 400)
            return
        sid = str(req.get("session_id", "web"))
        if self.path == "/api/chat":
            if req.get("emotion") and req["emotion"] != SESSION.emotion:
                SESSION.emotion = str(req["emotion"])
                SESSION.memory.set_emotion(sid, SESSION.emotion)
            SESSION.session_id = sid
            r = SESSION.reply(str(req.get("message", "")))
            self._json(r)
        elif self.path == "/api/reset":
            SESSION.memory.clear(sid)
            self._json({"ok": True, "reply": "记忆已清空"})
        else:
            self._json({"ok": False, "error": "unknown endpoint"}, 404)

    def log_message(self, fmt, *args) -> None:  # 安静模式
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description="LoganLM 网页聊天")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    print(f"小罗网页版启动：http://localhost:{args.port}  （Ctrl+C 退出）")
    print(f"模型 {SESSION.model.num_params() / 1e6:.2f}M 参数，记忆目录 ~/.loganlm")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
