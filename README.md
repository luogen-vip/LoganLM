<div align="center">

# LoganLM · 罗根大模型

**从 0 手写的多模态大语言模型系统 —— 每一行代码都是重新实现的**

分词器 · Transformer 语言模型 · 联网搜索 · 持久记忆 · 情绪引擎 · 命令行/网页聊天

作者：**罗根 (Logan)** · 协议：MIT · 版本：v1.1.0

</div>

---

## 这是什么？

LoganLM 是一个**完全从零开始**实现的 AI 大模型系统：不使用任何现成模型、
预训练权重或第三方分词/模型库，只参考业界公开的架构思想
（LLaMA 式 Transformer、Byte-level BPE、GQA、SwiGLU），
然后每一行代码亲手重写。

它能做到：

| 能力 | 说明 |
| --- | --- |
| 💬 基础聊天 | 自研中文对话语料训练，有人格（小罗，作者罗根） |
| 🧠 持久记忆 | JSON 存储，**只要不手动清空，一直记得**聊过的每一轮（重启也不丢） |
| 🔍 联网搜索 | 说"搜索 xxx"自动搜索 → 拼进提示词 → 模型总结（无需 API Key） |
| 🌤 实时天气 | wttr.in 查询全球城市实时天气 |
| 🕐 时间感知 | 今天几号、星期几、几点 |
| 🧮 数学引擎 | 算术题直接算（白名单校验，安全求值） |
| 🎭 情绪引擎 | "切换毒舌/开心/温柔/自然模式"，同一个大脑不同的味道 |
| 🖥 命令行聊天 | `python chat.py`，交互 + 管道两种模式 |
| 🌐 网页聊天 | `python webui.py`，零依赖起一个聊天网站 |

## 快速开始

```bash
# 1) 安装依赖（只需要 torch，CPU 就能跑）
pip install -r requirements.txt

# 2) 训练演示模型（CPU 约 2 分钟；仓库已附带训好的 checkpoints/demo，可跳过）
python scripts/train_demo.py

# 3a) 命令行聊天
python chat.py

# 3b) 网页聊天（浏览器打开 http://localhost:7860）
python webui.py
```

聊天里可以直接说：

```text
你好 / 你是谁 / 讲个笑话          ← 模型回答
现在几点了 / 今天几号             ← 时间工具
查北京天气 / 上海气温怎么样       ← 天气工具
23*7+15等于几 / (3+5)×2           ← 数学引擎
搜索 大语言模型是什么             ← 联网搜索后总结（附来源链接）
切换开心模式 / 毒舌 / 温柔        ← 情绪引擎
清空记忆 / 忘了我吧               ← 删除本会话记忆（唯一会失忆的方式）
```

## 网站调用 API

`python -m loganlm.api` 让任何程序都能调用小罗：

```bash
python -m loganlm.api --message "你好" --session web1
# → {"ok": true, "reply": "你好呀！我是小罗…", "tool": "model", "sources": []}

python -m loganlm.api --message "查上海天气" --session web1
python -m loganlm.api --reset --session web1     # 清空记忆
```

- 同一个 `--session` 共享同一份持久记忆（存于 `~/.loganlm/session_<id>.json`）
- 输出永远是单行 JSON，出错也返回 JSON，方便前端解析

## 网页版界面

`python webui.py` 启动后是一个完整的聊天网站：气泡对话、情绪切换按钮、
清空记忆按钮、搜索来源链接展示、思考中动画，手机浏览器也能用。

## 项目结构

```text
LoganLM/
├── chat.py                 # 命令行聊天入口
├── webui.py                # 网页聊天（零依赖 HTTP 服务）
├── loganlm/
│   ├── config.py           # 全局配置中心（LLM/Vision/多模态/训练/聊天）
│   ├── tokenizer/bpe.py    # 从0手写 Byte-level BPE（训练+编解码+特殊token）
│   ├── model/llm.py        # 从0手写 Transformer：RMSNorm/RoPE/GQA/SwiGLU/权重共享
│   ├── data/corpus.py      # 内置中文对话语料（130+ 组，4 种情绪增强）
│   ├── data/dataset.py     # 数据管线（对话模板/动态padding/loss遮罩）
│   ├── train/trainer.py    # 训练循环（AdamW+预热余弦退火+梯度裁剪）
│   ├── generate/sampler.py # 采样器（温度/top-k/top-p/重复惩罚）
│   ├── nlu/                # 意图识别 / 情绪引擎 / 数学引擎
│   ├── memory.py           # 持久记忆（JSON，手动清空才消失）
│   ├── tools/web.py        # 联网搜索 / 天气 / 时间（纯标准库）
│   ├── chat.py             # 聊天编排器（意图→工具→搜索→模型→记忆）
│   └── api.py              # JSON 命令行 API
├── scripts/train_demo.py   # 一键训练演示模型
├── scripts/bench.py        # 训练速度基准
├── tests/                  # 冒烟测试 + 单元测试
└── .github/workflows/      # CI（自动测试）+ 演示训练
```

## 自己训练 / 调参

```bash
python scripts/train_demo.py --steps 900              # 默认配置
python scripts/train_demo.py --dim 256 --layers 6     # 更大的模型
```

改语料就能教小罗新知识：编辑 `loganlm/data/corpus.py` 的 `DIALOGS`
（每条是 `(用户说, 小罗答)`），重新运行训练即可。

超参数都在 `loganlm/config.py` 一个文件里：词表、层数、头数、GQA 分组、
学习率、温度、top-k/top-p、重复惩罚、记忆轮数……

## 测试

```bash
python -m pytest tests/ -v
python tests/test_smoke.py     # 1 分钟冒烟：分词器→模型→生成→采样
```

## GitHub Actions

- **CI**（`ci.yml`）：每次 push / PR 自动安装依赖并跑全部测试
- **演示训练**（`train-demo.yml`）：手动触发，在云端从零训练一个新模型并上传为 Artifact

## 技术细节（为什么这样设计）

- **Byte-level BPE**：万物皆字节，任何文本都能编码，永远不会 `<unk>`；
  高频字词被合并成单 token，中文压缩率约 4-6 倍
- **GQA 分组查询注意力**：本仓库 6 个 Q 头共享 2 组 KV 头，推理更快、参数更省
- **RoPE 旋转位置编码**：把相对位置信息旋进 Q/K，外推性更好
- **只在回答部分计算 loss**：模型专心学"怎么回答"，不是学"怎么提问"
- **工具与模型分工**：时间/天气/算术这类"确定性事实"由规则与 API 直答，
  模型负责"像人一样说话"，搜索结果以提示词注入由模型总结
- **持久记忆**：会话历史存 JSON 文件，进程重启不丢，只有手动清空才删除

## License

MIT © 2026 罗根 (Logan)
