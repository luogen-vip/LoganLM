# LoganLM 架构详解（作者：罗根）

本文面向想读懂每一行代码的人，按数据流顺序拆解 LoganLM 的全部模块与设计决策。

## 1. 总体数据流

```
用户输入(文本+可选图片)
    │
    ▼
[chat.py LoganChat.reply]
    ├─ 1. 情绪指令检测 (nlu/emotions.detect_emotion_request)
    ├─ 2. 意图识别     (nlu/intent.detect_intent)
    ├─ 3. 路由到处理器：
    │      math   → nlu/math_engine.safe_eval        (AST 白名单求值)
    │      time   → tools/weather.now_text/city_time (本地时钟+时区表)
    │      weather→ tools/weather.get_weather        (wttr.in 实时)
    │      search → tools/websearch.search_and_summarize (搜索+抓取+总结)
    │      memory → memory.ConversationMemory.summary
    │      vision → vision_analyze + model/vision    (经典CV+ViT)
    │      chat   → 语言模型生成
    ├─ 4. 情绪风格化 (nlu/emotions.apply_emotion)
    └─ 5. 写入持久记忆 → 返回 {response, intent, sources, emotion, ...}
```

## 2. 分词器（tokenizer/bpe.py）

**为什么 Byte-level BPE**：任何 UTF-8 字符串都能编码（兜底是 256 个字节 token），
中文单字天然是一个"预切分片段"，英文按词切分，永不产生 UNK。

训练算法（GPT-2 同源思想，代码重写）：

1. 正则预切分：`'(?:[sdmt]|ll|ve|re)`（英文缩写）、` ?[A-Za-z]+`（英文词）、
   ` ?[0-9]+`（数字）、`[\u2e80-\uffef]`（**单个** CJK 字符）……
2. 片段 → UTF-8 字节 id 序列（0~255）。
3. 反复统计相邻 token 对频率，合并最高频对，新 id = `256 + 特殊token数 + 步数`。
4. 词表布局：`[0..255 字节][256.. 特殊 token][...] [BPE 合并 token]`。

编码：预切分 → 字节 → 按 `merge_ranks` 贪心合并（rank 越小越先合并），
带 LRU 缓存。解码：id → 字节 → `utf-8 errors="replace"`，永不抛异常。

特殊 token：`<|pad|>=256 <|bos|>=257 <|eos|>=258 <|img|>=259 <|sep|>=260`。
对话模板：`<|bos|> 用户输入 <|sep|> 回复 <|eos|>`，训练时 prompt 部分标签置 -100。

## 3. 语言模型（model/layers.py, transformer.py）

LLaMA 同构、代码全重写：

| 组件 | 实现 | 备注 |
| --- | --- | --- |
| RMSNorm | `x*rsqrt(mean(x²)+eps)*w` | 替代 LayerNorm，省均值项 |
| RoPE | rotate_half 风格，预计算 cos/sin 表 | 增量解码按 `past_len` 切片 |
| 注意力 | 手写 QKV 投影 + 缩放点积 + 因果掩码 | 支持 GQA（`n_kv_heads < n_heads` 时 Q 头共享 KV 头） |
| KV Cache | 预分配 `[b, n_kv, max_seq_len, d]` 缓冲区 | 每次 forward 把新 k/v 写入 `[past_len:past_len+s]`，注意力拼接 `cache[:past_len]` |
| FFN | SwiGLU：`W2(silu(W1x)⊙W3x)` | 门控前馈 |
| 结构 | Pre-Norm 残差 × n_layers | 深层稳定 |
| 权重 | 输入 embedding 与 lm_head 共享（可选） | 小模型省参数 |

**KV Cache 正确性**有单元测试保证：整段前向 vs 两段增量前向，logits 最大误差 < 1e-3
（实测 ~2e-7）。**因果掩码正确性**也有测试：截断后缀不改变前缀位置的输出。

多模态支持：`forward(inputs_embeds=...)` 允许直接喂 embedding（图像软 token 就是从这里进的）。

## 4. 视觉通路（model/vision.py + multimodal.py）

- **PatchEmbed**：`Conv2d(k=16, s=16)` 一步完成切块+线性映射 → 64 个 patch token。
- **编码器**：可学习位置编码 + 6 层 Pre-Norm Transformer（双向注意力，GELU MLP）。
- **三个分类头**（颜色 8 / 形状 6 / 位置 9，作用于平均池化特征）：
  预训练数据由 `data/synthetic.py` **程序合成**——随机背景上画随机颜色随机位置的
  形状（PIL 绘制 + 4× 超采样抗锯齿），数据无限且零下载。
- **Perceiver 重采样器**：8 个可学习查询 token 对 patch 特征做交叉注意力，
  把 64 个 patch 压成 8 个图像软 token（长度与图片分辨率无关）。
- **投影 + 拼接**：软 token 经 MLP 投影到 LLM 空间，文本里的每个 `<|img|>` 占位符
  被展开为一整张图的 8 个 token，图像位置标签置 -100（不参与语言损失）。

## 5. 训练（train/trainer.py）

- AdamW（weight_decay=0.1）+ 线性 warmup + 余弦衰减（min_lr_ratio=0.1）。
- 梯度裁剪 1.0；损失只统计回复部分（`ignore_index=-100`）。
- 检查点 `torch.save({"model": state_dict, "meta": {...}})`，加载时 `strict=False`
  以兼容结构微调。
- 视觉训练：三分类交叉熵相加，held-out 合成集验证准确率。

## 6. 生成（generate/sampler.py）

策略链：`repetition_penalty → top-k → top-p → temperature 采样`（temperature=0 时贪心）。
KV Cache 全程增量：prompt 预填充一次，之后每步只算 1 个 token。

## 7. 联网搜索（tools/websearch.py）

1. `POST https://html.duckduckgo.com/html/`（备用 `lite.duckduckgo.com/lite/`）；
2. 正则解析 `result__a` 标题链接 + `result__snippet` 摘要；DDG 跳转链接解 `uddg` 参数；
3. 抓正文：200KB 截断、编码降级（resp.encoding → utf-8 → gb18030）、去 script/style/nav、
   取 `<title>` + `<p>` 段落、NFKC 规范化；
4. 抽取式总结：查询词覆盖度 × `len^0.25` 位置加权，Top 句按原顺序拼接。
5. 无第三方 NLP 依赖，全失败时返回空结果由上层优雅降级。

## 8. 记忆（memory.py）

- 目录 `~/.loganlm/memory_<session>.json`，每轮立即落盘（先写临时文件再 `os.replace`，防损坏）。
- `recent(n)` 供 LLM 上下文（token 预算内从旧到新截断）；`summary()` 抽取式总结全部历史。
- 只有 `/clear` 清空；切换 `/session` 各自独立保留。

## 9. 情绪引擎（nlu/emotions.py）

7 种风格（自然/开心/温柔/傲娇/严肃/撒娇/冷静），每种定义前缀池、后缀池、
颜文字池、文本替换规则、统一结尾标点。指令如"用傲娇的语气回答"会**粘滞切换**全局风格，
之后所有回答（含工具类回答）都被风格化——信息内容不变，表达方式改变。

## 10. 安全边界

- 数学引擎：`ast` 白名单，只允许数字/四则/乘方/取模/括号/一元正负，pow 限幅；
  `__import__`、名称、属性一概拒绝。
- Web 服务：纯标准库 `ThreadingHTTPServer`，仅本地回环默认监听，POST 体限 32MB。
- 分词器/模型：无任何 `eval`/`exec`。

## 11. 如何扩展

| 目标 | 改哪里 |
| --- | --- |
| 更大的模型 | `config.py` 的 `LLMConfig`（dim/n_layers/heads/vocab） |
| 更强的语言能力 | `data/corpus.py` 换成海量语料；先下一个词预训练再 SFT |
| 真实图像理解 | `model/vision.py` 换大数据集（ImageNet/COCO）预训练，桥接代码不变 |
| 更好的搜索 | `tools/websearch.py` 接入你喜欢的搜索 API（函数签名已抽象） |
| GPU 训练 | `--device cuda`；DataLoader 已就绪，可再上 DDP |
