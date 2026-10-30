# -*- coding: utf-8 -*-
"""
LoganLM —— 从 0 手写的多模态 AI 语言模型
==========================================

作者：罗根（Logan）
协议：MIT

本包完全从零实现，不使用任何现成模型 / 预训练权重 / 第三方分词库：

    loganlm.tokenizer  从0手写的 Byte-level BPE 分词器
    loganlm.model      从0手写的 LLaMA 风格 Transformer（RMSNorm/RoPE/GQA/SwiGLU）
    loganlm.vision     从0手写的 ViT 视觉编码器 + 多模态融合
    loganlm.data       内置中文语料 + 数据管线
    loganlm.train      训练器
    loganlm.generate   采样器（温度/top-k/top-p/重复惩罚）
    loganlm.tools      联网搜索 / 天气 / 时间
    loganlm.nlu        意图识别 / 情绪引擎 / 数学引擎
"""

__version__ = "0.1.0"
__author__ = "罗根 (Logan)"
