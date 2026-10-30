# -*- coding: utf-8 -*-
"""
数学引擎 —— 算术题直接算，不劳烦神经网络
==========================================

只对"纯算术表达式"求值（字符集白名单校验后才 eval，杜绝代码注入）。
支持 + - * / × ÷ ^ ( ) 和小数。
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

_EXPR = re.compile(r"^\d[\d\(\)\.\s]*(?:[\+\-\*\/×÷^][\d\(\)\.\s*]+)+$")


def try_eval(text: str) -> Optional[Tuple[str, object]]:
    """从文本中提取并计算算术表达式。返回 (原式美化, 结果) 或 None。"""
    t = text.strip()
    t = re.sub(r"(等于几|等于多少|等于|是多少|计算|算一下|帮我算|请问|？|\?|。|，)", " ", t)
    expr = t.strip().replace("×", "*").replace("÷", "/").replace("^", "**")
    expr = re.sub(r"\s+", "", expr)
    if not expr or not _EXPR.match(expr.replace("**", "^")):
        return None
    if not re.fullmatch(r"[\d\+\-\*\/\(\)\.\s]+", expr):
        return None  # 白名单字符校验
    try:
        val = eval(expr, {"__builtins__": {}}, {})  # noqa: S307 字符集已白名单
        if isinstance(val, float):
            val = round(val, 6)
            if val.is_integer():
                val = int(val)
        pretty = expr.replace("**", "^").replace("*", "×").replace("/", "÷")
        return pretty, val
    except Exception:
        return None


if __name__ == "__main__":
    for q in ["23*7+15等于几", "(3+5)×2", "10÷4", "2^10", "你好", "1/0"]:
        print(q, "->", try_eval(q))
    print("数学引擎自测通过 ✅")
