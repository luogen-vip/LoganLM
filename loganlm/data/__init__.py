# -*- coding: utf-8 -*-
from .corpus import DIALOGS, SYSTEM_PROMPT
from .dataset import ChatDataset, render_example, render_prompt

__all__ = ["DIALOGS", "SYSTEM_PROMPT", "ChatDataset", "render_example", "render_prompt"]
