#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""点石成光 · PLQY 预测 Demo — Streamlit 入口

真正的页面在 `03_预测Demo页面/streamlit预测页面.py`。这里只做一次转发，
**不复制页面内容**——页面改了这里不用动，也就不会出现两份代码不一致。

为什么需要这个纯 ASCII 的入口：
Streamlit Community Cloud 创建应用时要填「主文件路径」。中文路径在部分环节会被
URL 编码搞坏，用一个 ASCII 入口最稳。

页面里有 `HERE = dirname(__file__)` → `BASE = 上一级`。
用 runpy 执行时 `__file__` 仍是真实页面路径，所以 BASE 会正确指向仓库根目录，
`02_PLQY预测模型/` 和 `09_可复跑源码/src/features.py` 都能按相对路径找到。
"""
import os
import runpy

ROOT = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(ROOT, "03_预测Demo页面", "streamlit预测页面.py")

if not os.path.exists(PAGE):
    import streamlit as st
    st.error(f"找不到页面文件：{PAGE}")
else:
    runpy.run_path(PAGE, run_name="__main__")
