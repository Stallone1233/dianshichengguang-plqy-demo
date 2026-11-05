#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""点石成光 · PLQY 预测 Demo（Streamlit）

设计原则：
- 模型只做推理，不重新训练
- 必须显示 90% 区间与训练数据覆盖度，不能只给一个好看的数字
- 超范围输入要给黄色警告，而不是装作很有把握
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)                      # 交付包/
MODEL_DIR = os.path.join(BASE, "02_PLQY预测模型")
sys.path.insert(0, os.path.join(BASE, "09_可复跑源码", "src"))

st.set_page_config(page_title="点石成光 · 钙钛矿量子点 PLQY 预测", page_icon="💡", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "模型文件_PLQY预测_lgbm.joblib"))
    schema = json.load(open(os.path.join(MODEL_DIR, "推理配置_特征域与预测区间.json"), encoding="utf-8"))
    return model, schema


model, schema = load_artifacts()
CATS = schema["categories"]
RANGES = schema["train_ranges"]
Q = schema["q_hat"]

# 对外展示的三个数字：全库条数 / 来源文献数 / 可溯源比例。
# 注意这跟 n_train（177）不是一回事：213 是全部记录，177 是能进训练的。
# 两个数都要能自洽，页面上分开说，别混。
N_RECORDS = schema.get("n_records", 213)
N_PAPERS = schema["n_papers"]
TRACE_PCT = schema.get("traceable_pct", 90.1)

st.title("AI 帮小团队少试错")
st.caption("点石成光 · 钙钛矿量子点研发辅助工具｜输入配方与工艺参数，预测荧光量子产率")

k1, k2, k3 = st.columns(3)
k1.metric("文献记录", f"{N_RECORDS} 条")
k2.metric("来源文献", f"{N_PAPERS} 篇")
k3.metric("可溯源 DOI", f"{TRACE_PCT:.1f}%")

st.markdown(
    "把散落在论文里的「配方—工艺—性能」数据结构化，"
    "让小团队先算一遍，再决定做哪几组实验。"
)

# 数据来源清单：平时折叠着不吵人，录屏和答辩时展开，让评委自己看 DOI。
# 清单来自 出处表，缺文件时整块隐藏，不影响预测功能。
SRC_JSON = os.path.join(MODEL_DIR, "数据来源_12篇.json")
if os.path.exists(SRC_JSON):
    _src = json.load(open(SRC_JSON, encoding="utf-8"))
    with st.expander(f"数据来源：{len(_src)} 篇文献，{TRACE_PCT:.1f}% 可回溯到 DOI"):
        st.caption(
            "每条记录都保留原文表格位置（表号 + 行号）。点最后一列的 DOI 可直接打开原文。"
        )
        st.dataframe(
            pd.DataFrame([{
                "编号": r["编号"],
                "期刊": r["期刊"],
                "年份": r["年份"],
                "记录数": r["记录数"],
                "原文（点开）": r["链接"],
            } for r in _src]),
            hide_index=True,
            use_container_width=True,
            column_config={
                "原文（点开）": st.column_config.LinkColumn(
                    "原文（点开）", display_text=r"https://doi\.org/(.*)"),
            },
        )

with st.sidebar:
    st.header("关于这个工具")
    st.markdown(
        f"""
- 数据基础：**{N_RECORDS} 条记录 / {N_PAPERS} 篇文献**
- 其中 **{schema['n_train']} 条**为单值 PLQY，用于建模
- **{TRACE_PCT:.1f}%** 的记录可回溯到原始文献 DOI
- 只用**实验前可设定的工艺参数**，不含波长、稳定性等结果量
"""
    )
    st.warning("预测仅供参考，以实验验证为准。")

c1, c2 = st.columns(2)
with c1:
    system = st.selectbox("量子点体系", CATS["system"],
                          index=CATS["system"].index("CsPbBr3") if "CsPbBr3" in CATS["system"] else 0)
    solvent = st.selectbox("溶剂", CATS["solvent_family"])
    ligand = st.selectbox("配体", CATS["ligand_family"])
    method = st.selectbox("合成方法", CATS["method_family"])
with c2:
    temp = st.slider("合成温度 (℃)", 20, 400, 170, 5)
    time_min = st.slider("反应时间 (min)", 0.0, 180.0, 10.0, 0.5)
    lig_amt = st.number_input("配体用量（可选）", min_value=0.0, value=0.0, step=0.1,
                              help="单位为 mmol / mol% / mL·min⁻¹，按原文口径；不确定就留 0")
    ratio = st.number_input("前驱体摩尔比 Pb:Cs（可选）", min_value=0.0, value=0.0, step=0.1,
                            help="1.0 表示 PbX₂:CsX = 1:1；数据集里这一列几乎全缺，留 0 表示未知")

if st.button("预测 PLQY", type="primary", use_container_width=True):
    row = {
        "system": system,
        "method_family": method,
        "solvent_family": solvent,
        "ligand_family": ligand,
        "temp_C": float(temp),
        "time_min": float(time_min),
        "ligand_amount": float(lig_amt) if lig_amt > 0 else np.nan,
        "precursor_ratio": float(ratio) if ratio > 0 else np.nan,
    }
    X = pd.DataFrame([row])[schema["features"]]
    for c in schema["cat"]:
        X[c] = pd.Categorical(X[c], categories=CATS[c])
    for c in schema["num"]:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    pred = float(np.clip(model.predict(X)[0], 0, 100))
    lo, hi = max(0.0, pred - Q), min(100.0, pred + Q)

    st.metric("预测 PLQY", f"{pred:.1f} %", delta=f"90% 区间 {lo:.0f}–{hi:.0f}%")

    # 覆盖度检查
    ood = []
    for c in schema["num"]:
        if c in RANGES and not pd.isna(X[c].iloc[0]):
            v = float(X[c].iloc[0])
            lo_t, hi_t = RANGES[c]
            if v < lo_t or v > hi_t:
                ood.append(f"{c}={v:g} 超出训练范围 [{lo_t:g}, {hi_t:g}]")
    if ood:
        st.warning("该配方超出训练数据覆盖范围，预测仅供参考：\n\n- " + "\n- ".join(ood))
    else:
        st.success("该配方落在训练数据覆盖范围内。")

    # 单样本贡献（LightGBM 内置 TreeSHAP，无需额外依赖）
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 图内文字一律用英文。原因：matplotlib 不像浏览器会自动回退字体，
        # Linux 服务器上没有中文字体，中文会全渲染成方块（□）。本机 Windows 装了
        # 微软雅黑所以看不出问题，一部署就露馅。特征名本来就是英文，用英文标注更统一。
        # 下面这行保留，万一部署环境恰好有中文字体也不冲突。
        plt.rcParams["font.sans-serif"] = [
            "Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Zen Hei",
            "Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False      # 负号也要，否则 −10 变方块

        contrib = model.predict(X, pred_contrib=True)[0]
        sv, base = contrib[:-1], contrib[-1]

        def _fmt(v):
            return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else v

        names = [f"{c}={_fmt(X.iloc[0][c])}" for c in schema["features"]]
        order = np.argsort(sv)
        fig, ax = plt.subplots(figsize=(7, 3.4))
        colors = ["#c0392b" if v > 0 else "#2c6fbb" for v in sv[order]]
        ax.barh([names[i] for i in order], sv[order], color=colors)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("Contribution to this prediction (PLQY percentage points)")
        ax.set_title(f"Per-sample explanation: base {base:.1f}%  ->  predicted {pred:.1f}%")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        st.caption("红色=拉高预测，蓝色=拉低预测；n/a 表示该特征本行未填。由 LightGBM 内置 TreeSHAP 计算。")
    except Exception as e:
        st.info(f"特征贡献图暂不可用：{type(e).__name__}: {e}")

st.divider()
st.caption(
    f"数据来源：公开文献，逐条保留原文表格位置；{TRACE_PCT:.1f}% 的记录可回溯到原始 DOI。"
    "本页面为研发辅助工具，预测仅供参考，以实验验证为准。"
)
