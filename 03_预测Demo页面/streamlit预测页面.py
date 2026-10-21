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
    try:
        metrics = json.load(open(os.path.join(MODEL_DIR, "模型指标报告_分组交叉验证与基线.json"), encoding="utf-8"))
    except FileNotFoundError:
        metrics = {}
    return model, schema, metrics


model, schema, metrics = load_artifacts()
CATS = schema["categories"]
RANGES = schema["train_ranges"]
Q = schema["q_hat"]
M = metrics.get("holdout", {})

st.title("钙钛矿量子点 PLQY 预测")
st.caption("点石成光 · 数据与模型线 v0 ｜ 输入配方与工艺参数，预测荧光量子产率")

with st.sidebar:
    st.header("模型说明")
    st.markdown(
        f"""
- 训练样本：**{schema['n_train']} 条** / **{schema['n_papers']} 篇文献**
- 测试集 MAE：**{M.get('MAE', 'NA')} 个百分点**
- 测试集 R²：**{M.get('R2', 'NA')}**
- 90% 区间半宽：**±{Q:.1f} 个百分点**

模型只使用**实验前可设定的工艺参数**，
不使用发射波长、稳定性等结果量。
"""
    )
    st.warning(
        "训练文献仅 "
        f"{schema['n_papers']} 篇，模型目前只能判断趋势，"
        "不能替代实验。配方落在训练范围外时，预测可信度显著下降。"
    )

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
    f"数据来源：公开文献（每条可溯源 DOI，已 Crossref 核验），"
    f"当前版本 {schema['n_train']} 条 / {schema['n_papers']} 篇。"
    "本页面为研发辅助工具，输出不构成实验保证。"
)
