#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""特征工程：训练与推理共用同一份逻辑，避免线上线下不一致。

铁律：只允许「实验前可以设定的量」进特征。
波长、稳定性、PLQY 测量方式是结果或测量口径，一律不进特征（否则就是用答案预测答案）。
"""
import re

CAT_FEATURES = ["system", "method_family", "solvent_family", "ligand_family"]
NUM_FEATURES = ["temp_C", "time_min", "ligand_amount", "precursor_ratio"]
FEATURES = CAT_FEATURES + NUM_FEATURES

# 明确排除，写在这里是为了让评审一眼看到我们考虑过
EXCLUDED = ["emission_nm", "stability_desc", "stability_retention_pct", "plqy_method"]


def method_family(text):
    t = (text or "").lower()
    if "microfluidic" in t:
        return "microfluidic"
    if "hot injection" in t or "hot-injection" in t or "hot injection" in t.replace("-", " "):
        return "hot_injection"
    if "larp" in t or "reprecipitat" in t:
        return "LARP"
    if "exchange" in t:
        return "post_exchange"
    if "mill" in t or "grind" in t or "solid" in t:
        return "solid_state"
    if "solvothermal" in t or "hydrothermal" in t:
        return "solvothermal"
    if "not_reported" in t or not t.strip():
        return "unknown"
    return "other"


def solvent_family(text):
    t = (text or "").lower()
    if "not_reported" in t or not t.strip():
        return "unknown"
    fams = []
    if "ode" in t or "octadecene" in t:
        fams.append("ODE")
    if "toluene" in t:
        fams.append("toluene")
    if "hexane" in t:
        fams.append("hexane")
    if "limonene" in t:
        fams.append("limonene")
    if "dmf" in t:
        fams.append("DMF")
    if "chloroform" in t or "chcl3" in t:
        fams.append("chloroform")
    return "+".join(fams) if fams else "other"


def ligand_family(text):
    t = (text or "").lower()
    if "not_reported" in t or not t.strip():
        return "unknown"
    fams = []
    if re.search(r"\boa\b|oleic acid", t):
        fams.append("OA")
    if re.search(r"\boam\b|oleylamine", t):
        fams.append("OAm")
    if re.search(r"\bola\b|oleylamine", t):
        fams.append("OLA")
    if "ddab" in t or "didodecyl" in t:
        fams.append("DDAB")
    if "top" in t or "trioctylphosphine" in t:
        fams.append("TOP")
    return "+".join(sorted(set(fams))) if fams else "other"


def to_float(x):
    try:
        v = float(x)
        return v
    except (TypeError, ValueError):
        return None


def build_frame(rows):
    """rows: list[dict]（csv.DictReader 输出）。返回 (X_dict, y_list, groups, kept_rows)"""
    import pandas as pd

    X, y, groups, kept = [], [], [], []
    for r in rows:
        if (r.get("status") or "").strip() != "OK":
            continue
        val = to_float(r.get("plqy_pct"))
        if val is None or not (0 < val <= 100):
            continue
        X.append({
            "system": (r.get("system") or "other").strip() or "other",
            "method_family": method_family(r.get("method")),
            "solvent_family": solvent_family(r.get("solvent")),
            "ligand_family": ligand_family(r.get("ligand_type")),
            "temp_C": to_float(r.get("temp_C")),
            "time_min": to_float(r.get("time_min")),
            "ligand_amount": to_float(r.get("ligand_amount")),
            "precursor_ratio": to_float(r.get("precursor_ratio")),
        })
        y.append(val)
        groups.append(r.get("doi") or "NO_DOI")
        kept.append(r)

    df = pd.DataFrame(X, columns=FEATURES)
    return df, y, groups, kept


def prepare_for_lightgbm(df):
    """类别列转 category 并固定取值域，保证推理时列顺序、类别集合一致。"""
    import pandas as pd
    out = df.copy()
    for c in CAT_FEATURES:
        out[c] = out[c].astype("category")
    for c in NUM_FEATURES:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out[FEATURES]
