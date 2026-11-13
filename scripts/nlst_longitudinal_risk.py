"""
nlst_longitudinal_risk.py — 用「本次 + 前次」掃描預測 1–6 年內確診肺癌（CPU 即可，輸入是已抽好的特徵）

單位 = 一位病人的一次篩檢（exam 0/1/2）。標籤 event_{h}y 來自 cohort.csv（−1 = 追蹤不足，該年限排除）。
比較三個模型（同一組病人層級 5-fold，同一位病人的所有檢查在同一 fold）：
  single  ：只看本次掃描 —— 結節惡性度最大值（等於現在的 LUNA25 流程）
  growth  ：Lung-RADS 式規則 —— 本次惡性度最大值 + 是否有結節符合生長 / VDT < 400 天
  longit  ：HistGradientBoosting，特徵 = 本次前 3 顆結節（惡性度、大小、體積）+ 與前次比較（體積變化、VDT、
            是否新發、是否生長）+ 前次的惡性度最大值 + 年齡、性別、吸菸狀態
指標：每個年限的 AUC（bootstrap 95% CI），以及 longit 對 single 的配對 AUC 差。
LUNA25 用過的病人（in_luna25）不進評估，避免惡性度模型看過答案。

用法：
    python -m scripts.nlst_longitudinal_risk --tracks runs/tracks/tracks.csv --cohort runs/nlst_long/cohort.csv \
        --out runs/risk
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

HORIZONS = (1, 2, 3, 4, 5, 6)
TOP_N = 3


def exam_features(tracks: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    t = tracks.copy()
    t["log_vol"] = np.log1p(t.volume.clip(lower=0))
    t["vol_change"] = t.volume_change_pct.clip(-100, 500)
    t["inv_vdt"] = (1.0 / t.vdt_days.replace(0, np.nan)).clip(-0.05, 0.05)  # 長越快越大；縮小 = 0（inf）
    t["inv_vdt"] = t.inv_vdt.fillna(0.0)
    t["fast_vdt"] = (t.vdt_days < 400).astype(float)
    for c in ("is_new", "lungrads_growth", "comparable"):
        t[c] = t[c].astype(str).str.lower().isin(["true", "1"]).astype(float)
    rows = []
    for (pid, ex), g in t.groupby(["PatientID", "exam"]):
        g = g.sort_values("malignancy", ascending=False)
        r = {"PatientID": pid, "exam": int(ex), "n_nodules": len(g),
             "max_mal": g.malignancy.max(), "max_diam": g.mean_d.max(), "max_logvol": g.log_vol.max(),
             "any_growth": g.lungrads_growth.max(), "any_fast_vdt": g.fast_vdt.max(),
             "max_inv_vdt": g.inv_vdt.max(), "n_new": g.is_new.sum() * g.comparable.max(),
             "comparable": g.comparable.max()}
        for k in range(TOP_N):
            if k < len(g):
                n = g.iloc[k]
                r.update({f"mal{k}": n.malignancy, f"diam{k}": n.mean_d, f"logvol{k}": n.log_vol,
                          f"volchg{k}": n.vol_change if n.comparable else np.nan,
                          f"invvdt{k}": n.inv_vdt if n.comparable else np.nan,
                          f"new{k}": n.is_new if n.comparable else np.nan, f"grow{k}": n.lungrads_growth})
        rows.append(r)
    f = pd.DataFrame(rows)
    # 前一次檢查的惡性度最大值（沒有前次 = NaN）
    f = f.sort_values(["PatientID", "exam"])
    f["prior_max_mal"] = f.groupby("PatientID").max_mal.shift(1)
    f["mal_change"] = f.max_mal - f.prior_max_mal
    c = cohort.copy()
    c["PatientID"] = c.PatientID.astype(str)
    keep = ["PatientID", "exam", "age", "gender", "cigsmok", "in_luna25"] + [f"event_{h}y" for h in HORIZONS]
    m = c[[k for k in keep if k in c.columns]].merge(f, on=["PatientID", "exam"], how="left")
    m["n_nodules"] = m.n_nodules.fillna(0)
    m["max_mal"] = m.max_mal.fillna(0.0)    # 沒有候選結節 = 惡性度 0
    for k in ("any_growth", "any_fast_vdt"):
        m[k] = m[k].fillna(0.0)
    return m


def boot_ci(y, p, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    a = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            a.append(roc_auc_score(y[i], p[i]))
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if a else [None, None]


def paired_delta(y, p1, p0, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            d.append(roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p0[i]))
    return float(np.mean(d)), [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


FEATS_EXCLUDE = {"PatientID", "in_luna25"} | {f"event_{h}y" for h in HORIZONS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--include_luna25", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tracks = pd.read_csv(a.tracks, dtype={"PatientID": str})
    cohort = pd.read_csv(a.cohort, dtype={"PatientID": str})
    df = exam_features(tracks, cohort)
    if not a.include_luna25 and "in_luna25" in df:
        df = df[~df.in_luna25.astype(str).str.lower().isin(["true", "1"])]
    feats = [c for c in df.columns if c not in FEATS_EXCLUDE]
    df.to_csv(out / "exam_features.csv", index=False)

    res, preds = {}, []
    for h in HORIZONS:
        d = df[df[f"event_{h}y"] >= 0].reset_index(drop=True)
        y = d[f"event_{h}y"].to_numpy().astype(int)
        if y.sum() < 10 or (1 - y).sum() < 10:
            res[f"{h}y"] = {"skipped": f"too few events ({int(y.sum())})"}
            continue
        p_long = np.zeros(len(d))
        for tr, te in GroupKFold(a.folds).split(d, y, d.PatientID):
            clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                                 l2_regularization=1.0, random_state=0)
            clf.fit(d.loc[tr, feats], y[tr])
            p_long[te] = clf.predict_proba(d.loc[te, feats])[:, 1]
        p_single = d.max_mal.to_numpy()
        p_growth = p_single + d.any_growth.to_numpy() + d.any_fast_vdt.to_numpy()
        follow = d.exam.to_numpy() > 0
        r = {"n": int(len(d)), "events": int(y.sum())}
        for name, p in (("single", p_single), ("growth", p_growth), ("longit", p_long)):
            r[name] = {"auc": float(roc_auc_score(y, p)), "ci": boot_ci(y, p)}
            if len(np.unique(y[follow])) == 2:  # 只看有前次可比的追蹤檢查（T1、T2）
                r[name]["auc_followup_exams"] = float(roc_auc_score(y[follow], p[follow]))
        r["longit_minus_single"] = paired_delta(y, p_long, p_single)
        res[f"{h}y"] = r
        preds.append(d[["PatientID", "exam"]].assign(horizon=h, y=y, single=p_single, growth=p_growth, longit=p_long))
        print(h, json.dumps({k: (v["auc"] if isinstance(v, dict) and "auc" in v else v) for k, v in r.items()}), flush=True)
    if preds:
        pd.concat(preds).to_csv(out / "oof_predictions.csv", index=False)
    summary = {"unit": "patient-exam (NLST T0/T1/T2)", "excluded_luna25_patients": not a.include_luna25,
               "patients": int(df.PatientID.nunique()), "exams": int(len(df)), "features": feats, "results": res}
    json.dump(summary, open(out / "summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
