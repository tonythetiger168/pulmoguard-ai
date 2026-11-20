"""
luna25_ensemble.py — 把多次 LUNA25 訓練（不同骨幹）的 out-of-fold 預測做排名平均集成，報 AUC 與配對 bootstrap 差異

各次訓練必須用同一個 folds.csv（同 seed、同資料 → make_folds 結果相同），程式會檢查。
用法：
  python -m scripts.luna25_ensemble --runs results/luna25_20261103_0849:prob \\
      results/luna25_20261103_1724/medicalnet18:prob_tta --out runs/luna25_ensemble.json
每個 --runs 項目是「資料夾:機率欄位」，第一個當作對照基準。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


def main(argv=None) -> dict:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--n_boot", type=int, default=1000)
    a = p.parse_args(argv)

    merged, folds, cols = None, None, []
    for i, spec in enumerate(a.runs):
        d, col = spec.rsplit(":", 1) if ":" in spec else (spec, "prob")
        oof = pd.read_csv(Path(d) / "oof_predictions.csv", dtype={"AnnotationID": str})
        f = oof.set_index("AnnotationID").fold
        if folds is not None and not f.reindex(folds.index).eq(folds).all():
            raise ValueError(f"{d}: folds differ from {a.runs[0]}")
        folds = f if folds is None else folds
        name = f"m{i}"
        part = oof[["AnnotationID", "label", col]].rename(columns={col: name})
        merged = part if merged is None else merged.merge(part.drop(columns="label"), on="AnnotationID")
        cols.append(name)
    y = merged.label.to_numpy()
    ranks = np.mean([rankdata(merged[c]) / len(merged) for c in cols], axis=0)
    base = merged[cols[0]].to_numpy()

    rng = np.random.default_rng(0)
    ens_b, diff_b = [], []
    for _ in range(a.n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        e, b = roc_auc_score(y[i], ranks[i]), roc_auc_score(y[i], base[i])
        ens_b.append(e)
        diff_b.append(e - b)
    res = {
        "runs": a.runs, "n": int(len(y)), "n_pos": int(y.sum()),
        "per_run_auc": {s: float(roc_auc_score(y, merged[c])) for s, c in zip(a.runs, cols)},
        "ensemble_auc": float(roc_auc_score(y, ranks)),
        "ensemble_auc_95ci": [float(np.percentile(ens_b, 2.5)), float(np.percentile(ens_b, 97.5))],
        "gain_vs_first": float(np.median(diff_b)),
        "gain_vs_first_95ci": [float(np.percentile(diff_b, 2.5)), float(np.percentile(diff_b, 97.5))],
    }
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main()
