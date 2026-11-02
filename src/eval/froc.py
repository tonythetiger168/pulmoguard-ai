"""
froc.py — LUNA16 FROC / CPM 評估（依官方規則）

規則：
  - 偵測中心落在結節半徑內（歐氏距離 ≤ diameter/2）即命中
  - 同一結節多個命中只取最高分為 TP，其餘忽略（不算 FP）
  - 命中 annotations_excluded（非結節發現）者忽略
  - CPM = 在每例 1/8, 1/4, 1/2, 1, 2, 4, 8 個 FP 時敏感度的平均
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FP_RATES = (0.125, 0.25, 0.5, 1, 2, 4, 8)


def froc(detections: pd.DataFrame, annotations: pd.DataFrame, n_scans: int,
         excluded: pd.DataFrame | None = None,
         fp_rates: tuple = FP_RATES) -> dict:
    """detections: seriesuid, coordX, coordY, coordZ, probability
    annotations: seriesuid, coordX, coordY, coordZ, diameter_mm（僅限受測 series）
    回傳 {"sensitivity": {fp: sens}, "cpm": float, "n_nodules": int}
    """
    ann = annotations.reset_index(drop=True)
    n_nodules = len(ann)
    best = np.full(n_nodules, -np.inf)       # 每顆結節最高命中分數
    fp_scores = []

    for uid, dets in detections.groupby("seriesuid"):
        a = ann[ann.seriesuid == uid]
        ex = excluded[excluded.seriesuid == uid] if excluded is not None else None
        for d in dets.itertuples():
            p = np.array([d.coordX, d.coordY, d.coordZ])
            hit = False
            if len(a):
                dist = np.linalg.norm(a[["coordX", "coordY", "coordZ"]].to_numpy() - p, axis=1)
                inside = dist <= a.diameter_mm.to_numpy() / 2
                for i in a.index[inside]:
                    best[i] = max(best[i], d.probability)
                    hit = True
            if hit:
                continue
            if ex is not None and len(ex):
                dist = np.linalg.norm(ex[["coordX", "coordY", "coordZ"]].to_numpy() - p, axis=1)
                r = np.where(ex.diameter_mm.to_numpy() > 0, ex.diameter_mm.to_numpy() / 2, 0)
                if np.any(dist <= r):
                    continue
            fp_scores.append(d.probability)

    fp_scores = np.sort(np.asarray(fp_scores))[::-1]
    sens = {}
    for rate in fp_rates:
        k = int(np.floor(rate * n_scans))   # 允許的 FP 數
        if k < len(fp_scores):
            # 閾值須嚴格高於第 k+1 個 FP 的分數
            thr = fp_scores[k]
            tp = np.sum(best > thr)
        else:
            tp = np.sum(np.isfinite(best))
        sens[rate] = float(tp / n_nodules) if n_nodules else float("nan")
    return {"sensitivity": sens, "cpm": float(np.mean(list(sens.values()))), "n_nodules": n_nodules}
