"""
nodule_growth.py — 縱向（T0/T1/T2）結節配對、體積、體積倍增時間（VDT）與 Lung-RADS 生長判定

全部是純函式，不依賴 GPU：
  - mask_volume_mm3 / mask_diameters_mm：由二值分割遮罩算體積與長短徑
  - vdt_days：VDT = Δt × ln2 / ln(V2 / V1)，體積沒長（V2 ≤ V1）回傳 inf
  - lungrads_growth：Lung-RADS v2022 的「生長」= 12 個月內平均徑增加 > 1.5 mm
  - match_nodules：把前一次掃描的結節（先經對位轉換到本次座標）與本次結節做一對一配對（匈牙利法 + 距離門檻）
  - build_tracks：把每次掃描的配對串成跨年的結節軌跡，並輸出給 Lung-RADS 報告產生器的逐列表格
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

LUNGRADS_GROWTH_MM = 1.5


def mask_volume_mm3(mask: np.ndarray, spacing_mm: Sequence[float] = (1.0, 1.0, 1.0)) -> float:
    return float(np.count_nonzero(mask) * np.prod(spacing_mm))


def mask_diameters_mm(mask: np.ndarray, spacing_mm: Sequence[float] = (1.0, 1.0, 1.0)) -> tuple[float, float, float]:
    """回傳 (長徑, 短徑, 平均徑)，以面積最大的軸向切面量測（與 Lung-RADS 手量習慣一致）。

    mask 軸序為 (z, y, x)；spacing_mm 也是 (z, y, x)。
    """
    m = mask.astype(bool)
    if not m.any():
        return 0.0, 0.0, 0.0
    z = int(np.argmax(m.reshape(m.shape[0], -1).sum(1)))
    ys, xs = np.nonzero(m[z])
    pts = np.stack([ys * spacing_mm[1], xs * spacing_mm[2]], 1).astype(float)
    if len(pts) == 1:
        d = float(min(spacing_mm[1], spacing_mm[2]))
        return d, d, d
    c = pts - pts.mean(0)
    _, vecs = np.linalg.eigh(np.cov(c.T))
    long_ax, short_ax = vecs[:, 1], vecs[:, 0]
    px = float(min(spacing_mm[1], spacing_mm[2]))  # 像素本身的寬度
    long_d = float(np.ptp(c @ long_ax)) + px
    short_d = float(np.ptp(c @ short_ax)) + px
    return long_d, short_d, (long_d + short_d) / 2


def vdt_days(v1_mm3: float, v2_mm3: float, dt_days: float) -> float:
    if v1_mm3 <= 0 or v2_mm3 <= 0 or dt_days <= 0:
        return float("nan")
    if v2_mm3 <= v1_mm3:
        return float("inf")
    return dt_days * math.log(2) / math.log(v2_mm3 / v1_mm3)


def lungrads_growth(mean_d1_mm: float, mean_d2_mm: float, dt_days: float) -> bool:
    """Lung-RADS v2022：平均徑增加 > 1.5 mm（12 個月內）。間隔超過一年不另外放寬，間隔天數另欄輸出供報告端判斷。"""
    if dt_days <= 0:
        return False
    return (mean_d2_mm - mean_d1_mm) > LUNGRADS_GROWTH_MM


def match_nodules(prior_xyz: np.ndarray, current_xyz: np.ndarray,
                  transform: Callable[[np.ndarray], np.ndarray] | None = None,
                  max_dist_mm: float = 15.0) -> list[tuple[int, int, float]]:
    """前次結節（世界座標 mm）→ 轉換到本次座標後與本次結節一對一配對。回傳 [(prior_i, current_j, 距離 mm)]。"""
    from scipy.optimize import linear_sum_assignment

    prior_xyz = np.asarray(prior_xyz, float).reshape(-1, 3)
    current_xyz = np.asarray(current_xyz, float).reshape(-1, 3)
    if len(prior_xyz) == 0 or len(current_xyz) == 0:
        return []
    p = transform(prior_xyz) if transform is not None else prior_xyz
    d = np.linalg.norm(p[:, None, :] - current_xyz[None, :, :], axis=2)
    big = 1e6
    cost = np.where(d <= max_dist_mm, d, big)
    rows, cols = linear_sum_assignment(cost)
    return [(int(i), int(j), float(d[i, j])) for i, j in zip(rows, cols) if d[i, j] <= max_dist_mm]


def build_tracks(nodules: pd.DataFrame,
                 transforms: dict[tuple[str, int, int], Callable[[np.ndarray], np.ndarray]] | None = None,
                 max_dist_mm: float = 15.0) -> pd.DataFrame:
    """把逐次掃描的結節串成跨年軌跡，並算出與前一次相比的生長量。

    nodules 欄位：patient_id, exam（0/1/2）, exam_day（自 T0 起的天數）, x, y, z（mm）,
                 long_d, short_d, mean_d（mm）, volume（mm³），可另帶 risk_* 等欄位會原樣保留。
    transforms[(patient_id, prior_exam, current_exam)]：把前次座標映到本次座標的函式（沒給就用恆等）。
    回傳每次掃描每顆結節一列，加上 nodule_id（跨年固定）、prior_*、days_since_prior、volume_change_pct、
    vdt_days、is_new、lungrads_growth。
    """
    transforms = transforms or {}
    out = []
    for pid, g in nodules.groupby("patient_id", sort=False):
        g = g.sort_values("exam").copy()
        exams = sorted(g.exam.unique())
        next_id = 0
        prev = None
        prev_exam = None
        for e in exams:
            cur = g[g.exam == e].reset_index(drop=True).copy()
            cur["nodule_id"] = -1
            for col in ("prior_volume", "prior_mean_d", "days_since_prior", "match_dist_mm"):
                cur[col] = np.nan
            if prev is not None and len(prev) and len(cur):
                tf = transforms.get((pid, prev_exam, e))
                pairs = match_nodules(prev[["x", "y", "z"]].to_numpy(), cur[["x", "y", "z"]].to_numpy(),
                                      tf, max_dist_mm)
                for i, j, dist in pairs:
                    cur.loc[j, "nodule_id"] = prev.loc[i, "nodule_id"]
                    cur.loc[j, "prior_volume"] = prev.loc[i, "volume"]
                    cur.loc[j, "prior_mean_d"] = prev.loc[i, "mean_d"]
                    cur.loc[j, "days_since_prior"] = cur.loc[j, "exam_day"] - prev.loc[i, "exam_day"]
                    cur.loc[j, "match_dist_mm"] = dist
            for j in cur.index[cur.nodule_id < 0]:
                cur.loc[j, "nodule_id"] = next_id
                next_id += 1
            if len(cur):
                next_id = max(next_id, int(cur.nodule_id.max()) + 1)
            cur["is_new"] = cur.prior_volume.isna() & (prev is not None)
            out.append(cur)
            prev, prev_exam = cur, e
    if not out:
        return nodules.assign(nodule_id=[], is_new=[])
    df = pd.concat(out, ignore_index=True)
    df["nodule_id"] = df.nodule_id.astype(int)
    df["volume_change_pct"] = (df.volume / df.prior_volume - 1) * 100
    df["vdt_days"] = [vdt_days(v1, v2, dt) if not np.isnan(v1) else np.nan
                      for v1, v2, dt in zip(df.prior_volume, df.volume, df.days_since_prior)]
    df["lungrads_growth"] = [bool(lungrads_growth(d1, d2, dt)) if not np.isnan(d1) else False
                             for d1, d2, dt in zip(df.prior_mean_d, df.mean_d, df.days_since_prior)]
    return df
