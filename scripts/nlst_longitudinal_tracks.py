"""
nlst_longitudinal_tracks.py — 把 nlst_longitudinal_extract 的逐次掃描結節串成跨年軌跡（CPU）

輸入：extract 的輸出資料夾（nodules.csv、transforms.csv、cohort.csv）
輸出（--out）：
  tracks.csv        每次掃描每顆結節一列（給 Lung-RADS 報告產生器）：PatientID, SeriesInstanceUID, exam,
                    nodule_id, x, y, z, mean_d, volume, prior_mean_d, prior_volume, days_since_prior,
                    volume_change_pct, vdt_days, is_new, lungrads_growth, det_prob, malignancy
  qc.json           對位成功率、配對率、配對距離、與醫師「長大」紀錄（nlst_ctabc）的一致性

只保留「像真結節」的候選：偵測分數 ≥ --min_det，且分割到體積（seg_voxels > 0）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.nlst_longitudinal_extract import apply_affine
from src.eval.nodule_growth import build_tracks


def make_transforms(tf: pd.DataFrame, max_metric: float = -0.2) -> dict:
    """MI metric 越負越好；比 max_metric 差的對位視為失敗（該對檢查不做配對）。"""
    out = {}
    for _, r in tf.iterrows():
        if r.metric <= max_metric:
            out[(str(r.PatientID), int(r.prior_exam), int(r.exam))] = (lambda row: lambda p: apply_affine(row, p))(r)
    return out


def chain_ok(t: pd.DataFrame, nod: pd.DataFrame, tf_keys: set) -> pd.Series:
    """某次檢查能不能和前一次比：前一次存在且那一對對位成功。"""
    prev = nod.groupby("PatientID").exam.apply(lambda s: sorted(set(s)))
    ok = []
    for r in t.itertuples():
        ex = prev.get(r.PatientID, [])
        i = ex.index(r.exam) if r.exam in ex else 0
        ok.append(i > 0 and (r.PatientID, ex[i - 1], r.exam) in tf_keys)
    return pd.Series(ok, index=t.index)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_det", type=float, default=0.5)
    ap.add_argument("--max_dist_mm", type=float, default=10.0)
    ap.add_argument("--ctabc", default=None, help="nlst_ctabc parquet（沒給就從 IDC 臨床表找）")
    a = ap.parse_args()

    run = Path(a.run_dir)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    nod = pd.read_csv(run / "nodules.csv", dtype={"PatientID": str})
    tf = pd.read_csv(run / "transforms.csv", dtype={"PatientID": str}) if (run / "transforms.csv").exists() \
        else pd.DataFrame(columns=["PatientID", "prior_exam", "exam", "metric"])
    n_all = len(nod)
    nod = nod[(nod.det_prob >= a.min_det) & (nod.seg_voxels > 0)].copy()
    nod["patient_id"] = nod.PatientID
    nod["exam_day"] = nod.exam_day.fillna(nod.exam * 365)
    transforms = make_transforms(tf)
    t = build_tracks(nod, transforms, max_dist_mm=a.max_dist_mm)
    t["comparable"] = chain_ok(t, nod, set(transforms))
    cols = ["PatientID", "SeriesInstanceUID", "exam", "exam_day", "nodule_id", "x", "y", "z", "long_d", "short_d",
            "mean_d", "volume", "prior_mean_d", "prior_volume", "days_since_prior", "volume_change_pct", "vdt_days",
            "is_new", "lungrads_growth", "comparable", "match_dist_mm", "det_prob", "malignancy", "emb_row"]
    t[[c for c in cols if c in t.columns]].to_csv(out / "tracks.csv", index=False)

    comp = t[t.comparable]
    qc = {
        "candidates_all": int(n_all), "nodules_kept": int(len(nod)),
        "patients": int(nod.PatientID.nunique()), "scans": int(nod.SeriesInstanceUID.nunique()),
        "registrations": int(len(tf)), "registrations_ok": int(len(transforms)),
        "registration_metric_median": float(tf.metric.median()) if len(tf) else None,
        "follow_up_nodules": int(len(comp)),
        "matched_to_prior_pct": float(100 * (~comp.is_new).mean()) if len(comp) else None,
        "match_dist_mm_median": float(comp.match_dist_mm.median()) if comp.match_dist_mm.notna().any() else None,
        "lungrads_growth_nodules": int(comp.lungrads_growth.sum()),
        "vdt_lt_400_nodules": int((comp.vdt_days < 400).sum()),
        "new_nodules": int(comp.is_new.sum()),
    }
    # 與醫師「比前次長大」紀錄（病人 × 檢查層級）的一致性
    try:
        if a.ctabc:
            c = pd.read_parquet(a.ctabc)
        else:
            import glob
            c = pd.read_parquet(sorted(glob.glob(str(Path.home() / ".local/share/IDC/*/clinical_data/nlst_ctabc")))[-1])
        from scripts.nlst_longitudinal_cohort import growth_flags
        g = growth_flags(c)
        ours = comp.groupby(["PatientID", "exam"]).lungrads_growth.any().reset_index()
        m = ours.merge(g, on=["PatientID", "exam"], how="left").fillna({"grew_flag": False})
        tp = int((m.lungrads_growth & m.grew_flag).sum())
        qc["vs_radiologist_growth"] = {
            "exams": int(len(m)), "radiologist_grew": int(m.grew_flag.sum()), "ours_grew": int(m.lungrads_growth.sum()),
            "both": tp, "sensitivity": float(tp / max(1, m.grew_flag.sum())),
            "ppv": float(tp / max(1, m.lungrads_growth.sum())),
        }
    except Exception as e:  # 臨床表不在手邊也不影響主要輸出
        qc["vs_radiologist_growth"] = f"skipped: {type(e).__name__}: {e}"
    json.dump(qc, open(out / "qc.json", "w"), indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(qc, indent=2, default=str))


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
