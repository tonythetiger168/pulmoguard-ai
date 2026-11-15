"""
adapters.py — 把現有管線的輸出檔轉成 Study

支援的輸入：
- MONAI 偵測器結果 JSON（scripts/eval_monai_detector.py、scripts/nlst_validate.py 產生的 det_*.json）
  {"validation": [{"image": "<uid>.mhd", "box": [[x,y,z,w,h,d], ...], "label_scores": [...]}]}
  box 為 cccwhd，世界座標 mm。
- LUNA25 惡性度 candidate_scores.csv（nlst_validate.py）：seriesuid, coordX, coordY, coordZ, probability, malignancy
  以座標最近鄰（預設 ≤ 2 mm）對回偵測框。
- 分割結果 JSON（LIDC 偵測器裁切分割，仍在訓練中，可先給 mock）：
  {"mock": true, "lesions": [{"center_mm": [x,y,z], "diam_mm": 7.1, "volume_mm3": 190.0}]}
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from .schema import ModelInfo, Nodule, Study


def _uid(image: str) -> str:
    name = Path(image).name
    for ext in (".nii.gz", ".mhd", ".nii", ".nrrd"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


def load_detector_json(path: str | Path) -> dict[str, list[dict]]:
    res = json.loads(Path(path).read_text())
    items = res["validation"] if isinstance(res, dict) else res
    out: dict[str, list[dict]] = {}
    for it in items:
        out.setdefault(_uid(it["image"]), []).extend(
            {"box": [float(v) for v in b], "det_prob": float(s)}
            for b, s in zip(it["box"], it["label_scores"]))
    return out


def _dist(a, b) -> float:
    return math.dist(a, b)


def build_study(study_id: str, dets: list[dict], *, det_threshold: float = 0.5, top_k: int = 5,
                malignancy: pd.DataFrame | None = None, seg: dict | None = None,
                match_mm: float = 2.0) -> Study:
    keep = sorted((d for d in dets if d["det_prob"] >= det_threshold), key=lambda d: -d["det_prob"])[:top_k]
    mal = None
    if malignancy is not None and len(malignancy):
        mal = malignancy[malignancy.seriesuid.astype(str) == str(study_id)]
    lesions = (seg or {}).get("lesions", [])
    nodules = []
    for i, d in enumerate(keep, start=1):
        x, y, z, w, h, _ = d["box"]
        center = (x, y, z)
        n = Nodule(nodule_id=i, center_mm=center, diameter_mm=round((w + h) / 2, 1),
                   diameter_source="detector_box", det_prob=round(d["det_prob"], 3))
        if mal is not None and len(mal):
            dist = mal.apply(lambda r, c=center: _dist(c, (r.coordX, r.coordY, r.coordZ)), axis=1)
            j = dist.idxmin()
            if dist[j] <= match_mm:
                n.malignancy_prob = round(float(mal.loc[j, "malignancy"]), 3)
        for les in lesions:
            if _dist(center, les["center_mm"]) <= max(n.diameter_mm / 2, match_mm):
                n.diameter_mm = round(float(les["diam_mm"]), 1)
                n.volume_mm3 = round(float(les["volume_mm3"]), 1) if les.get("volume_mm3") is not None else None
                n.diameter_source = "segmentation"
                break
        nodules.append(n)
    used_seg = any(n.diameter_source == "segmentation" for n in nodules)
    used_mal = any(n.malignancy_prob is not None for n in nodules)
    models = ModelInfo(malignancy=ModelInfo().malignancy if used_mal else None,
                       segmentation=("LIDC detector-crop segmentation (MOCK — model still training)"
                                     if used_seg and seg.get("mock") else
                                     "LIDC detector-crop segmentation" if used_seg else None))
    return Study(study_id=study_id, nodules=nodules, det_threshold=det_threshold, models=models)


def load_study_json(path: str | Path) -> Study:
    """直接讀一份 Study JSON（例如 examples/ 或其他系統送來的結構化結果）。"""
    raw = json.loads(Path(path).read_text())
    nods = [Nodule(**{**n, "center_mm": tuple(n["center_mm"])}) for n in raw.pop("nodules", [])]
    models = ModelInfo(**raw.pop("models", {}))
    return Study(nodules=nods, models=models, **raw)


def _f(v) -> float | None:
    return None if pd.isna(v) else float(v)


def load_tracks(path: str | Path, *, det_threshold: float = 0.5, top_k: int = 5) -> dict[str, Study]:
    """讀 NLST 序列影像追蹤結果（scripts/nlst_longitudinal_tracks.py 的 tracks.csv），每個 series 一個 Study。

    欄位：PatientID, SeriesInstanceUID, exam, nodule_id, x/y/z, long_d/short_d/mean_d, volume, prior_mean_d,
    prior_volume, days_since_prior, volume_change_pct, vdt_days, is_new, lungrads_growth, comparable,
    det_prob, malignancy；選配 is_new_confident, prior_nearest_mm（沒有時所有 is_new 都視為確定新生）。
    """
    t = pd.read_csv(path, dtype={"SeriesInstanceUID": str, "PatientID": str})
    studies: dict[str, Study] = {}
    for uid, g in t.groupby("SeriesInstanceUID", sort=False):
        exam = int(g.exam.iloc[0])
        g = g[g.det_prob >= det_threshold].sort_values("det_prob", ascending=False).head(top_k)
        nods = []
        for i, r in enumerate(g.itertuples(), start=1):
            followup = exam > 0
            nods.append(Nodule(
                nodule_id=i, center_mm=(float(r.x), float(r.y), float(r.z)),
                diameter_mm=round(float(r.mean_d), 1), diameter_source="tracking",
                det_prob=round(float(r.det_prob), 3), malignancy_prob=round(float(r.malignancy), 3),
                volume_mm3=_f(r.volume),
                comparable=bool(r.comparable) if followup else None,
                is_new=bool(r.is_new) if followup and r.comparable else None,
                new_confident=(bool(r.is_new_confident) if followup and r.comparable and r.is_new
                               and hasattr(r, "is_new_confident") else None),
                prior_nearest_mm=_f(getattr(r, "prior_nearest_mm", float("nan"))) if followup else None,
                prior_diameter_mm=round(_f(r.prior_mean_d), 1) if followup and _f(r.prior_mean_d) else None,
                days_since_prior=_f(r.days_since_prior) if followup else None,
                volume_change_pct=_f(r.volume_change_pct) if followup else None,
                vdt_days=_f(r.vdt_days) if followup else None,
                upstream_growth_flag=bool(r.lungrads_growth) if followup and r.comparable else None))
        models = ModelInfo(detector=ModelInfo().detector + " + NLST longitudinal tracking (PR #10)")
        studies[str(uid)] = Study(study_id=str(uid), nodules=nods, det_threshold=det_threshold, models=models,
                                  baseline=exam == 0, exam_index=exam)
    return studies
