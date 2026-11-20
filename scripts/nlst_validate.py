"""
nlst_validate.py — NLST 病人層級驗證：MONAI 預訓練偵測器 → LUNA25 惡性度模型（5-fold 集成）→ 病人分數

流程：
  1) 偵測：lung_nodule_ct_detection bundle 對每個 1 mm .mhd 推論（與 LUNA16 試跑同一設定，CPM 0.888）
  2) 惡性度：每例取偵測分數最高的 TOP_K 個候選（分數 ≥ DET_MIN），以候選中心取 50 mm 視野 64³ patch，
     丟進 LUNA25 5 個 fold 的 ResNet-18，取平均機率
  3) 病人分數 = 候選惡性度最大值（沒有候選 = 0）；另外報「只看偵測分數最大值」當對照
  4) 指標：AUC（bootstrap 95% CI）、AP；並依偵測門檻分層列出

用法：
    python -m scripts.nlst_validate --cohort runs/nlst/cohort.csv --mhd_dir data/nlst/mhd \
        --luna25_runs runs/luna25 --out runs/nlst
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai.networks.nets import resnet18
from sklearn.metrics import average_precision_score, roc_auc_score

from scripts.eval_monai_detector import BUNDLE, fetch_bundle, load_detections, run
from src.datasets.luna25 import hu_scale, resample_patch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 5
DET_MIN = 0.3
BLOCK_HALF = 40   # 81³ 個 1 mm 體素的區塊，足夠 50 mm 視野加旋轉


def bootstrap_auc(y, p, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    a = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            a.append(roc_auc_score(y[i], p[i]))
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def metrics(y, p) -> dict:
    return {"auc": float(roc_auc_score(y, p)), "auc_95ci": bootstrap_auc(y, p),
            "ap": float(average_precision_score(y, p)), "n": int(len(y)), "n_pos": int(np.sum(y))}


def load_models(runs_dir: str) -> list[torch.nn.Module]:
    paths = sorted(glob.glob(f"{runs_dir}/fold*/best.pt"))
    if not paths:
        raise FileNotFoundError(f"no fold*/best.pt under {runs_dir}")
    models = []
    for p in paths:
        m = resnet18(spatial_dims=3, n_input_channels=1, num_classes=1)
        sd = torch.load(p, map_location="cpu")
        m.load_state_dict({k: v.float() for k, v in sd.items()})
        models.append(m.to(DEVICE).eval())
    print(f"loaded {len(models)} LUNA25 fold models", flush=True)
    return models


def extract_block(arr_zyx: np.ndarray, center_zyx) -> np.ndarray:
    c = np.round(center_zyx).astype(int)
    box = [(ci - BLOCK_HALF, ci + BLOCK_HALF + 1) for ci in c]
    out = np.full((2 * BLOCK_HALF + 1,) * 3, -1000, dtype=np.float32)
    src = tuple(slice(max(a, 0), min(b, n)) for (a, b), n in zip(box, arr_zyx.shape))
    dst = tuple(slice(s.start - a, s.stop - a) for s, (a, _) in zip(src, box))
    if all(s.stop > s.start for s in src):
        out[dst] = arr_zyx[src]
    return out


@torch.no_grad()
def score_candidates(models, mhd: Path, cands: pd.DataFrame) -> np.ndarray:
    import SimpleITK as sitk
    img = sitk.ReadImage(str(mhd))
    arr = sitk.GetArrayFromImage(img)
    sp_zyx = np.array(img.GetSpacing())[::-1]
    patches = []
    for r in cands.itertuples():
        idx_xyz = img.TransformPhysicalPointToContinuousIndex((float(r.coordX), float(r.coordY), float(r.coordZ)))
        block = extract_block(arr, np.array(idx_xyz)[::-1])
        patches.append(hu_scale(resample_patch(block, sp_zyx, 64, 50.0)))
    x = torch.from_numpy(np.stack(patches)[:, None]).float().to(DEVICE)
    probs = [torch.sigmoid(m(x).squeeze(1).float()).cpu().numpy() for m in models]
    return np.mean(probs, axis=0)


CHUNK = 40


def _run_bundle(bundle_root: Path, mhd_dir: Path, out: Path, uids: list[str], tag: str) -> list[dict] | None:
    datalist = out / f"datalist_{tag}.json"
    res = out / f"det_{tag}.json"
    datalist.write_text(json.dumps({"validation": [{"image": f"{u}.mhd"} for u in uids]}))
    try:
        run([sys.executable, "-m", "monai.bundle", "run", "--config_file", "configs/inference.json",
             "--bundle_root", str(bundle_root), "--whether_raw_luna16", "True",
             "--dataset_dir", str(mhd_dir), "--data_list_file_path", str(datalist),
             "--output_dir", str(out), "--output_filename", res.name], cwd=bundle_root)
    except Exception as e:  # 某些掃描會讓 RetinaNet 的 anchor 數對不上而整批中止 → 交給呼叫端二分
        print(f"bundle failed on {len(uids)} scans ({tag}): {type(e).__name__}", flush=True)
        return None
    r = json.loads(res.read_text())
    r = r.get("validation", r.get("test", [])) if isinstance(r, dict) else r
    datalist.unlink(missing_ok=True)
    res.unlink(missing_ok=True)
    return r


def detect_all(bundle_root: Path, mhd_dir: Path, out: Path, uids: list[str], failed: list[str]) -> list[dict]:
    """分批跑偵測器；整批失敗時對半切重跑，直到找出單一失敗的掃描（記到 failed）"""
    items: list[dict] = []
    todo = [uids[i: i + CHUNK] for i in range(0, len(uids), CHUNK)]
    n = 0
    while todo:
        chunk = todo.pop(0)
        r = _run_bundle(bundle_root, mhd_dir, out, chunk, f"{n:04d}")
        n += 1
        if r is not None:
            items += r
            print(f"detector: {len(items)} scans done, {len(failed)} failed", flush=True)
        elif len(chunk) == 1:
            failed.append(chunk[0])
        else:
            h = len(chunk) // 2
            todo[:0] = [chunk[:h], chunk[h:]]
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--mhd_dir", required=True)
    ap.add_argument("--luna25_runs", required=True)
    ap.add_argument("--bundle_dir", default="models/bundles")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip_detection", action="store_true")
    a = ap.parse_args()

    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    mhd_dir = Path(a.mhd_dir).resolve()
    cohort = pd.read_csv(a.cohort, dtype={"PatientID": str, "SeriesInstanceUID": str})
    have = cohort.SeriesInstanceUID.map(lambda u: (mhd_dir / f"{u}.mhd").exists())
    if (~have).any():
        print(f"warning: {int((~have).sum())} cohort scans have no .mhd and are dropped")
    cohort = cohort[have].reset_index(drop=True)

    det_json = out / "detections.json"
    failed: list[str] = []
    if not a.skip_detection:
        bundle_root = Path(a.bundle_dir).resolve() / BUNDLE
        fetch_bundle(bundle_root)
        items = detect_all(bundle_root, mhd_dir, out, list(cohort.SeriesInstanceUID), failed)
        det_json.write_text(json.dumps({"validation": items}))
        if failed:
            print(f"warning: detector failed on {len(failed)} scans; they are dropped", flush=True)
            (out / "detector_failed.txt").write_text("\n".join(failed) + "\n")
            cohort = cohort[~cohort.SeriesInstanceUID.isin(failed)].reset_index(drop=True)
    dets = load_detections(det_json)
    dets.to_csv(out / "detections.csv", index=False)
    print(f"{len(dets)} detections on {dets.seriesuid.nunique()} scans", flush=True)

    models = load_models(a.luna25_runs)
    rows, cand_rows = [], []
    for r in cohort.itertuples():
        d = dets[(dets.seriesuid == r.SeriesInstanceUID) & (dets.probability >= DET_MIN)]
        d = d.sort_values("probability", ascending=False).head(TOP_K)
        mal = score_candidates(models, mhd_dir / f"{r.SeriesInstanceUID}.mhd", d) if len(d) else np.array([])
        for (_, dr), m in zip(d.iterrows(), mal):
            cand_rows.append({**dr.to_dict(), "PatientID": r.PatientID, "malignancy": float(m)})
        rows.append({"PatientID": r.PatientID, "SeriesInstanceUID": r.SeriesInstanceUID, "label": int(r.label),
                     "n_candidates": len(d), "max_det": float(d.probability.max()) if len(d) else 0.0,
                     "max_malignancy": float(mal.max()) if len(mal) else 0.0})
    pt = pd.DataFrame(rows)
    pt.to_csv(out / "patient_scores.csv", index=False)
    pd.DataFrame(cand_rows).to_csv(out / "candidate_scores.csv", index=False)

    summary = {
        "task": "NLST T0 LDCT -> lung cancer diagnosed within 365 days (patient level)",
        "cohort_note": "NLST patients not in LUNA25; same trial and scanners, so internal-external rather than external",
        "pipeline": f"MONAI lung_nodule_ct_detection -> top {TOP_K} candidates with det >= {DET_MIN} -> "
                    "LUNA25 ResNet-18 5-fold ensemble -> patient score = max malignancy",
        "malignancy_model": metrics(pt.label, pt.max_malignancy),
        "detector_only": metrics(pt.label, pt.max_det),
        "scans_without_candidates": int((pt.n_candidates == 0).sum()),
        "detector_failed_scans": len(failed),
        "positives_without_candidates": int(((pt.n_candidates == 0) & (pt.label == 1)).sum()),
    }
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
