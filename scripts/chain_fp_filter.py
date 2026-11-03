"""
chain_fp_filter.py — 把假陽性過濾器（train_baseline 的 best.pt）接在偵測器後面，重算 FROC

流程：
  1) 讀偵測器輸出 detections.csv（seriesuid, coordX/Y/Z, probability）
  2) 對每個偵測點切 64³ patch，用過濾器算「是真結節」機率 p_cls
  3) 用幾種方式融合分數，各自重算 LUNA16 FROC/CPM，與偵測器單獨的結果比較

融合方式：
  det      偵測器分數（原始基準）
  cls      只用過濾器分數
  product  p_det × p_cls（兩者都要高；等同幾何平均的排序）
  mean     (p_det + p_cls) / 2

注意：過濾器的訓練資料不可包含受測 subset（預設偵測器在 subset0 上評估，
      過濾器只用 subset 1–4 訓練），否則結果會偏樂觀。

用法：
  python -m scripts.chain_fp_filter --detections runs/monai_detector/detections.csv \
      --weights runs/fp_filter/best.pt --meta data/processed/meta.csv \
      --npy_dir data/processed/npy --luna_root data/luna16 --subset 0 --out runs/fp_chain
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.froc import froc

FUSIONS = ("det", "cls", "product", "mean")


def fuse(p_det: np.ndarray, p_cls: np.ndarray, how: str) -> np.ndarray:
    if how == "det":
        return p_det
    if how == "cls":
        return p_cls
    if how == "product":
        return p_det * p_cls
    if how == "mean":
        return (p_det + p_cls) / 2
    raise ValueError(how)


def score_detections(dets: pd.DataFrame, meta: pd.DataFrame, npy_dir: str, weights: str,
                     patch_size: int = 64, batch_size: int = 32, tta: bool = True) -> np.ndarray:
    """回傳每個偵測點的過濾器機率；沒有前處理影像的 series 回傳 NaN"""
    import torch

    from scripts.train_baseline import DEVICE, build_model
    from src.datasets.luna16 import crop_pad
    from src.preprocess.pipeline import world_to_voxel

    model = build_model()
    w = Path(weights)
    if not w.exists() and (w.parent / "last.pt").exists():   # 驗證 AUC 從未 > 0 時不會有 best.pt
        print(f"{w} not found: using {w.parent / 'last.pt'}")
        w = w.parent / "last.pt"
    state = torch.load(w, map_location=DEVICE)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    p_cls = np.full(len(dets), np.nan)
    size = (patch_size,) * 3
    for uid, g in dets.groupby("seriesuid"):
        npy = Path(npy_dir) / f"{uid}.npy"
        if uid not in meta.index or not npy.exists():
            continue
        vol = np.load(npy, mmap_mode="r")
        m = meta.loc[uid]
        patches = np.stack([
            crop_pad(vol, world_to_voxel((r.coordX, r.coordY, r.coordZ),
                                         m.origin_xyz, m.spacing_xyz, m.direction), size)
            for r in g.itertuples()
        ])[:, None]
        probs = []
        with torch.no_grad():
            for i in range(0, len(patches), batch_size):
                x = torch.as_tensor(patches[i:i + batch_size], dtype=torch.float32, device=DEVICE)
                views = [x] + ([x.flip(2), x.flip(3), x.flip(4)] if tta else [])
                p = torch.stack([torch.softmax(model(v).float(), 1)[:, 1] for v in views]).mean(0)
                probs.append(p.cpu().numpy())
        p_cls[dets.index.get_indexer(g.index)] = np.concatenate(probs)
    return p_cls


def evaluate_fusions(dets: pd.DataFrame, p_cls: np.ndarray, ann: pd.DataFrame, n_scans: int,
                     excluded: pd.DataFrame | None) -> dict:
    """沒有過濾器分數的偵測點保留偵測器原分數，避免被不公平地丟掉"""
    p_det = dets.probability.to_numpy(dtype=float)
    p_cls = np.where(np.isnan(p_cls), p_det, p_cls)
    out = {}
    for how in FUSIONS:
        d = dets.assign(probability=fuse(p_det, p_cls, how))
        out[how] = froc(d, ann, n_scans=n_scans, excluded=excluded)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--detections", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--meta", default="data/processed/meta.csv")
    p.add_argument("--npy_dir", default="data/processed/npy")
    p.add_argument("--luna_root", default="data/luna16")
    p.add_argument("--subset", type=int, default=0, help="偵測器受測的 subset")
    p.add_argument("--out", default="runs/fp_chain")
    p.add_argument("--patch_size", type=int, default=64)
    p.add_argument("--no_tta", dest="tta", action="store_false")
    a = p.parse_args()

    from src.datasets.luna16 import load_meta

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    luna = Path(a.luna_root)
    uids = sorted(x.stem for x in (luna / f"subset{a.subset}").glob("*.mhd"))
    if not uids:
        raise FileNotFoundError(f"no .mhd in {luna}/subset{a.subset}")
    dets = pd.read_csv(a.detections)
    dets = dets[dets.seriesuid.isin(uids)].reset_index(drop=True)
    meta = load_meta(a.meta)

    p_cls = score_detections(dets, meta, a.npy_dir, a.weights, a.patch_size, tta=a.tta)
    n_missing = int(np.isnan(p_cls).sum())
    dets.assign(p_cls=p_cls).to_csv(out / "detections_scored.csv", index=False)

    ann = pd.read_csv(luna / "annotations.csv")
    ann = ann[ann.seriesuid.isin(uids)]
    ex_path = luna / "annotations_excluded.csv"
    excluded = pd.read_csv(ex_path) if ex_path.exists() else None
    if excluded is not None:
        excluded = excluded[excluded.seriesuid.isin(uids)]

    res = evaluate_fusions(dets, p_cls, ann, len(uids), excluded)
    summary = {
        "subset": a.subset, "n_scans": len(uids), "n_detections": len(dets),
        "n_detections_without_filter_score": n_missing,
        "cpm": {k: round(v["cpm"], 4) for k, v in res.items()},
        "froc": res,
    }
    (out / "chain_froc.json").write_text(json.dumps(summary, indent=2))
    print("fusion   CPM    " + "  ".join(f"{r:>5}" for r in res["det"]["sensitivity"]))
    for k, v in res.items():
        print(f"{k:8s} {v['cpm']:.3f}  " + "  ".join(f"{s:.3f}" for s in v["sensitivity"].values()))
    print(f"{n_missing} detections had no preprocessed scan and kept the detector score")


if __name__ == "__main__":
    main()
