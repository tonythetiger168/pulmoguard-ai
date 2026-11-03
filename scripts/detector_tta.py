"""
detector_tta.py — MONAI 預訓練偵測器 + 翻轉測試時增強（TTA），在 LUNA16 subset0 上比較 CPM

做法（不改 bundle、不重訓）：
  1) 每種翻轉（none / x / y / xy …）把 .mhd 影像陣列沿指定軸翻轉，原點、間距、方向不變，另存一份
  2) 每份各跑一次 bundle 推論；偵測框中心在世界座標中「翻回」原影像：x = 2*origin + (N-1)*spacing - x'
  3) 合併：各次偵測依距離分群（中心距離 ≤ max(MIN_R, 0.5 × 框大小)），每群每次推論最多取一個
     - mean：分群分數總和 / 推論次數（只被一次推論找到的候選分數會被壓低 → 去假陽性）
     - max ：分群內最高分
  4) 報告：每次單獨推論的 CPM、合併後的 CPM

方向矩陣不是單位矩陣的影像（LUNA16 幾乎沒有）不翻轉，直接沿用原圖，log 裡會列出。

用法（repo 根目錄）：
  python -m scripts.detector_tta --luna_root data/luna16 --subset 0 --out runs/detector_tta
  試跑：加 --limit 3 --flips none,x
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

from scripts.eval_monai_detector import BUNDLE, fetch_bundle, load_detections, run, write_datalist
from src.eval.froc import froc

AXES = {"x": 0, "y": 1, "z": 2}   # 世界座標軸（coordX/Y/Z）；SimpleITK 陣列是 (z, y, x)
MIN_R = 4.0                        # mm，分群半徑下限


def flip_volume(src: Path, dst: Path, flip: str) -> dict | None:
    """回傳翻回用的參數 {axis: (origin, size, spacing)}；方向不是單位矩陣則回傳 None（不翻）"""
    img = sitk.ReadImage(str(src))
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not np.allclose(np.array(img.GetDirection()), np.eye(3).ravel(), atol=1e-3):
        sitk.WriteImage(img, str(dst))
        return None
    arr = sitk.GetArrayFromImage(img)
    params = {}
    for c in flip:
        ax = AXES[c]
        arr = np.flip(arr, axis=2 - ax)
        params[ax] = (img.GetOrigin()[ax], img.GetSize()[ax], img.GetSpacing()[ax])
    out = sitk.GetImageFromArray(np.ascontiguousarray(arr))
    out.CopyInformation(img)
    sitk.WriteImage(out, str(dst))
    return params


def unflip(dets: pd.DataFrame, params: dict[str, dict | None]) -> pd.DataFrame:
    dets = dets.copy()
    cols = ["coordX", "coordY", "coordZ"]
    for uid, p in params.items():
        if not p:
            continue
        m = dets.seriesuid == uid
        for ax, (o, n, s) in p.items():
            dets.loc[m, cols[ax]] = 2 * o + (n - 1) * s - dets.loc[m, cols[ax]]
    return dets


def merge(passes: list[pd.DataFrame], mode: str) -> pd.DataFrame:
    """各次推論的偵測依距離分群（貪婪：分數高的先當群中心），每群每次推論最多一個"""
    k = len(passes)
    allp = pd.concat([d.assign(pass_id=i) for i, d in enumerate(passes)], ignore_index=True)
    rows = []
    for uid, g in allp.groupby("seriesuid"):
        g = g.sort_values("probability", ascending=False).reset_index(drop=True)
        xyz = g[["coordX", "coordY", "coordZ"]].to_numpy()
        size = g["size"].to_numpy() if "size" in g else np.full(len(g), 2 * MIN_R)
        used = np.zeros(len(g), bool)
        for i in range(len(g)):
            if used[i]:
                continue
            r = max(MIN_R, 0.5 * size[i])
            d = np.linalg.norm(xyz - xyz[i], axis=1)
            members, seen = [], set()
            for j in np.where((d <= r) & ~used)[0]:   # 已依分數排序 → 每次推論取最高分那個
                pid = g.pass_id[j]
                if pid in seen:
                    continue
                seen.add(pid)
                members.append(j)
            used[members] = True
            sc = g.probability.to_numpy()[members]
            w = sc / sc.sum()
            c = (xyz[members] * w[:, None]).sum(0)
            score = sc.sum() / k if mode == "mean" else sc.max()
            rows.append({"seriesuid": uid, "coordX": c[0], "coordY": c[1], "coordZ": c[2],
                         "probability": float(score), "n_passes": len(members)})
    return pd.DataFrame(rows)


def load_with_size(result_json: Path) -> pd.DataFrame:
    """同 load_detections，另外留下框的最大邊長（mm）給分群用"""
    dets = load_detections(result_json)
    res = json.loads(result_json.read_text())
    res = res.get("validation", res.get("test", [])) if isinstance(res, dict) else res
    sizes = [float(max(b[3:6])) for item in res for b in item["box"]]
    dets["size"] = sizes if len(sizes) == len(dets) else 2 * MIN_R
    return dets


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--luna_root", default="data/luna16")
    p.add_argument("--subset", type=int, default=0)
    p.add_argument("--bundle_dir", default="models/bundles")
    p.add_argument("--out", default="runs/detector_tta")
    p.add_argument("--flips", default="none,x,y,xy", help="逗號分隔，例如 none,x,y,z,xy")
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()

    luna_root = Path(a.luna_root).resolve()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    bundle_root = Path(a.bundle_dir).resolve() / BUNDLE
    fetch_bundle(bundle_root)
    datalist = out / "datalist.json"
    uids = write_datalist(luna_root, a.subset, datalist, a.limit)

    ann = pd.read_csv(luna_root / "annotations.csv")
    ann = ann[ann.seriesuid.isin(uids)]
    ex_path = luna_root / "annotations_excluded.csv"
    excluded = pd.read_csv(ex_path) if ex_path.exists() else None
    if excluded is not None:
        excluded = excluded[excluded.seriesuid.isin(uids)]

    def score(dets):
        r = froc(dets, ann, n_scans=len(uids), excluded=excluded)
        return {"cpm": r["cpm"], "sensitivity": r["sensitivity"]}

    summary = {"subset": a.subset, "n_scans": len(uids), "n_nodules": int(len(ann)), "passes": {}}
    passes = []
    for flip in [f.strip() for f in a.flips.split(",") if f.strip()]:
        t0 = time.time()
        fdir = out / f"flip_{flip}"
        fdir.mkdir(exist_ok=True)
        result = fdir / "detections.json"
        if not result.exists():
            data_dir = luna_root
            params: dict[str, dict | None] = {}
            if flip != "none":
                data_dir = out / "_images" / flip
                for uid in uids:
                    src = luna_root / f"subset{a.subset}" / f"{uid}.mhd"
                    params[uid] = flip_volume(src, data_dir / f"subset{a.subset}" / f"{uid}.mhd", flip)
                skipped = [u for u, v in params.items() if v is None]
                if skipped:
                    print(f"flip {flip}: {len(skipped)} scans with non-identity direction left unflipped")
            (fdir / "flip_params.json").write_text(json.dumps(params))
            run([sys.executable, "-m", "monai.bundle", "run",
                 "--config_file", "configs/inference.json",
                 "--bundle_root", str(bundle_root),
                 "--whether_raw_luna16", "True",
                 "--dataset_dir", str(data_dir),
                 "--data_list_file_path", str(datalist),
                 "--output_dir", str(fdir),
                 "--output_filename", result.name], cwd=bundle_root)
            if flip != "none":   # 翻轉影像很大，用完就刪
                import shutil
                shutil.rmtree(data_dir, ignore_errors=True)
        params = json.loads((fdir / "flip_params.json").read_text()) if (fdir / "flip_params.json").exists() else {}
        params = {u: ({int(k): tuple(v) for k, v in p_.items()} if p_ else None) for u, p_ in params.items()}
        dets = unflip(load_with_size(result), params)
        dets.drop(columns="size").to_csv(fdir / "detections.csv", index=False)
        passes.append(dets)
        summary["passes"][flip] = {**score(dets), "minutes": round((time.time() - t0) / 60, 1)}
        print(f"flip {flip}: CPM {summary['passes'][flip]['cpm']:.4f}", flush=True)

    if len(passes) > 1:
        for mode in ("mean", "max"):
            m = merge(passes, mode)
            m.to_csv(out / f"merged_{mode}.csv", index=False)
            summary[f"tta_{mode}"] = score(m)
            print(f"TTA {mode} ({len(passes)} passes): CPM {summary[f'tta_{mode}']['cpm']:.4f}")
        base = summary["passes"].get("none", {}).get("cpm")
        best = max(summary["tta_mean"]["cpm"], summary["tta_max"]["cpm"])
        summary["baseline_cpm"] = base
        summary["best_tta_cpm"] = best
        summary["gain"] = None if base is None else round(best - base, 4)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "passes"}, indent=2))


if __name__ == "__main__":
    main()
