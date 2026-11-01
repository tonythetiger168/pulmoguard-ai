"""
eval_monai_detector.py — 用 MONAI 預訓練 RetinaNet（lung_nodule_ct_detection bundle）
在 LUNA16 subset0 上推論並計算 FROC / CPM。只做推論，不訓練（RTX 4090 約 1 小時內）。

bundle 權重以官方 fold0 切分訓練（train = subset1–9，val = subset0），
因此 subset0 對它是未見過的資料，可公平評估。其他 subset 曾用於訓練，結果會偏高。

用法（repo 根目錄）：
  python -m scripts.eval_monai_detector --luna_root data/luna16 --subset 0
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.eval.froc import froc

BUNDLE = "lung_nodule_ct_detection"


def write_datalist(luna_root: Path, subset: int, path: Path, limit: int = 0) -> list[str]:
    mhds = sorted((luna_root / f"subset{subset}").glob("*.mhd"))
    if limit:
        mhds = mhds[:limit]
    if not mhds:
        raise FileNotFoundError(f"no .mhd in {luna_root}/subset{subset}")
    items = [{"image": str(p.relative_to(luna_root))} for p in mhds]
    path.write_text(json.dumps({"validation": items}, indent=1))
    return [p.stem for p in mhds]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def load_detections(result_json: Path) -> pd.DataFrame:
    res = json.loads(result_json.read_text())
    if isinstance(res, dict):
        res = res.get("validation", res.get("test", []))
    rows = []
    for item in res:
        uid = Path(item["image"]).name.removesuffix(".mhd")
        for box, score in zip(item["box"], item["label_scores"]):
            # cccwhd：前三個為世界座標中心（與 LUNA16 coordX/Y/Z 相同座標系）
            rows.append({"seriesuid": uid, "coordX": box[0], "coordY": box[1],
                         "coordZ": box[2], "probability": score})
    return pd.DataFrame(rows, columns=["seriesuid", "coordX", "coordY", "coordZ", "probability"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--luna_root", default="data/luna16")
    p.add_argument("--subset", type=int, default=0)
    p.add_argument("--bundle_dir", default="models/bundles")
    p.add_argument("--out", default="runs/monai_detector")
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 例（試跑用）")
    p.add_argument("--skip_inference", action="store_true", help="只重算 FROC")
    a = p.parse_args()

    luna_root = Path(a.luna_root).resolve()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    bundle_root = Path(a.bundle_dir).resolve() / BUNDLE
    datalist = out / "datalist.json"
    uids = write_datalist(luna_root, a.subset, datalist, a.limit)
    result = out / "detections.json"

    if not a.skip_inference:
        if not (bundle_root / "models" / "model.pt").exists():
            run([sys.executable, "-m", "monai.bundle", "download", "--name", BUNDLE,
                 "--bundle_dir", str(bundle_root.parent)])
        run([sys.executable, "-m", "monai.bundle", "run",
             "--config_file", "configs/inference.json",
             "--bundle_root", str(bundle_root),
             "--whether_raw_luna16", "True",
             "--dataset_dir", str(luna_root),
             "--data_list_file_path", str(datalist),
             "--output_dir", str(out),
             "--output_filename", result.name], cwd=bundle_root)

    dets = load_detections(result)
    ann = pd.read_csv(luna_root / "annotations.csv")
    ann = ann[ann.seriesuid.isin(uids)]
    ex_path = luna_root / "annotations_excluded.csv"
    excluded = pd.read_csv(ex_path) if ex_path.exists() else None
    if excluded is not None:
        excluded = excluded[excluded.seriesuid.isin(uids)]
    dets.to_csv(out / "detections.csv", index=False)

    r = froc(dets, ann, n_scans=len(uids), excluded=excluded)
    r["n_scans"] = len(uids)
    (out / "froc.json").write_text(json.dumps(r, indent=2))
    print(json.dumps(r, indent=2))
    print(f"CPM = {r['cpm']:.3f}  (published LUNA16 detectors are around 0.85-0.90)")


if __name__ == "__main__":
    main()
