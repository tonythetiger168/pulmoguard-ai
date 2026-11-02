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
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.eval.froc import froc

BUNDLE = "lung_nodule_ct_detection"
# 不走 `monai.bundle download`（會連 raw.githubusercontent.com，部分 RunPod 主機 IPv6 不通）：
# 設定檔用 git 從 github.com 取，權重從 NVIDIA CDN 取（URL 與 md5 來自 bundle 的 large_files.yml）
MODEL_ZOO = "https://github.com/Project-MONAI/model-zoo.git"
WEIGHTS_URL = ("https://developer.download.nvidia.com/assets/Clara/monai/tutorials/"
               "model_zoo/model_lung_nodule_ct_detection.pt")
WEIGHTS_MD5 = "b2e732a7c2786e2f8398fa45293b1e73"


def fetch_bundle(bundle_root: Path) -> None:
    """取得 bundle 設定與預訓練權重（可重複執行；已存在則跳過）"""
    if not (bundle_root / "configs" / "inference.json").exists():
        tmp = bundle_root.parent / "_model_zoo"
        if tmp.exists():
            shutil.rmtree(tmp)
        run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--sparse", MODEL_ZOO, str(tmp)])
        run(["git", "-C", str(tmp), "sparse-checkout", "set", f"models/{BUNDLE}"])
        bundle_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp / "models" / BUNDLE), str(bundle_root))
        shutil.rmtree(tmp)
    weights = bundle_root / "models" / "model.pt"
    if weights.exists() and _md5(weights) == WEIGHTS_MD5:
        return
    weights.parent.mkdir(parents=True, exist_ok=True)
    # -4：強制 IPv4；--retry：暫時性網路錯誤自動重試
    run(["curl", "-4", "-fL", "--retry", "5", "--retry-all-errors", "-o", str(weights), WEIGHTS_URL])
    if _md5(weights) != WEIGHTS_MD5:
        raise RuntimeError(f"md5 mismatch for {weights}")


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
        fetch_bundle(bundle_root)
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
    try:
        main()
    except Exception as e:  # 失敗原因寫進結果目錄，試跑結果裡看得到
        out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else Path("runs/monai_detector")
        out.mkdir(parents=True, exist_ok=True)
        (out / "error.txt").write_text(f"{type(e).__name__}: {e}\n")
        raise
