"""
nlst_prepare.py — 依 cohort.csv 從 IDC 分批下載 NLST CT，轉成 1 mm 等向性 int16 壓縮 .mhd，轉完立即刪 DICOM

一次只下載一批（--batch 個 series），磁碟最多同時放一批 DICOM（約 85 MB × batch）。
輸出 <out>/<SeriesInstanceUID>.mhd（+ .zraw），SimpleITK 的 LPS 世界座標，可直接給 MONAI 偵測器
（whether_raw_luna16=True，與 LUNA16 .mhd 相同處理）。已轉好的 series 會跳過，可中斷續跑。

用法：
    python -m scripts.nlst_prepare --cohort runs/nlst/cohort.csv --out data/nlst/mhd --workers 6
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

AIR_HU = -1000


def convert_series(dicom_dir: str, out_path: str, spacing_mm: float = 1.0) -> dict:
    import SimpleITK as sitk

    r = sitk.ImageSeriesReader()
    files = r.GetGDCMSeriesFileNames(dicom_dir)
    if len(files) < 20:
        raise RuntimeError(f"only {len(files)} DICOM files in {dicom_dir}")
    r.SetFileNames(files)
    img = sitk.Cast(r.Execute(), sitk.sitkFloat32)
    sp = np.array(img.GetSpacing())
    size = [int(round(n * s / spacing_mm)) for n, s in zip(img.GetSize(), sp)]
    res = sitk.Resample(img, size, sitk.Transform(), sitk.sitkLinear, img.GetOrigin(),
                        (spacing_mm,) * 3, img.GetDirection(), AIR_HU, sitk.sitkFloat32)
    res = sitk.Cast(sitk.Clamp(res, sitk.sitkFloat32, -1024, 3071), sitk.sitkInt16)
    sitk.WriteImage(res, out_path, useCompression=True)
    return {"n_slices": len(files), "spacing_in": [float(x) for x in sp], "size_out": size}


def _convert(args):
    uid, ddir, out = args
    try:
        dirs = [p for p in Path(ddir).rglob("*") if p.is_dir()] + [Path(ddir)]
        d = max(dirs, key=lambda p: sum(1 for f in p.iterdir() if f.is_file()))
        info = convert_series(str(d), str(Path(out, f"{uid}.mhd")))
        return {"SeriesInstanceUID": uid, "status": "ok", **info}
    except Exception as e:
        return {"SeriesInstanceUID": uid, "status": "error", "error": f"{type(e).__name__}: {e}"}
    finally:
        shutil.rmtree(ddir, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from idc_index import IDCClient
    c = IDCClient()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    uids = pd.read_csv(args.cohort, dtype=str).SeriesInstanceUID.tolist()
    if args.limit:
        uids = uids[: args.limit]
    todo = [u for u in uids if not (out / f"{u}.mhd").exists()]
    print(f"{len(uids)} series in cohort, {len(todo)} to prepare", flush=True)
    log_path = out / "prepare_log.jsonl"
    tmp_root = Path(tempfile.mkdtemp(prefix="nlst_dcm_", dir=out.parent))
    n_err = 0
    with ProcessPoolExecutor(args.workers) as ex:
        for b in range(0, len(todo), args.batch):
            batch = todo[b: b + args.batch]
            bdir = tmp_root / f"b{b:05d}"
            c.download_from_selection(seriesInstanceUID=batch, downloadDir=str(bdir),
                                      dirTemplate="%SeriesInstanceUID", show_progress_bar=False, quiet=True)
            jobs = [(u, str(bdir / u), str(out)) for u in batch]
            with open(log_path, "a") as log:
                for r in ex.map(_convert, jobs):
                    log.write(json.dumps(r) + "\n")
                    if r["status"] != "ok":
                        n_err += 1
                        print(f"  ERROR {r['SeriesInstanceUID']}: {r['error']}", flush=True)
            shutil.rmtree(bdir, ignore_errors=True)
            print(f"[{min(b + args.batch, len(todo))}/{len(todo)}] prepared, {n_err} errors", flush=True)
    shutil.rmtree(tmp_root, ignore_errors=True)
    if n_err > 0.1 * max(1, len(todo)):
        raise SystemExit(f"too many conversion errors: {n_err}")


if __name__ == "__main__":
    main()
