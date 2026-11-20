"""
lidc_to_nnunet.py — LIDC-IDRI（DICOM + XML 標註）→ nnU-Net v2 資料集（結節二元分割）

標註來源：pylidc 內建的 LIDC XML 資料庫（不必另外下載 XML）。
遮罩規則：同一顆結節（pylidc cluster_annotations）至少 MIN_READERS 位放射科醫師標註，
         再取 50% 共識（consensus clevel=0.5）。只有 1 位醫師標的結節不列入（視為背景）。
影像：保留原始 spacing（nnU-Net 自己會規劃重採樣），HU int16。
切分：以病人分組的 5-fold（同一病人的多次掃描不會跨 train/val），寫成 splits_final.json。

DICOM 目錄結構需為 <dicom_root>/<PatientID>/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm
（IDC 下載時用 --dir-template '%PatientID/%StudyInstanceUID/%SeriesInstanceUID'）

用法：
    python -m scripts.lidc_to_nnunet --dicom_root /workspace/data/lidc_dicom \
        --out /workspace/nnunet/nnUNet_raw/Dataset501_LIDCNodule --workers 6
"""
from __future__ import annotations

import argparse
import configparser
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# pylidc 0.2.3 仍用已移除的 numpy 別名
for _alias, _t in (("int", int), ("float", float), ("bool", bool)):
    if not hasattr(np, _alias):
        setattr(np, _alias, _t)
# pylidc 用了 Python 3.12 移除的 SafeConfigParser
if not hasattr(configparser, "SafeConfigParser"):
    configparser.SafeConfigParser = configparser.ConfigParser

MIN_READERS = 2
CLEVEL = 0.5


def write_pylidcrc(dicom_root: str) -> None:
    rc = Path.home() / ".pylidcrc"
    rc.write_text(f"[dicom]\npath = {dicom_root}\nwarn = False\n")


def convert_one(scan_id: int, out: str) -> dict:
    import pydicom
    import pylidc as pl
    import SimpleITK as sitk
    from pylidc.utils import consensus

    scan = pl.query(pl.Scan).filter(pl.Scan.id == scan_id).first()
    case = f"{scan.patient_id.replace('LIDC-IDRI-', 'LIDC')}_{scan.id:04d}"
    img_p = Path(out, "imagesTr", f"{case}_0000.nii.gz")
    lab_p = Path(out, "labelsTr", f"{case}.nii.gz")
    info = {"case": case, "patient_id": scan.patient_id, "scan_id": scan.id,
            "series_uid": scan.series_instance_uid}
    if img_p.exists() and lab_p.exists():
        info["status"] = "exists"
        return info

    vol = scan.to_volume(verbose=False)                # (rows=y, cols=x, slices=z)，z 由小到大
    mask = np.zeros(vol.shape, dtype=np.uint8)
    n_nod = 0
    for anns in scan.cluster_annotations(verbose=False):
        if len(anns) < MIN_READERS:
            continue
        cmask, cbbox = consensus(anns, clevel=CLEVEL, pad=None, ret_masks=False)
        mask[cbbox][cmask] = 1
        n_nod += 1

    # 幾何：x/y 原點取自 DICOM，z 取 pylidc 排序後的第一張
    path = scan.get_path_to_dicom_files()
    first = next(f for f in sorted(os.listdir(path)) if f.endswith(".dcm"))
    ipp = pydicom.dcmread(os.path.join(path, first), stop_before_pixels=True).ImagePositionPatient
    spacing = (float(scan.pixel_spacing), float(scan.pixel_spacing), float(scan.slice_spacing))
    origin = (float(ipp[0]), float(ipp[1]), float(scan.slice_zvals[0]))

    for arr, p in ((vol.astype(np.int16), img_p), (mask, lab_p)):
        im = sitk.GetImageFromArray(np.ascontiguousarray(arr.transpose(2, 0, 1)))  # → (z, y, x)
        im.SetSpacing(spacing)
        im.SetOrigin(origin)
        sitk.WriteImage(im, str(p) + ".tmp.nii.gz", useCompression=True)
        os.replace(str(p) + ".tmp.nii.gz", p)

    info.update(status="ok", n_nodules=n_nod, voxels=int(mask.sum()),
                shape_zyx=[int(vol.shape[2]), int(vol.shape[0]), int(vol.shape[1])],
                spacing_xyz=spacing, slice_thickness=float(scan.slice_thickness))
    return info


def _safe(scan_id: int, out: str) -> dict:
    try:
        return convert_one(scan_id, out)
    except Exception as e:  # 單一掃描壞掉不影響整批
        return {"scan_id": scan_id, "status": "error", "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc(limit=3)}


def patient_folds(cases: list[dict], k: int = 5, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    pids = sorted({c["patient_id"] for c in cases})
    rng.shuffle(pids)
    fold_of = {p: i % k for i, p in enumerate(pids)}
    splits = []
    for f in range(k):
        val = sorted(c["case"] for c in cases if fold_of[c["patient_id"]] == f)
        tr = sorted(c["case"] for c in cases if fold_of[c["patient_id"]] != f)
        splits.append({"train": tr, "val": val})
    return splits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicom_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="只轉前 N 個掃描（試跑）")
    args = ap.parse_args()

    write_pylidcrc(args.dicom_root)
    import pylidc as pl

    for sub in ("imagesTr", "labelsTr"):
        Path(args.out, sub).mkdir(parents=True, exist_ok=True)
    have = set(os.listdir(args.dicom_root))
    scans = [s.id for s in pl.query(pl.Scan).order_by(pl.Scan.id) if s.patient_id in have]
    if args.limit:
        scans = scans[: args.limit]
    print(f"pylidc scans with DICOM present: {len(scans)}", flush=True)

    results = []
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(_safe, i, args.out) for i in scans]
        for n, f in enumerate(as_completed(futs), 1):
            r = f.result()
            results.append(r)
            if r["status"] == "error":
                print(f"  ERROR scan {r['scan_id']}: {r['error']}", flush=True)
            if n % 50 == 0 or n == len(futs):
                print(f"[{n}/{len(futs)}] converted", flush=True)

    ok = [r for r in results if r["status"] in ("ok", "exists")]
    err = [r for r in results if r["status"] == "error"]
    # exists 的案例沒有 patient_id 以外的統計，但切分只需要 case / patient_id
    json.dump(results, open(Path(args.out, "conversion_log.json"), "w"), indent=1)
    json.dump({
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "nodule": 1},
        "numTraining": len(ok),
        "file_ending": ".nii.gz",
        "name": "LIDC-IDRI nodule segmentation",
        "description": f"pylidc consensus clevel={CLEVEL}, nodules with >= {MIN_READERS} readers",
    }, open(Path(args.out, "dataset.json"), "w"), indent=1)
    json.dump(patient_folds(ok), open(Path(args.out, "splits_final.json"), "w"), indent=1)
    n_nod = sum(r.get("n_nodules", 0) for r in ok)
    print(f"done: {len(ok)} ok, {len(err)} errors, {n_nod} nodules (>= {MIN_READERS} readers)")
    if len(ok) == 0 or len(err) > 0.1 * max(1, len(results)):
        sys.exit("too many conversion errors, stopping before GPU training")


if __name__ == "__main__":
    main()
