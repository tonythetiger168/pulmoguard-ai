"""
lidc_crops.py — 把全肺 LIDC nnU-Net 資料集切成「以結節為中心」的 ROI 資料集（偵測 → 分割的第二段）

背景：2026-11-02/03 兩次全肺 3d_fullres 訓練（含正式版 torch、關掉 compile）pseudo Dice 都在
第 17 epoch 後歸零、loss 停在 -0.2。結節只佔全肺體素約 3e-5，全肺分割學不起來；
實務流程本來就是先用偵測器（MONAI 預訓練，LUNA16 CPM 0.888）找候選，再對 ROI 分割。

每顆 GT 結節（連通區塊）產生一個 ROI：
  - 物理大小 ROI_MM（預設 64 mm 立方），保留原始 spacing（nnU-Net 自己規劃重採樣）
  - 中心 = 結節質心 + 每軸 ±JITTER_MM 的固定亂數位移（模擬偵測器定位誤差，避免結節永遠在正中央）
  - 標籤 = ROI 內所有結節（鄰近結節也保留，不會變成「背景」）
  - 超出影像邊界的部分補 -1000 HU / 標籤 0
切分沿用病人層級 5-fold（同一病人的所有 ROI 在同一 fold）。

用法：
    python -m scripts.lidc_crops --src <nnUNet_raw>/Dataset501_LIDCNodule \
        --out <nnUNet_raw>/Dataset503_LIDCNoduleROI --workers 8
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from scripts.lidc_to_nnunet import patient_folds

ROI_MM = 64.0
JITTER_MM = 10.0


def crop_box(center_zyx, size_zyx):
    lo = [int(round(c - s / 2)) for c, s in zip(center_zyx, size_zyx)]
    return [(lo_i, lo_i + s) for lo_i, s in zip(lo, size_zyx)]


def extract(arr: np.ndarray, box, fill) -> np.ndarray:
    out = np.full([b - a for a, b in box], fill, dtype=arr.dtype)
    src = tuple(slice(max(a, 0), min(b, n)) for (a, b), n in zip(box, arr.shape))
    dst = tuple(slice(s.start - a, s.stop - a) for s, (a, _) in zip(src, box))
    out[dst] = arr[src]
    return out


def crop_case(case: str, src: str, out: str, roi_mm: float, jitter_mm: float, seed: int) -> list[dict]:
    import SimpleITK as sitk
    from scipy import ndimage

    li = sitk.ReadImage(str(Path(src, "labelsTr", f"{case}.nii.gz")))
    lab = sitk.GetArrayFromImage(li)
    if lab.max() == 0:
        return []
    ii = sitk.ReadImage(str(Path(src, "imagesTr", f"{case}_0000.nii.gz")))
    img = sitk.GetArrayFromImage(ii)
    sp_zyx = np.array(ii.GetSpacing())[::-1]
    size = [int(round(roi_mm / s)) for s in sp_zyx]
    comps, n = ndimage.label(lab > 0, ndimage.generate_binary_structure(3, 3))
    cents = ndimage.center_of_mass(lab > 0, comps, range(1, n + 1))
    rng = np.random.default_rng(seed + int.from_bytes(case.encode(), "little") % 100000)
    rows = []
    for j, c in enumerate(cents):
        c = np.array(c) + rng.uniform(-jitter_mm, jitter_mm, 3) / sp_zyx
        box = crop_box(c, size)
        name = f"{case}_n{j:02d}"
        origin = ii.TransformContinuousIndexToPhysicalPoint([float(box[2][0]), float(box[1][0]), float(box[0][0])])
        for arr, fill, p in ((img, -1000, Path(out, "imagesTr", f"{name}_0000.nii.gz")),
                             (lab, 0, Path(out, "labelsTr", f"{name}.nii.gz"))):
            im = sitk.GetImageFromArray(extract(arr, box, fill))
            im.SetSpacing(ii.GetSpacing())
            im.SetDirection(ii.GetDirection())
            im.SetOrigin(origin)
            sitk.WriteImage(im, str(p), useCompression=True)
        rows.append({"case": name, "source_case": case, "patient_id": case.split("_")[0]})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--roi_mm", type=float, default=ROI_MM)
    ap.add_argument("--jitter_mm", type=float, default=JITTER_MM)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    for sub in ("imagesTr", "labelsTr"):
        Path(args.out, sub).mkdir(parents=True, exist_ok=True)
    cases = sorted(p.name[: -len(".nii.gz")] for p in Path(args.src, "labelsTr").glob("*.nii.gz"))
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(crop_case, cases, [args.src] * len(cases), [args.out] * len(cases),
                          [args.roi_mm] * len(cases), [args.jitter_mm] * len(cases),
                          [args.seed] * len(cases), chunksize=4))
    rows = [r for rr in res for r in rr]
    print(f"{len(rows)} nodule ROIs from {sum(1 for r in res if r)} scans "
          f"({len({r['patient_id'] for r in rows})} patients)", flush=True)
    json.dump(rows, open(Path(args.out, "rois.json"), "w"), indent=1)
    json.dump({
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "nodule": 1},
        "numTraining": len(rows),
        "file_ending": ".nii.gz",
        "name": "LIDC-IDRI nodule ROI segmentation",
        "description": f"{args.roi_mm:g} mm ROIs around each consensus nodule, +-{args.jitter_mm:g} mm jitter",
    }, open(Path(args.out, "dataset.json"), "w"), indent=1)
    json.dump(patient_folds(rows), open(Path(args.out, "splits_final.json"), "w"), indent=1)
    if not rows:
        raise SystemExit("no ROIs produced")


if __name__ == "__main__":
    main()
