"""
lesion_metrics.py — 結節層級（lesion-level）評估：nnU-Net 的 Dice 是體素平均，小結節容易被大結節蓋過

每個 GT 連通區塊 = 一顆結節；預測連通區塊與之有任何重疊即算偵測到。
輸出：結節敏感度、每例假陽性數、偵測到的結節的平均 Dice、依直徑分層（<6 / 6–10 / >10 mm）的敏感度。

用法：
    python -m scripts.lesion_metrics --pred <fold_0/validation> --gt <nnUNet_raw/Dataset/labelsTr> --out m.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage


def case_metrics(pred: np.ndarray, gt: np.ndarray, spacing_zyx: np.ndarray) -> dict:
    s = ndimage.generate_binary_structure(3, 3)
    g_lab, ng = ndimage.label(gt > 0, s)
    p_lab, npred = ndimage.label(pred > 0, s)
    vox_ml = float(np.prod(spacing_zyx))
    lesions, hit_pred = [], set()
    for i in range(1, ng + 1):
        g = g_lab == i
        overl = np.unique(p_lab[g])
        overl = overl[overl > 0]
        hit_pred.update(int(x) for x in overl)
        p = np.isin(p_lab, overl) if len(overl) else np.zeros_like(g)
        inter = np.logical_and(g, p).sum()
        dice = 2 * inter / (g.sum() + p.sum()) if len(overl) else 0.0
        diam = (6 * g.sum() * vox_ml / np.pi) ** (1 / 3)  # 等體積球直徑（mm）
        lesions.append({"diam_mm": float(diam), "detected": bool(len(overl)), "dice": float(dice)})
    return {"lesions": lesions, "fp": int(npred - len(hit_pred))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows, fps = [], []
    preds = sorted(Path(args.pred).glob("*.nii.gz"))
    for pf in preds:
        gi = sitk.ReadImage(str(Path(args.gt, pf.name)))
        pi = sitk.ReadImage(str(pf))
        m = case_metrics(sitk.GetArrayFromImage(pi), sitk.GetArrayFromImage(gi),
                         np.array(gi.GetSpacing())[::-1])
        rows += [dict(l, case=pf.name) for l in m["lesions"]]
        fps.append(m["fp"])

    def sens(sel):
        return (sum(r["detected"] for r in sel) / len(sel)) if sel else None

    bins = {"<6mm": lambda d: d < 6, "6-10mm": lambda d: 6 <= d <= 10, ">10mm": lambda d: d > 10}
    out = {
        "n_cases": len(preds),
        "n_lesions": len(rows),
        "sensitivity": sens(rows),
        "fp_per_case": float(np.mean(fps)) if fps else None,
        "dice_detected_mean": float(np.mean([r["dice"] for r in rows if r["detected"]])) if rows else None,
        "sensitivity_by_size": {k: {"n": len([r for r in rows if f(r["diam_mm"])]),
                                    "sens": sens([r for r in rows if f(r["diam_mm"])])}
                                for k, f in bins.items()},
    }
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
