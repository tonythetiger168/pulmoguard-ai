"""
nlst_longitudinal_extract.py — NLST 縱向世代逐次掃描的 GPU 特徵抽取（每次掃描只上 GPU 一次）

依病人分批，每批：
  1) MONAI 偵測器（沿用 nlst_validate.detect_all，分批 + 二分排除壞掃描）
  2) 每次掃描取偵測分數 ≥ DET_MIN 的前 TOP_K 個候選，以候選中心切 64 mm（1 mm → 64³）小方塊
  3) LIDC ROI nnU-Net（release lidc-nnunet-20261103_0855，fold 0）分割 → 取包含或最靠近中心的連通區塊
     → 體積 mm³、長短徑、平均徑
  4) LUNA25 ResNet-18 5-fold 集成：惡性度機率 + 512 維特徵（fold 平均，fp16 存檔）
  5) 同一病人相鄰兩次檢查做剛體 → 仿射對位（CPU，SimpleITK，4 mm 降採樣），存「前次 → 本次」座標轉換
處理完的病人寫進 done.txt，可中斷續跑；處理完即刪掉該病人的 .mhd。

輸出（--out）：
  nodules.csv      每次掃描每個候選一列：PatientID, exam, exam_day, SeriesInstanceUID, x, y, z（LPS mm）,
                   det_prob, malignancy, volume, long_d, short_d, mean_d, seg_voxels, emb_row
  embeddings/*.npy 每批一個 float16 陣列（emb_row 指向「批次檔:列」）
  transforms.csv   PatientID, prior_exam, exam, metric, 12 個仿射參數（把前次 LPS 點映到本次）
  detector_failed.txt

用法（pod 內）：
    python -m scripts.nlst_longitudinal_extract --cohort runs/nlst_long/cohort.csv --mhd_dir data/nlst/mhd \
        --luna25_runs runs/luna25 --seg_model models/lidc_roi --out runs/nlst_long
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROI = 64            # 64 mm @ 1 mm
TOP_K = 10
DET_MIN = 0.3


def crop_cube(arr_zyx: np.ndarray, center_zyx, size: int = ROI, fill: float = -1000) -> np.ndarray:
    lo = [int(round(c - size / 2)) for c in center_zyx]
    out = np.full((size,) * 3, fill, dtype=arr_zyx.dtype)
    src = tuple(slice(max(a, 0), min(a + size, n)) for a, n in zip(lo, arr_zyx.shape))
    dst = tuple(slice(s.start - a, s.stop - a) for s, a in zip(src, lo))
    if all(s.stop > s.start for s in src):
        out[dst] = arr_zyx[src]
    return out


def central_component(seg: np.ndarray, max_dist_vox: float = 12.0) -> np.ndarray:
    """取包含中心、或質心最靠近中心（且在 max_dist_vox 內）的連通區塊；沒有就回傳全 0。"""
    from scipy import ndimage

    lab, n = ndimage.label(seg > 0)
    if n == 0:
        return np.zeros_like(seg, dtype=bool)
    c = tuple(s // 2 for s in seg.shape)
    if lab[c] > 0:
        return lab == lab[c]
    cents = np.array(ndimage.center_of_mass(seg > 0, lab, range(1, n + 1)))
    d = np.linalg.norm(cents - np.array(c), axis=1)
    k = int(np.argmin(d))
    return lab == (k + 1) if d[k] <= max_dist_vox else np.zeros_like(seg, dtype=bool)


def register_prior_to_current(prior_path: Path, cur_path: Path, shrink: int = 4) -> dict:
    """回傳把前次 LPS 點映到本次 LPS 點的仿射參數（SimpleITK AffineTransform）與最終 metric。"""
    import SimpleITK as sitk

    def prep(p):
        im = sitk.ReadImage(str(p), sitk.sitkFloat32)
        im = sitk.Clamp(im, sitk.sitkFloat32, -1000, 200)
        return sitk.Shrink(im, [shrink] * 3)

    fixed, moving = prep(cur_path), prep(prior_path)

    def run(tf):
        r = sitk.ImageRegistrationMethod()
        r.SetMetricAsMattesMutualInformation(32)
        r.SetMetricSamplingStrategy(r.RANDOM)
        r.SetMetricSamplingPercentage(0.2, seed=0)
        r.SetInterpolator(sitk.sitkLinear)
        r.SetOptimizerAsRegularStepGradientDescent(1.0, 1e-4, 200, relaxationFactor=0.5)
        r.SetOptimizerScalesFromPhysicalShift()
        r.SetShrinkFactorsPerLevel([2, 1])
        r.SetSmoothingSigmasPerLevel([1, 0])
        r.SetInitialTransform(tf, inPlace=True)
        r.Execute(fixed, moving)
        return float(r.GetMetricValue())

    rigid = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(),
                                              sitk.CenteredTransformInitializerFilter.GEOMETRY)
    rigid_metric = run(rigid)
    tf = sitk.AffineTransform(3)  # 第二階段：以剛體結果為起點做仿射
    tf.SetCenter(rigid.GetCenter())
    tf.SetMatrix(rigid.GetMatrix())
    tf.SetTranslation(rigid.GetTranslation())
    out = {"metric_rigid": rigid_metric, "metric": run(tf)}
    # tf：本次（fixed）點 → 前次（moving）點；要的是反方向
    inv = tf.GetInverse()
    a = sitk.AffineTransform(inv)
    m = np.array(a.GetMatrix()).reshape(3, 3)
    c = np.array(a.GetCenter())
    t = np.array(a.GetTranslation())
    # y = M (x − c) + c + t = M x + (c + t − M c)
    off = c + t - m @ c
    out.update({f"m{i}{j}": float(m[i, j]) for i in range(3) for j in range(3)})
    out.update({f"o{i}": float(off[i]) for i in range(3)})
    return out


def apply_affine(row: dict | pd.Series, pts_xyz: np.ndarray) -> np.ndarray:
    m = np.array([[row[f"m{i}{j}"] for j in range(3)] for i in range(3)])
    o = np.array([row[f"o{i}"] for i in range(3)])
    return np.asarray(pts_xyz, float) @ m.T + o


class Segmenter:
    def __init__(self, model_dir: str, device):
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        self.p = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
                                 device=device, verbose=False, verbose_preprocessing=False, allow_tqdm=False)
        self.p.initialize_from_trained_model_folder(model_dir, use_folds=(0,), checkpoint_name="checkpoint_final.pth")

    def __call__(self, cube_zyx: np.ndarray) -> np.ndarray:
        x = cube_zyx.astype(np.float32)[None]
        return np.asarray(self.p.predict_single_npy_array(x, {"spacing": [1.0, 1.0, 1.0]}, None, None, False))


def load_luna25(runs_dir: str, device):
    import glob

    import torch
    from monai.networks.nets import resnet18

    models = []
    for p in sorted(glob.glob(f"{runs_dir}/fold*/best.pt")):
        m = resnet18(spatial_dims=3, n_input_channels=1, num_classes=1)
        m.load_state_dict({k: v.float() for k, v in torch.load(p, map_location="cpu").items()})
        models.append(m.to(device).eval())
    if not models:
        raise FileNotFoundError(f"no fold*/best.pt under {runs_dir}")
    return models


def luna25_scores(models, cubes: list[np.ndarray], device) -> tuple[np.ndarray, np.ndarray]:
    """回傳 (惡性度機率 n, 512 維特徵 n×512)，皆為 fold 平均。輸入是 1 mm 的 64³ 小方塊。"""
    import torch

    from src.datasets.luna25 import hu_scale, resample_patch

    x = np.stack([hu_scale(resample_patch(c.astype(np.float32), np.ones(3), 64, 50.0)) for c in cubes])
    x = torch.from_numpy(x[:, None]).float().to(device)
    probs, feats = [], []
    with torch.no_grad():
        for m in models:
            fc = m.fc
            m.fc = torch.nn.Identity()
            f = m(x)
            m.fc = fc
            probs.append(torch.sigmoid(fc(f).squeeze(1)).cpu().numpy())
            feats.append(f.cpu().numpy())
    return np.mean(probs, 0), np.mean(feats, 0)


def process_scan(mhd: Path, dets: pd.DataFrame, seg, models, device) -> tuple[list[dict], np.ndarray]:
    import SimpleITK as sitk

    from src.eval.nodule_growth import mask_diameters_mm

    img = sitk.ReadImage(str(mhd))
    arr = sitk.GetArrayFromImage(img)
    d = dets[dets.probability >= DET_MIN].sort_values("probability", ascending=False).head(TOP_K)
    rows, cubes = [], []
    for r in d.itertuples():
        idx = np.array(img.TransformPhysicalPointToContinuousIndex((float(r.coordX), float(r.coordY), float(r.coordZ))))
        cube = crop_cube(arr, idx[::-1])
        cubes.append(cube)
        mask = central_component(seg(cube)) if seg is not None else np.zeros(cube.shape, bool)
        if mask.any():  # 用分割質心修正結節中心
            cz = np.array(np.nonzero(mask)).mean(1) - ROI / 2 + idx[::-1]
            x, y, z = img.TransformContinuousIndexToPhysicalPoint(tuple(float(v) for v in cz[::-1]))
        else:
            x, y, z = float(r.coordX), float(r.coordY), float(r.coordZ)
        long_d, short_d, mean_d = mask_diameters_mm(mask)
        rows.append({"x": x, "y": y, "z": z, "det_x": float(r.coordX), "det_y": float(r.coordY),
                     "det_z": float(r.coordZ), "det_prob": float(r.probability),
                     "volume": float(mask.sum()), "long_d": long_d, "short_d": short_d, "mean_d": mean_d,
                     "seg_voxels": int(mask.sum())})
    if not cubes:
        return [], np.zeros((0, 512), np.float16)
    mal, emb = luna25_scores(models, cubes, device)
    for row, m in zip(rows, mal):
        row["malignancy"] = float(m)
    return rows, emb.astype(np.float16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--mhd_dir", required=True)
    ap.add_argument("--luna25_runs", required=True)
    ap.add_argument("--seg_model", required=True, help="資料夾：plans.json、dataset.json、fold_0/checkpoint_final.pth")
    ap.add_argument("--bundle_dir", default="models/bundles")
    ap.add_argument("--out", required=True)
    ap.add_argument("--patients_per_batch", type=int, default=30)
    ap.add_argument("--wait_minutes", type=float, default=240, help="等背景下載轉檔的最長時間")
    ap.add_argument("--keep_mhd", action="store_true")
    ap.add_argument("--reg_workers", type=int, default=4)
    a = ap.parse_args()

    import torch

    from scripts.eval_monai_detector import BUNDLE, fetch_bundle, load_detections
    from scripts.nlst_validate import detect_all

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out).resolve()
    (out / "embeddings").mkdir(parents=True, exist_ok=True)
    mhd_dir = Path(a.mhd_dir).resolve()
    cohort = pd.read_csv(a.cohort, dtype={"PatientID": str, "SeriesInstanceUID": str})
    done_f = out / "done.txt"
    done = set(done_f.read_text().split()) if done_f.exists() else set()
    patients = [p for p in cohort.PatientID.drop_duplicates() if p not in done]
    print(f"{len(patients)} patients to process ({len(done)} already done)", flush=True)

    bundle_root = Path(a.bundle_dir).resolve() / BUNDLE
    fetch_bundle(bundle_root)
    seg = Segmenter(a.seg_model, device)
    models = load_luna25(a.luna25_runs, device)
    failed_f = out / "detector_failed.txt"
    prep_log = mhd_dir / "prepare_log.jsonl"
    reg_pool = ThreadPoolExecutor(a.reg_workers)

    def prep_status() -> dict:
        st = {}
        if prep_log.exists():
            for line in prep_log.read_text().splitlines():
                try:
                    r = json.loads(line)
                    st[r["SeriesInstanceUID"]] = r["status"]
                except Exception:
                    pass
        return st

    for b in range(0, len(patients), a.patients_per_batch):
        pts = patients[b: b + a.patients_per_batch]
        sub = cohort[cohort.PatientID.isin(pts)]
        t_wait = time.time()
        while True:  # 等這批的 .mhd 都轉好（或轉檔失敗）
            st = prep_status()
            if all(u in st or (mhd_dir / f"{u}.mhd").exists() for u in sub.SeriesInstanceUID):
                break
            if time.time() - t_wait > a.wait_minutes * 60:
                print("timeout waiting for prepared scans; continuing with what exists", flush=True)
                break
            time.sleep(20)
        sub = sub[sub.SeriesInstanceUID.map(lambda u: (mhd_dir / f"{u}.mhd").exists())]
        t0 = time.time()
        failed: list[str] = []
        items = detect_all(bundle_root, mhd_dir, out, list(sub.SeriesInstanceUID), failed)
        if failed:
            with open(failed_f, "a") as f:
                f.write("\n".join(failed) + "\n")
        det_json = out / "det_batch.json"
        det_json.write_text(json.dumps({"validation": items}))
        dets = load_detections(det_json) if items else pd.DataFrame(columns=["seriesuid"])
        t_det = time.time() - t0

        # 對位（CPU，背景執行緒）與 GPU 抽特徵同時進行
        reg_jobs = {}
        for pid, g in sub[~sub.SeriesInstanceUID.isin(failed)].groupby("PatientID"):
            g = g.sort_values("exam")
            for (_, pr), (_, cu) in zip(g.iloc[:-1].iterrows(), g.iloc[1:].iterrows()):
                reg_jobs[(pid, int(pr.exam), int(cu.exam))] = reg_pool.submit(
                    register_prior_to_current, mhd_dir / f"{pr.SeriesInstanceUID}.mhd",
                    mhd_dir / f"{cu.SeriesInstanceUID}.mhd")

        rows, embs = [], []
        n_emb = 0
        emb_name = f"emb_{b // a.patients_per_batch:05d}.npy"
        for r in sub[~sub.SeriesInstanceUID.isin(failed)].itertuples():
            d = dets[dets.seriesuid == r.SeriesInstanceUID]
            try:
                rr, ee = process_scan(mhd_dir / f"{r.SeriesInstanceUID}.mhd", d, seg, models, device)
            except Exception as e:
                print(f"  scan {r.SeriesInstanceUID} failed: {type(e).__name__}: {e}", flush=True)
                continue
            for k, row in enumerate(rr):
                row.update({"PatientID": r.PatientID, "exam": int(r.exam), "exam_day": r.exam_day,
                            "SeriesInstanceUID": r.SeriesInstanceUID, "emb_row": f"{emb_name}:{n_emb + k}"})
            n_emb += len(rr)
            rows += rr
            embs.append(ee)
        np.save(out / "embeddings" / emb_name, np.concatenate(embs) if embs else np.zeros((0, 512), np.float16))
        t_gpu = time.time() - t0 - t_det

        tf_rows = []
        for (pid, pe, ce), fut in reg_jobs.items():
            try:
                tf_rows.append({"PatientID": pid, "prior_exam": pe, "exam": ce, **fut.result()})
            except Exception as e:
                print(f"  registration {pid} {pe}->{ce} failed: {type(e).__name__}: {e}", flush=True)
        for name, data in (("nodules.csv", rows), ("transforms.csv", tf_rows)):
            if data:
                df = pd.DataFrame(data)
                p = out / name
                df.to_csv(p, mode="a", header=not p.exists(), index=False)
        with open(done_f, "a") as f:
            f.write("\n".join(pts) + "\n")
        if not a.keep_mhd:
            for u in sub.SeriesInstanceUID:
                for ext in (".mhd", ".zraw", ".raw"):
                    (mhd_dir / f"{u}{ext}").unlink(missing_ok=True)
        n_done = len(done) + b + len(pts)
        print(f"[{n_done}/{len(done) + len(patients)} patients] scans {len(sub)}, nodules {len(rows)}, "
              f"registrations {len(tf_rows)}/{len(reg_jobs)}, detector {t_det:.0f}s, seg+feat {t_gpu:.0f}s, "
              f"total {time.time() - t_wait:.0f}s", flush=True)
    reg_pool.shutdown()


if __name__ == "__main__":
    main()
