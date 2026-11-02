"""
pipeline.py — LDCT 前處理管線（L0–L3）
對應研發執行手冊 1.1 節管線

LUNA16 主流程（mhd/raw）：
    讀取 → 層厚過濾 → 各向同性重採樣（保留 origin/direction）→ HU 肺窗 →
    存成 float16 .npy（可 mmap 快速裁切）＋ meta.csv（origin/spacing/direction/subset）

用法：
    python -m src.preprocess.pipeline --luna_root data/luna16 --out data/processed \
        --subsets 0 --workers 8
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import SimpleITK as sitk

# ---------- 設定 ----------
HU_WINDOW = (-1000, 400)          # 肺窗
AIR_HU = -1000                    # 重採樣邊界填充值（空氣，而非 0 HU 的水）
TARGET_SPACING = (1.0, 1.0, 1.0)  # mm，(x, y, z) 各向同性
MAX_SLICE_THICKNESS = 2.5         # mm，超過即剔除
META_FIELDS = ["seriesuid", "subset", "shape_zyx", "origin_xyz",
               "spacing_xyz", "direction"]


def dicom_to_nifti(dicom_dir: str, out_dir: str) -> None:
    """L0-1：dcm2niix 批次轉檔（去識別化複查須在轉檔前完成）"""
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run(
        ["dcm2niix", "-z", "y", "-ba", "y", "-f", "%p_%s", "-o", out_dir, dicom_dir],
        check=True,
    )


def load_ct(path: str) -> sitk.Image:
    """讀取 CT（.mhd / .nii.gz / .nrrd），回傳 SimpleITK image（保留幾何資訊）"""
    return sitk.ReadImage(str(path))


def filter_by_slice_thickness(spacing_xyz: tuple) -> bool:
    """層厚 >2.5mm 直接剔除（不重採樣，避免細節模糊）。spacing 為 (x, y, z)"""
    return spacing_xyz[2] <= MAX_SLICE_THICKNESS


def resample_isotropic(img: sitk.Image,
                       target_xyz: tuple = TARGET_SPACING,
                       default_value: float = AIR_HU) -> sitk.Image:
    """L1：重採樣至目標 spacing，保留 origin 與 direction。

    SimpleITK 的 spacing/size 一律為 (x, y, z)；numpy 陣列為 (z, y, x)。
    """
    in_spacing = np.array(img.GetSpacing(), dtype=float)
    in_size = np.array(img.GetSize(), dtype=float)
    target = np.array(target_xyz, dtype=float)
    out_size = [int(round(s)) for s in in_size * in_spacing / target]

    f = sitk.ResampleImageFilter()
    f.SetOutputSpacing(tuple(target))
    f.SetSize(out_size)
    f.SetOutputOrigin(img.GetOrigin())
    f.SetOutputDirection(img.GetDirection())
    f.SetInterpolator(sitk.sitkLinear)
    f.SetDefaultPixelValue(default_value)
    f.SetOutputPixelType(sitk.sitkFloat32)
    return f.Execute(img)


def apply_hu_window(arr: np.ndarray,
                    window: tuple = HU_WINDOW) -> np.ndarray:
    """L2：HU 窗寬截斷後 min-max 至 (0,1)"""
    lo, hi = window
    arr = np.clip(arr.astype(np.float32), lo, hi)
    return (arr - lo) / (hi - lo)


def world_to_voxel(xyz_mm, origin_xyz, spacing_xyz, direction=None) -> np.ndarray:
    """世界座標（mm, x/y/z，LUNA16 coordX/Y/Z）→ numpy 索引 (z, y, x)（float）"""
    origin = np.asarray(origin_xyz, dtype=float)
    spacing = np.asarray(spacing_xyz, dtype=float)
    d = np.eye(3) if direction is None else np.asarray(direction, dtype=float).reshape(3, 3)
    ijk = np.linalg.solve(d * spacing, np.asarray(xyz_mm, dtype=float) - origin)
    return ijk[::-1]


# ---------- LUNA16 球形偽標籤（解決無 mask 問題，手冊 1.3-3） ----------
def sphere_labelmap(shape: tuple, center_xyz: tuple,
                    diameter_mm: float,
                    spacing: tuple = TARGET_SPACING,
                    origin: tuple = (0.0, 0.0, 0.0)) -> np.ndarray:
    """以結節中心座標＋直徑產生球形 label map（shape 為 (z, y, x)）
    center_xyz: (x, y, z) 物理座標（mm）；spacing/origin 為 (x, y, z)
    """
    cz, cy, cx = world_to_voxel(center_xyz, origin, spacing)
    sx, sy, sz = spacing
    zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    dist2 = ((xx - cx) * sx) ** 2 + ((yy - cy) * sy) ** 2 + ((zz - cz) * sz) ** 2
    return (dist2 <= (diameter_mm / 2) ** 2).astype(np.uint8)


# ---------- LIDC XML（分割／惡性度任務用；正式流程建議改用 pylidc） ----------
def parse_lidc_xml(xml_path: str) -> list[dict]:
    """解析單一 LIDC-IDRI XML（http://www.nih.gov 命名空間），回傳每位讀片者的 nodule 清單"""
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)
    nodules = []
    for roi in tree.iter("{*}unblindedReadNodule"):
        mal = roi.findtext("{*}characteristics/{*}malignancy")
        nodule = {"id": roi.findtext("{*}noduleID"),
                  "malignancy": int(mal) if mal else None,
                  "slices": []}
        for seg in roi.iter("{*}roi"):
            nodule["slices"].append({
                "z": float(seg.findtext("{*}imageZposition")),
                "xy": [(int(e.findtext("{*}xCoord")), int(e.findtext("{*}yCoord")))
                       for e in seg.findall("{*}edgeMap")],
            })
        nodules.append(nodule)
    return nodules


# ---------- LUNA16 主流程 ----------
def preprocess_series(mhd_path: str, out_dir: str, subset: int) -> dict | None:
    """單一 series：讀取 → 層厚過濾 → 重採樣 → HU 窗 → float16 .npy。已存在則跳過（可續跑）"""
    uid = Path(mhd_path).stem
    npy = Path(out_dir) / f"{uid}.npy"
    meta_json = Path(out_dir) / f"{uid}.json"
    if npy.exists() and meta_json.exists():
        return json.loads(meta_json.read_text())

    img = load_ct(mhd_path)
    if not filter_by_slice_thickness(img.GetSpacing()):
        return None
    rs = resample_isotropic(img)
    arr = apply_hu_window(sitk.GetArrayFromImage(rs)).astype(np.float16)

    tmp = npy.with_suffix(".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, npy)
    meta = {
        "seriesuid": uid,
        "subset": subset,
        "shape_zyx": list(arr.shape),
        "origin_xyz": list(rs.GetOrigin()),
        "spacing_xyz": list(rs.GetSpacing()),
        "direction": list(rs.GetDirection()),
    }
    meta_json.write_text(json.dumps(meta))
    return meta


def write_meta_csv(metas: list[dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=META_FIELDS)
        w.writeheader()
        for m in metas:
            w.writerow({k: json.dumps(m[k]) if isinstance(m[k], list) else m[k]
                        for k in META_FIELDS})


def process_luna16(luna_root: str, out_dir: str, subsets: list[int],
                   workers: int = 4) -> list[dict]:
    """處理 luna_root/subsetN/*.mhd，輸出 out_dir/npy/*.npy 與 out_dir/meta.csv"""
    npy_dir = Path(out_dir) / "npy"
    npy_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(str(p), str(npy_dir), s)
            for s in subsets
            for p in sorted((Path(luna_root) / f"subset{s}").glob("*.mhd"))]
    if not jobs:
        raise FileNotFoundError(f"no .mhd files under {luna_root}/subset{subsets}")

    metas, skipped = [], 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, m in enumerate(ex.map(preprocess_series, *zip(*jobs)), 1):
            if m is None:
                skipped += 1
            else:
                metas.append(m)
            if i % 10 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] processed, {skipped} skipped (slice > {MAX_SLICE_THICKNESS} mm)")
    write_meta_csv(metas, str(Path(out_dir) / "meta.csv"))
    return metas


def _container_limits() -> tuple[int, float | None]:
    """容器實際可用的 CPU 數與記憶體（GB）。nproc／os.cpu_count() 在 RunPod 等容器內常回報整台主機的核心數。"""
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    try:  # cgroup v2 CPU quota
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            cpus = min(cpus, max(1, int(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    mem_gb = None
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = open(path).read().strip()
            if v != "max" and int(v) < 1 << 50:
                mem_gb = int(v) / 2**30
                break
        except (OSError, ValueError):
            continue
    if mem_gb is None and hasattr(os, "sysconf"):
        try:
            mem_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES") / 2**30
        except (OSError, ValueError):
            pass
    return cpus, mem_gb


GB_PER_WORKER = 4.0   # 單一 CT 讀入＋重採樣＋float32 暫存的保守峰值


def default_workers() -> int:
    """依容器 CPU 與記憶體決定平行數，上限 8，避免 OOM（BrokenProcessPool）"""
    cpus, mem_gb = _container_limits()
    by_mem = int(mem_gb // GB_PER_WORKER) if mem_gb else 4
    return max(1, min(cpus, by_mem, 8))


def main() -> None:
    p = argparse.ArgumentParser(description="LUNA16 前處理（mhd → 1mm float16 npy）")
    p.add_argument("--luna_root", default="data/luna16")
    p.add_argument("--out", default="data/processed")
    p.add_argument("--subsets", type=int, nargs="+", default=[0],
                   help="要處理的 subset 編號；先用 0 做最小試跑")
    p.add_argument("--workers", type=int, default=0,
                   help="平行數；0 = 依容器 CPU／記憶體自動決定（上限 8）")
    a = p.parse_args()
    workers = a.workers or default_workers()
    print(f"workers: {workers}")
    metas = process_luna16(a.luna_root, a.out, a.subsets, workers)
    print(f"done: {len(metas)} series → {a.out}/meta.csv")


if __name__ == "__main__":
    main()
