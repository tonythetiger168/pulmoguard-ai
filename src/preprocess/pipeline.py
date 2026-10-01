"""
preprocess.py — DICOM → NIfTI 統一前處理管線（L0–L3）
對應研發執行手冊 1.1 節管線
相依：dcm2niix, SimpleITK, numpy, pandas, pydicom
"""
import os
import subprocess
import numpy as np
import SimpleITK as sitk
import pandas as pd
from pathlib import Path

# ---------- 設定 ----------
HU_WINDOW = (-1000, 400)          # 肺窗
TARGET_SPACING = (1.0, 1.0, 1.0)  # mm，各向同性（依任務可改）
MAX_SLICE_THICKNESS = 2.5         # mm，超過即剔除
LUNA16_DIR = Path("data/luna16")
OUT_DIR = Path("data/processed")


def dicom_to_nifti(dicom_dir: str, out_dir: str) -> None:
    """L0-1：dcm2niix 批次轉檔（保留去識別化複查在轉檔前完成）"""
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run(
        ["dcm2niix", "-z", "y", "-f", "%p_%s", "-o", out_dir, dicom_dir],
        check=True,
    )


def load_ct(nifti_path: str) -> tuple[np.ndarray, tuple]:
    """讀取 CT 並回傳 (HU array, spacing)"""
    img = sitk.ReadImage(nifti_path)
    arr = sitk.GetArrayFromImage(img)  # (z, y, x)
    spacing = img.GetSpacing()         # (x, y, z)
    return arr, spacing


def filter_by_slice_thickness(spacing: tuple) -> bool:
    """層厚 >2.5mm 直接剔除（不重採樣，避免細節模糊）"""
    return spacing[2] <= MAX_SLICE_THICKNESS


def resample_isotropic(arr: np.ndarray, spacing: tuple,
                       target: tuple = TARGET_SPACING) -> np.ndarray:
    """L1：重採樣至各向同性 spacing"""
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing[::-1])  # SimpleITK 為 (x,y,z)
    rs = sitk.Resample(
        img,
        size=[int(round(arr.shape[i] * spacing[::-1][i] / target[i]))
              for i in range(3)],
        interpolator=sitk.sitkLinear,
        outputSpacing=target[::-1],
    )
    return sitk.GetArrayFromImage(rs)


def apply_hu_window(arr: np.ndarray,
                    window: tuple = HU_WINDOW) -> np.ndarray:
    """L2：HU 窗寬截斷後 min-max 至 (0,1)"""
    lo, hi = window
    arr = np.clip(arr, lo, hi)
    return (arr - lo) / (hi - lo)


# ---------- LUNA16 球形偽標籤（解決無 mask 問題，手冊 1.3-3） ----------
def sphere_labelmap(shape: tuple, center_xyz: tuple,
                    diameter_mm: float,
                    spacing: tuple = TARGET_SPACING) -> np.ndarray:
    """以結節中心座標＋直徑產生球形 label map
    center_xyz: (x, y, z) 物理座標（mm）
    """
    zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    cx, cy, cz = (c / s for c, s in zip(center_xyz, spacing))
    r = (diameter_mm / 2) / np.mean(spacing)
    dist2 = ((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2)
    return (dist2 <= r ** 2).astype(np.uint8)


def build_luna16_labels(csv_path: str, out_dir: str) -> None:
    """LUNA16 annotations.csv → 每例一顆球形 mask（.nii.gz）"""
    df = pd.read_csv(csv_path)
    os.makedirs(out_dir, exist_ok=True)
    for series, grp in df.groupby("seriesuid"):
        # 實務上需先載入該 series 的 CT 取得 shape/spacing
        # 此處僅示範群組邏輯，mask 產生整合於主流程
        pass  # 見主流程 process_all()


# ---------- LIDC XML → mask（分割任務用） ----------
def parse_lidc_xml(xml_path: str) -> list[dict]:
    """解析 LIDC-IDRI XML，回傳 nodule 清單（含 malignancy 1-5 序數）"""
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)
    nodules = []
    for roi in tree.iter("unblindedReadNodule"):
        nodule = {"label": roi.findtext("noduleID"),
                  "malignancy": roi.findtext("malignancy")}
        # 像素級 nodal 標註在 <roi> 內；bounding sphere 於 <roi> 外層
        for seg in roi.iter("roi"):
            nodule.setdefault("slices", []).append({
                "z": float(seg.findtext("imageZposition")),
                "xy": [(int(e.findtext("xCoord")),
                        int(e.findtext("yCoord")))
                       for e in seg.findall("edgeMap")],
            })
        nodules.append(nodule)
    return nodules


def lidc_consensus(nodules: list[dict], n_readers: int = 4) -> list[dict]:
    """標註共識規則：>=2 位同意為 positive；此處以 malignancy 欄位示意
    實務做法：跨 4 份 XML 統計每顆結節被標註次數"""
    # 完整實作需聚合同一 series 的 4 位醫師 XML
    return [n for n in nodules if n.get("malignancy") is not None]


# ---------- 主流程 ----------
def process_all(series_dirs: list[str]) -> None:
    for d in series_dirs:
        nii = dicom_to_nifti(d, str(OUT_DIR / "nii"))
        # 讀取、層厚過濾、重採樣、HU 窗 → 存 npz/nifti
        # 標籤依資料來源（LUNA16 sphere / LIDC pixel-wise）對齊
        ...


if __name__ == "__main__":
    process_all([])
