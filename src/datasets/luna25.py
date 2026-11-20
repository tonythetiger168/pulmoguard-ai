"""
luna25.py — LUNA25 結節惡性度資料集（官方 nodule blocks）

資料來源（Zenodo）：
  - 影像 14223624（CC BY 4.0）：luna25_nodule_blocks.zip.001/.002（約 6.8 GB）
  - 標註 14673658（CC BY-NC 4.0，僅限研究，不可用於商用產品訓練）：
    LUNA25_Public_Training_Development_Data.csv（6,163 顆結節，555 惡性）

nodule_blocks 目錄結構（與官方 baseline 相同）：
  image/{AnnotationID}.npy     以結節為中心的 HU 區塊，(z, y, x) = (64, 128, 128)，原始解析度
  metadata/{AnnotationID}.npy  dict：origin、spacing（z, y, x，mm）、transform

本模組把區塊重新取樣成等向性 patch（預設 64³、視野 50 mm），
訓練時加入隨機 3D 旋轉與平移；所有切分都以「病人」為單位，避免同一病人跨 fold 洩漏。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

HU_MIN, HU_MAX = -1000.0, 400.0
GROUP_COLS = ("PatientID", "patient_id", "PatientId", "pid")


def load_annotations(csv_path: str) -> pd.DataFrame:
    """讀官方 CSV，統一出 AnnotationID / label / group 三欄（group = 病人 ID，切分用）"""
    df = pd.read_csv(csv_path)
    missing = {"AnnotationID", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少欄位 {missing}；實際欄位：{list(df.columns)}")
    gcol = next((c for c in GROUP_COLS if c in df.columns), None)
    if gcol is None:
        gcol = "SeriesInstanceUID" if "SeriesInstanceUID" in df.columns else "AnnotationID"
        print(f"warning: CSV 沒有病人 ID 欄位，改用 {gcol} 分組（同病人多次掃描可能跨 fold）")
    df = df.copy()
    df["group"] = df[gcol].astype(str)
    df["label"] = df["label"].astype(int)
    df["AnnotationID"] = df["AnnotationID"].astype(str)
    return df


def filter_available(df: pd.DataFrame, blocks_dir: str) -> pd.DataFrame:
    """只保留 image 與 metadata 都存在的結節，並回報缺幾顆"""
    root = Path(blocks_dir)
    ok = df.AnnotationID.map(lambda a: (root / "image" / f"{a}.npy").exists()
                             and (root / "metadata" / f"{a}.npy").exists())
    if (~ok).any():
        print(f"warning: {int((~ok).sum())} / {len(df)} 顆結節找不到區塊檔，已略過")
    return df[ok].reset_index(drop=True)


def make_folds(df: pd.DataFrame, n_folds: int = 5, seed: int = 42) -> pd.DataFrame:
    """病人層級、分層（惡性比例相近）的 K-fold；回傳加上 fold 欄的 df"""
    df = df.copy()
    df["fold"] = -1
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, te) in enumerate(sgkf.split(df, df.label, groups=df.group)):
        df.loc[df.index[te], "fold"] = k
    return df


def inner_split(df: pd.DataFrame, frac: float = 0.15, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """從訓練 fold 再切出一小塊（病人層級、分層）作早停用驗證，測試 fold 完全不參與選模型"""
    n = max(2, round(1 / frac))
    sgkf = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)
    tr, va = next(sgkf.split(df, df.label, groups=df.group))
    return df.iloc[tr].reset_index(drop=True), df.iloc[va].reset_index(drop=True)


def _rotation(rng: np.random.Generator, max_deg: float) -> np.ndarray:
    a = np.deg2rad(rng.uniform(-max_deg, max_deg, 3))
    cx, cy, cz = np.cos(a)
    sx, sy, sz = np.sin(a)
    rz = np.array([[1, 0, 0], [0, cz, -sz], [0, sz, cz]])   # 繞 z 軸（作用在 y, x）
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[cx, -sx, 0], [sx, cx, 0], [0, 0, 1]])
    return rz @ ry @ rx


def resample_patch(block: np.ndarray, spacing_zyx, size_px: int = 64, size_mm: float = 50.0,
                   rot: np.ndarray | None = None, shift_mm=(0.0, 0.0, 0.0)) -> np.ndarray:
    """以區塊中心為中心，取 size_mm 視野、size_px³ 的等向性 patch（線性內插，界外填 -1000 HU）"""
    spacing = np.asarray(spacing_zyx, dtype=np.float64)
    out_sp = size_mm / size_px
    r = np.eye(3) if rot is None else rot
    matrix = np.diag(1.0 / spacing) @ r * out_sp            # 輸出 index → 輸入 voxel
    center_in = (np.asarray(block.shape) - 1) / 2.0 + np.asarray(shift_mm) / spacing
    center_out = np.full(3, (size_px - 1) / 2.0)
    offset = center_in - matrix @ center_out
    return ndi.affine_transform(np.asarray(block, dtype=np.float32), matrix, offset=offset,
                                output_shape=(size_px,) * 3, order=1, cval=HU_MIN, prefilter=False)


def hu_scale(x: np.ndarray) -> np.ndarray:
    return np.clip((x - HU_MIN) / (HU_MAX - HU_MIN), 0.0, 1.0)


class Luna25BlockDataset(Dataset):
    def __init__(self, df: pd.DataFrame, blocks_dir: str, size_px: int = 64, size_mm: float = 50.0,
                 augment: bool = False, max_rot_deg: float = 20.0, max_shift_mm: float = 2.5):
        self.df = df.reset_index(drop=True)
        self.root = Path(blocks_dir)
        self.size_px, self.size_mm = size_px, size_mm
        self.augment, self.max_rot, self.max_shift = augment, max_rot_deg, max_shift_mm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        block = np.load(self.root / "image" / f"{r.AnnotationID}.npy", mmap_mode="r")
        meta = np.load(self.root / "metadata" / f"{r.AnnotationID}.npy", allow_pickle=True).item()
        rot, shift = None, (0.0, 0.0, 0.0)
        if self.augment:
            rng = np.random.default_rng()   # 每次重新抽樣；worker 間種子由 torch 自動錯開
            rot = _rotation(rng, self.max_rot)
            v = rng.normal(size=3)
            shift = v / (np.linalg.norm(v) + 1e-8) * rng.uniform(0, self.max_shift)
        patch = resample_patch(block, meta["spacing"], self.size_px, self.size_mm, rot, shift)
        patch = hu_scale(patch)
        if self.augment:
            for ax in range(3):
                if np.random.rand() < 0.5:
                    patch = np.flip(patch, ax)
        return {"image": torch.from_numpy(np.ascontiguousarray(patch)[None]).float(),
                "label": torch.tensor(float(r.label)),
                "id": str(r.AnnotationID)}


def get_loaders(train_df, val_df, blocks_dir, size_px=64, size_mm=50.0, batch_size=32, num_workers=8,
                balanced=True):
    pin = torch.cuda.is_available()
    train_ds = Luna25BlockDataset(train_df, blocks_dir, size_px, size_mm, augment=True)
    sampler = None
    if balanced:   # 惡性只佔約 9%：每個 batch 以加權抽樣讓良惡性大致各半
        counts = train_df.label.value_counts().to_dict()
        w = train_df.label.map(lambda y: 1.0 / counts[y]).to_numpy()
        sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.double), len(w), replacement=True)
    kw = dict(num_workers=num_workers, pin_memory=pin, persistent_workers=num_workers > 0)
    return (
        DataLoader(train_ds, batch_size=batch_size, sampler=sampler, shuffle=sampler is None,
                   drop_last=True, **kw),
        DataLoader(Luna25BlockDataset(val_df, blocks_dir, size_px, size_mm), batch_size=batch_size,
                   shuffle=False, **kw),
    )
