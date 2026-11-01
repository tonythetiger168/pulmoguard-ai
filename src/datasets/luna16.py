"""
luna16.py — LUNA16 假陽性削減資料集（L4 patch 切片 + L5 增強）
對應研發執行手冊 1.1 節 L4/L5
相依：monai, torch, numpy, pandas, scikit-learn

輸入為 src.preprocess.pipeline 產出的 meta.csv 與 npy/*.npy（float16，已 HU 窗正規化）。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from monai.transforms import Compose, RandAffined, RandFlipd, RandGaussianNoised, RandRotate90d
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset

from src.preprocess.pipeline import world_to_voxel

PATCH_SIZE = (64, 64, 64)   # 64³ 比 96³ 少約 3.4 倍運算量，FP 削減足夠
NEG_PER_POS = 3             # 正:負 = 1:3（手冊 1.1）


def load_meta(meta_csv: str) -> pd.DataFrame:
    meta = pd.read_csv(meta_csv)
    for col in ("shape_zyx", "origin_xyz", "spacing_xyz", "direction"):
        meta[col] = meta[col].map(json.loads)
    return meta.set_index("seriesuid")


def make_candidate_df(candidates_csv: str, meta: pd.DataFrame,
                      neg_per_pos: int = NEG_PER_POS, seed: int = 42) -> pd.DataFrame:
    """candidates_V2.csv（class 1 = 真結節）→ 只保留已前處理的 series，負例降採樣。

    只用 candidates_V2 作標籤來源；不另外合併 annotations.csv，避免同一結節重複計為正例。
    欄位：seriesuid, coordX, coordY, coordZ, label, subset
    """
    cand = pd.read_csv(candidates_csv)
    cand = cand[cand.seriesuid.isin(meta.index)].copy()
    cand["label"] = (cand["class"] == 1).astype(int)
    cand["subset"] = cand.seriesuid.map(meta["subset"])
    pos = cand[cand.label == 1]
    neg = cand[cand.label == 0]
    neg = neg.sample(n=min(len(neg), len(pos) * neg_per_pos), random_state=seed)
    out = pd.concat([pos, neg]).sample(frac=1.0, random_state=seed)
    return out[["seriesuid", "coordX", "coordY", "coordZ", "label", "subset"]].reset_index(drop=True)


def split_by_series(df: pd.DataFrame, val_subset: int | None = None,
                    val_frac: float = 0.2, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """病人層級切分：同一 seriesuid 不會同時出現在 train 與 val。

    - 有多個 subset 且指定 val_subset → 使用 LUNA16 官方 fold（val = 該 subset）
    - 只有單一 subset（最小試跑）→ 依 seriesuid 分組隨機切出 val_frac
    """
    if val_subset is not None and df.subset.nunique() > 1:
        va = df.subset == val_subset
        return df[~va].reset_index(drop=True), df[va].reset_index(drop=True)
    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    tr_idx, va_idx = next(gss.split(df, groups=df.seriesuid))
    return df.iloc[tr_idx].reset_index(drop=True), df.iloc[va_idx].reset_index(drop=True)


def crop_pad(arr: np.ndarray, center_zyx, size_zyx, fill: float = 0.0) -> np.ndarray:
    """以 center 為中心裁切 size，超出邊界處以 fill 補齊（中心點永遠在 patch 正中央）"""
    out = np.full(size_zyx, fill, dtype=np.float32)
    src, dst = [], []
    for c, s, n in zip(center_zyx, size_zyx, arr.shape):
        lo = int(round(c)) - s // 2
        a0, a1 = max(lo, 0), min(lo + s, n)
        if a1 <= a0:
            return out
        src.append(slice(a0, a1))
        dst.append(slice(a0 - lo, a1 - lo))
    out[tuple(dst)] = arr[tuple(src)]
    return out


class Luna16PatchDataset(Dataset):
    """以候選點為中心的 3D patch 分類資料集（真結節 vs 假陽性）。

    CT 以 mmap 方式讀取，只載入需要的 patch；隨機增強每次 __getitem__ 重新抽樣（不快取）。
    """

    def __init__(self, df: pd.DataFrame, meta: pd.DataFrame, npy_dir: str,
                 patch_size: tuple = PATCH_SIZE, transform=None):
        self.df = df.reset_index(drop=True)
        self.meta = meta
        self.npy_dir = npy_dir
        self.patch_size = tuple(patch_size)
        self.transform = transform
        self._cache: dict[str, np.ndarray] = {}

    def __len__(self):
        return len(self.df)

    def _volume(self, uid: str) -> np.ndarray:
        if uid not in self._cache:
            self._cache[uid] = np.load(f"{self.npy_dir}/{uid}.npy", mmap_mode="r")
        return self._cache[uid]

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        m = self.meta.loc[r.seriesuid]
        center = world_to_voxel((r.coordX, r.coordY, r.coordZ),
                                m.origin_xyz, m.spacing_xyz, m.direction)
        patch = crop_pad(self._volume(r.seriesuid), center, self.patch_size)
        sample = {"image": patch[None], "label": int(r.label)}
        if self.transform:
            sample = self.transform(sample)
        return {
            "image": torch.as_tensor(np.asarray(sample["image"]), dtype=torch.float32),
            "label": torch.tensor(sample["label"], dtype=torch.long),
        }


def get_train_transforms():
    """L5 增強：小角度旋轉／縮放 ±10%、90° 旋轉、翻轉、高斯雜訊"""
    return Compose([
        RandAffined(keys=["image"], prob=0.5,
                    rotate_range=(0.1, 0.1, 0.1),
                    scale_range=(0.1, 0.1, 0.1),
                    padding_mode="zeros"),
        RandRotate90d(keys=["image"], prob=0.5, max_k=3, spatial_axes=(1, 2)),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
        RandGaussianNoised(keys=["image"], prob=0.3, std=0.01),
    ])


def get_loaders(train_df, val_df, meta, npy_dir, patch_size=PATCH_SIZE,
                batch_size=16, num_workers=4):
    pin = torch.cuda.is_available()
    train_ds = Luna16PatchDataset(train_df, meta, npy_dir, patch_size, get_train_transforms())
    val_ds = Luna16PatchDataset(val_df, meta, npy_dir, patch_size)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
                   num_workers=num_workers, pin_memory=pin, persistent_workers=num_workers > 0),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=pin, persistent_workers=num_workers > 0),
    )
