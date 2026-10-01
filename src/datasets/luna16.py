"""
dataset.py — MONAI Dataset（L4 patch 切片產生 + L5 增強）
對應研發執行手冊 1.1 節 L4/L5
相依：monai, torch, numpy, pandas
"""
import numpy as np
import pandas as pd
import torch
from monai.data import Dataset, DataLoader, CacheDataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged,
    RandRotate90d, RandFlipd, RandZoomd, RandGaussianNoised,
    RandAffined, ToTensord,
)

HU_LO, HU_HI = -1000.0, 400.0
PATCH_SIZE = (96, 96, 96)
POS_NEG_RATIO = 1 / 3   # 正:負 = 1:3（手冊 1.1）


def make_candidate_df(annotations_csv: str, candidates_csv: str) -> pd.DataFrame:
    """合併 LUNA16 annotations（正例）與 candidates_V2（候選點，含假陽性）
    產出統一欄位：seriesuid, x, y, z, diameter, label(1=nodule,0=candidate)
    """
    ann = pd.read_csv(annotations_csv)
    ann["label"] = 1
    cand = pd.read_csv(candidates_csv)
    cand["label"] = (cand["class"] == 1).astype(int)  # candidates_V2 已含 class
    df = pd.concat([
        ann.rename(columns={"diameter_mm": "diameter"})[
            ["seriesuid", "coordX", "coordY", "coordZ", "diameter", "label"]],
        cand.rename(columns={"diameter_mm": "diameter"})[
            ["seriesuid", "coordX", "coordY", "coordZ", "diameter", "label"]],
    ], ignore_index=True)
    # 正負樣本平衡：負例降採樣至正例 3 倍
    pos = df[df.label == 1]
    neg = df[df.label == 0].sample(n=min(len(neg) if (neg := df[df.label == 0]) is not None else 0,
                                         int(len(pos) / POS_NEG_RATIO)),
                                   random_state=42)
    return pd.concat([pos, neg]).sample(frac=1.0, random_state=42)


class Luna16PatchDataset(Dataset):
    """以結節中心為中心的 3D patch 分類資料集（baseline：真假結節分類）
    偵測任務可改用 SlidingWindow 推論；此資料集對應『假陽性削減』模組
    """

    def __init__(self, df: pd.DataFrame, ct_root: str,
                 patch_size: tuple = PATCH_SIZE, transform=None):
        self.df = df.reset_index(drop=True)
        self.ct_root = ct_root
        self.patch_size = patch_size
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        ct = np.load(f"{self.ct_root}/{r.seriesuid}.npz")["arr"]  # (z,y,x) 已 HU 窗正規化
        # 物理座標 → voxel 索引（假設已重採樣至 1mm 各向同性）
        cx, cy, cz = int(r.coordZ), int(r.coordY), int(r.coordX)
        pz, py, px = self.patch_size
        patch = self._crop_pad(ct, cz, cy, cx, pz, py, px)
        sample = {"image": patch[None], "label": int(r.label)}
        if self.transform:
            sample = self.transform(sample)
        sample["image"] = torch.as_tensor(sample["image"], dtype=torch.float32)
        sample["label"] = torch.tensor(sample["label"], dtype=torch.long)
        return sample

    @staticmethod
    def _crop_pad(arr, cz, cy, cx, pz, py, px):
        """邊界安全裁剪＋零填充"""
        out = np.zeros((pz, py, px), dtype=np.float32)
        z0, y0, x0 = max(0, cz - pz // 2), max(0, cy - py // 2), max(0, cx - px // 2)
        z1, y1, x1 = z0 + pz, y0 + py, x0 + px
        sz, sy, sx = arr.shape
        dz, dy, dx = min(z1, sz) - z0, min(y1, sy) - y0, min(x1, sx) - x0
        if dz > 0 and dy > 0 and dx > 0:
            out[:dz, :dy, :dx] = arr[z0:z0 + dz, y0:y0 + dy, x0:x0 + dx]
        return out


def get_train_transforms():
    """L5 增強：旋轉/翻轉/縮放 ±10%、gamma、noise、中心微擾動（RandAffine 內含）"""
    return Compose([
        RandAffined(keys=["image"], prob=0.5,
                    rotate_range=(0.1, 0.1, 0.1),
                    scale_range=(0.1, 0.1, 0.1)),
        RandRotate90d(keys=["image"], prob=0.5, max_k=3),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
        RandZoomd(keys=["image"], prob=0.3, min_zoom=0.9, max_zoom=1.1),
        RandGaussianNoised(keys=["image"], prob=0.3, std=0.01),
    ])


def get_loaders(df, ct_root, batch_size=8, num_workers=4, cache_rate=0.8):
    n_val = int(len(df) * 0.1)
    train_ds = CacheDataset(
        Luna16PatchDataset(df.iloc[n_val:], ct_root,
                           transform=get_train_transforms()),
        cache_rate=cache_rate, num_workers=num_workers)
    val_ds = CacheDataset(
        Luna16PatchDataset(df.iloc[:n_val], ct_root),
        cache_rate=cache_rate, num_workers=num_workers)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, pin_memory=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=True),
    )
