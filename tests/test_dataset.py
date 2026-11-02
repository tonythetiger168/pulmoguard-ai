"""LUNA16 資料集：座標換算、裁切置中、病人層級切分"""
import json

import numpy as np
import pandas as pd

from src.datasets.luna16 import Luna16PatchDataset, crop_pad, split_by_series


def test_crop_pad_centered_at_border():
    arr = np.arange(10 ** 3, dtype=np.float32).reshape(10, 10, 10)
    out = crop_pad(arr, (0, 5, 9), (4, 4, 4))
    assert out[2, 2, 2] == arr[0, 5, 9]   # 中心點永遠在 patch 中央
    assert out[0, 0, 0] == 0.0            # 超出邊界補 0


def test_dataset_patch_center_is_nodule(tmp_path):
    vol = np.zeros((40, 50, 60), dtype=np.float16)
    vol[10, 20, 30] = 1.0                 # (z, y, x)
    np.save(tmp_path / "s1.npy", vol)
    origin = (-100.0, -200.0, -300.0)
    meta = pd.DataFrame({"seriesuid": ["s1"], "subset": [0], "shape_zyx": [json.dumps([40, 50, 60])],
                         "origin_xyz": [json.dumps(origin)], "spacing_xyz": [json.dumps([1.0, 1.0, 1.0])],
                         "direction": [json.dumps([1, 0, 0, 0, 1, 0, 0, 0, 1])]})
    meta_csv = tmp_path / "meta.csv"
    meta.to_csv(meta_csv, index=False)
    from src.datasets.luna16 import load_meta
    m = load_meta(str(meta_csv))
    df = pd.DataFrame({"seriesuid": ["s1"], "coordX": [origin[0] + 30], "coordY": [origin[1] + 20],
                       "coordZ": [origin[2] + 10], "label": [1], "subset": [0]})
    ds = Luna16PatchDataset(df, m, str(tmp_path), patch_size=(8, 8, 8))
    img = ds[0]["image"]
    assert img.shape == (1, 8, 8, 8)
    assert img[0, 4, 4, 4] == 1.0


def test_split_has_no_patient_leakage():
    df = pd.DataFrame({"seriesuid": [f"s{i // 4}" for i in range(80)], "label": [i % 2 for i in range(80)],
                       "subset": [0] * 80})
    tr, va = split_by_series(df)
    assert set(tr.seriesuid).isdisjoint(va.seriesuid)
    assert len(va) > 0


def test_split_uses_official_fold():
    df = pd.DataFrame({"seriesuid": ["a", "b", "c"], "label": [1, 0, 1], "subset": [0, 1, 2]})
    tr, va = split_by_series(df, val_subset=1)
    assert list(va.seriesuid) == ["b"]
