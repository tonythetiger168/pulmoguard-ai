"""detector_tta：翻轉後的偵測座標能正確翻回、合併分群規則正確"""
import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk

from scripts.detector_tta import flip_volume, merge, unflip


@pytest.mark.parametrize("flip", ["x", "y", "z", "xy"])
def test_flip_roundtrip(tmp_path, flip):
    arr = np.zeros((20, 30, 40), np.int16)
    arr[5, 7, 9] = 1000                                   # (z, y, x)
    img = sitk.GetImageFromArray(arr)
    img.SetOrigin((-100.0, -50.0, -300.0))
    img.SetSpacing((0.7, 0.8, 2.5))
    sitk.WriteImage(img, str(tmp_path / "a.mhd"))
    truth = img.TransformIndexToPhysicalPoint((9, 7, 5))
    params = flip_volume(tmp_path / "a.mhd", tmp_path / flip / "a.mhd", flip)
    flipped = sitk.ReadImage(str(tmp_path / flip / "a.mhd"))
    z, y, x = [int(v[0]) for v in np.where(sitk.GetArrayFromImage(flipped) == 1000)]
    w = flipped.TransformIndexToPhysicalPoint((x, y, z))
    det = pd.DataFrame([{"seriesuid": "a", "coordX": w[0], "coordY": w[1], "coordZ": w[2], "probability": 0.9}])
    u = unflip(det, {"a": params}).iloc[0]
    assert np.allclose([u.coordX, u.coordY, u.coordZ], truth)


def test_merge_mean_and_max():
    p0 = pd.DataFrame({"seriesuid": ["a", "a"], "coordX": [0, 50], "coordY": [0, 0], "coordZ": [0, 0],
                       "probability": [0.9, 0.8], "size": [6, 6]})
    p1 = pd.DataFrame({"seriesuid": ["a"], "coordX": [1], "coordY": [0], "coordZ": [0],
                       "probability": [0.7], "size": [6]})
    mean = merge([p0, p1], "mean").sort_values("coordX").probability.tolist()
    mx = merge([p0, p1], "max").sort_values("coordX").probability.tolist()
    assert np.allclose(mean, [0.8, 0.4]) and np.allclose(mx, [0.9, 0.8])
