"""前處理邏輯單元測試（無需真實資料，純 numpy 驗證）"""
import numpy as np
import pytest


def hu_window(arr, lo=-1000, hi=400):
    from src.preprocess.pipeline import apply_hu_window
    return apply_hu_window(arr)


def sphere_mask(shape, center, diameter, spacing=(1, 1, 1)):
    from src.preprocess.pipeline import sphere_labelmap
    return sphere_labelmap(shape, center, diameter, spacing)


class TestHUWindow:
    def test_clip_range(self):
        arr = np.array([-1200.0, -850.0, 0.0, 400.0, 800.0])
        out = hu_window(arr)
        assert out.min() == 0.0 and out.max() == 1.0

    def test_air_is_zero(self):
        assert hu_window(np.array([-1000.0]))[0] == 0.0


class TestSphereLabel:
    def test_volume_approx(self):
        # 直徑 10mm、spacing 1mm → 球體積 ≈ 523.6 體素
        mask = sphere_mask((32, 32, 32), (16, 16, 16), 10.0)
        assert abs(mask.sum() - 523.6) / 523.6 < 0.05


class TestSliceThickness:
    def test_filter(self):
        from src.preprocess.pipeline import filter_by_slice_thickness
        assert filter_by_slice_thickness((1.0, 1.0, 2.0)) is True
        assert filter_by_slice_thickness((1.0, 1.0, 3.0)) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
