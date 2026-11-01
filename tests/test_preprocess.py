"""前處理邏輯單元測試（無需真實資料，合成影像驗證）"""
import numpy as np
import SimpleITK as sitk

from src.preprocess.pipeline import (
    apply_hu_window,
    filter_by_slice_thickness,
    resample_isotropic,
    sphere_labelmap,
    world_to_voxel,
)


class TestHUWindow:
    def test_clip_range(self):
        out = apply_hu_window(np.array([-1200.0, -850.0, 0.0, 400.0, 800.0]))
        assert out.min() == 0.0 and out.max() == 1.0

    def test_air_is_zero(self):
        assert apply_hu_window(np.array([-1000.0]))[0] == 0.0


class TestSphereLabel:
    def test_volume_approx(self):
        # 直徑 10mm、spacing 1mm → 球體積 ≈ 523.6 體素
        mask = sphere_labelmap((32, 32, 32), (16, 16, 16), 10.0)
        assert abs(mask.sum() - 523.6) / 523.6 < 0.05

    def test_anisotropic_spacing_and_origin(self):
        # spacing z=2mm：同樣 10mm 球，體素數約為一半
        mask = sphere_labelmap((32, 32, 32), (-84.0, -84.0, -268.0), 10.0,
                               spacing=(1.0, 1.0, 2.0), origin=(-100.0, -100.0, -300.0))
        assert abs(mask.sum() - 261.8) / 261.8 < 0.08
        assert mask[16, 16, 16] == 1


class TestSliceThickness:
    def test_filter(self):
        assert filter_by_slice_thickness((1.0, 1.0, 2.0)) is True
        assert filter_by_slice_thickness((1.0, 1.0, 3.0)) is False


def _synthetic_ct():
    """z=10 層 @2.5mm，y=x=20 @0.7mm，非零 origin，一個亮點在已知世界座標"""
    arr = np.full((10, 20, 20), -1000, dtype=np.float32)
    arr[4, 6, 12] = 500
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((0.7, 0.7, 2.5))
    img.SetOrigin((-50.0, -20.0, -300.0))
    return img


class TestResample:
    def test_shape_and_geometry(self):
        img = _synthetic_ct()
        rs = resample_isotropic(img)
        arr = sitk.GetArrayFromImage(rs)
        assert arr.shape == (25, 14, 14)            # (z, y, x)
        assert rs.GetSpacing() == (1.0, 1.0, 1.0)
        assert rs.GetOrigin() == img.GetOrigin()
        # 邊界外填空氣而非 0 HU
        assert arr.min() >= -1000 and arr[-1, -1, -1] == -1000

    def test_world_point_maps_to_same_voxel(self):
        img = _synthetic_ct()
        world = img.TransformIndexToPhysicalPoint((12, 6, 4))   # (x, y, z) index
        rs = resample_isotropic(img)
        z, y, x = np.round(world_to_voxel(world, rs.GetOrigin(), rs.GetSpacing(),
                                          rs.GetDirection())).astype(int)
        arr = sitk.GetArrayFromImage(rs)
        assert arr[z, y, x] > -1000                   # 亮點仍在換算出的位置
        assert np.unravel_index(arr.argmax(), arr.shape) == (z, y, x)
