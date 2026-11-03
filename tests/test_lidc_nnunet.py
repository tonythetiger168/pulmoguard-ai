import numpy as np

from scripts.lesion_metrics import case_metrics
from scripts.lidc_to_nnunet import patient_folds


def test_lesion_metrics_hit_miss_fp():
    g = np.zeros((40, 40, 40), np.uint8)
    g[5:10, 5:10, 5:10] = 1      # 偵測到
    g[30:33, 30:33, 30:33] = 1   # 漏掉
    p = np.zeros_like(g)
    p[6:10, 5:10, 5:10] = 1
    p[20:22, 1:3, 1:3] = 1       # 假陽性
    m = case_metrics(p, g, np.ones(3))
    assert [les["detected"] for les in m["lesions"]] == [True, False]
    assert m["fp"] == 1
    assert 0.85 < m["lesions"][0]["dice"] < 0.95


def test_patient_folds_keep_patient_together():
    cases = [{"case": f"c{i}", "patient_id": f"p{i // 2}"} for i in range(20)]
    splits = patient_folds(cases, k=5)
    assert len(splits) == 5
    for s in splits:
        val_p = {c["patient_id"] for c in cases if c["case"] in s["val"]}
        tr_p = {c["patient_id"] for c in cases if c["case"] in s["train"]}
        assert not val_p & tr_p
    assert sorted(sum((s["val"] for s in splits), [])) == sorted(c["case"] for c in cases)


def test_roi_extract_pads_outside_image():
    from scripts.lidc_crops import crop_box, extract
    a = np.arange(27).reshape(3, 3, 3)
    box = crop_box((0, 0, 0), (2, 2, 2))   # 一半在影像外
    out = extract(a, box, -1)
    assert out.shape == (2, 2, 2)
    assert out[0, 0, 0] == -1 and out[1, 1, 1] == a[0, 0, 0]
