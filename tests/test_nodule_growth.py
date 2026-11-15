import math

import numpy as np
import pandas as pd

from scripts.nlst_longitudinal_cohort import exam_labels, growth_flags
from src.eval.nodule_growth import (
    build_tracks,
    lungrads_growth,
    mask_diameters_mm,
    mask_volume_mm3,
    match_nodules,
    vdt_days,
)


def _sphere(r, n=32):
    z, y, x = np.mgrid[:n, :n, :n] - n // 2
    return (x**2 + y**2 + z**2) <= r**2


def test_volume_and_diameter_of_sphere():
    m = _sphere(5)
    assert abs(mask_volume_mm3(m) - 4 / 3 * math.pi * 125) / (4 / 3 * math.pi * 125) < 0.05
    long_d, short_d, mean_d = mask_diameters_mm(m)
    assert 10 <= long_d <= 12 and 10 <= short_d <= 12 and abs(mean_d - 11) < 1
    assert mask_diameters_mm(np.zeros((4, 4, 4))) == (0.0, 0.0, 0.0)


def test_vdt_doubling_and_shrinking():
    assert abs(vdt_days(100, 200, 365) - 365) < 1e-6
    assert vdt_days(200, 100, 365) == math.inf
    assert math.isnan(vdt_days(0, 100, 365))


def test_lungrads_growth_threshold():
    assert lungrads_growth(6.0, 7.6, 365)
    assert not lungrads_growth(6.0, 7.4, 365)
    assert not lungrads_growth(6.0, 9.0, 0)


def test_match_nodules_with_shift_and_threshold():
    prior = np.array([[0, 0, 0], [50, 0, 0]])
    cur = np.array([[52, 1, 0], [100, 0, 0], [2, 0, 1]])
    pairs = match_nodules(prior, cur, max_dist_mm=10)
    assert {(i, j) for i, j, _ in pairs} == {(0, 2), (1, 0)}
    shifted = match_nodules(prior, cur, transform=lambda p: p + [50, 0, 0], max_dist_mm=10)
    assert {(i, j) for i, j, _ in shifted} == {(0, 0), (1, 1)}


def test_build_tracks_growth_and_new():
    n = pd.DataFrame({
        "patient_id": ["p"] * 4, "exam": [0, 0, 1, 1], "exam_day": [0, 0, 365, 365],
        "x": [0, 40, 1, -60], "y": [0, 0, 0, 0], "z": [0, 0, 0, 0],
        "long_d": [6, 5, 8, 4], "short_d": [6, 5, 8, 4], "mean_d": [6.0, 5.0, 8.0, 4.0],
        "volume": [100.0, 60.0, 200.0, 30.0],
    })
    t = build_tracks(n)
    t1 = t[t.exam == 1].set_index("x")
    assert t1.loc[1, "nodule_id"] == t[(t.exam == 0) & (t.x == 0)].nodule_id.item()
    assert abs(t1.loc[1, "vdt_days"] - 365) < 1e-6 and t1.loc[1, "lungrads_growth"]
    assert t1.loc[-60, "is_new"] and not t1.loc[1, "is_new"]
    assert t.nodule_id.nunique() == 3


def test_exam_labels_horizons():
    prsn = pd.DataFrame({
        "dicom_patient_id": ["a", "b"], "scr_days0": [0, 0], "scr_days1": [365, 370], "scr_days2": [730, None],
        "candx_days": [900, None], "canc_free_days": [None, 2000],
        "age": 60, "gender": 1, "cigsmok": 1, "de_stag": None, "de_type": None,
    })
    x = exam_labels(prsn).set_index(["PatientID", "exam"])
    assert x.loc[("a", 2), "event_1y"] == 1 and x.loc[("a", 0), "event_1y"] == 0
    assert x.loc[("a", 0), "event_3y"] == 1
    assert x.loc[("b", 1), "event_4y"] == 1 - 1 and x.loc[("b", 1), "event_5y"] == -1
    gf = growth_flags(pd.DataFrame({"dicom_patient_id": ["a", "a"], "study_yr": [1, 2],
                                    "sct_ab_gwth": ["2", "1"], "sct_ab_preexist": [2, 1]}))
    assert gf.set_index("exam").grew_flag.to_dict() == {1: True, 2: False}


def test_crop_cube_and_central_component():
    from scripts.nlst_longitudinal_extract import central_component, crop_cube

    arr = np.zeros((100, 100, 100), np.int16)
    arr[48:53, 48:53, 48:53] = 7
    c = crop_cube(arr, (50, 50, 50))
    assert c.shape == (64, 64, 64) and (c == 7).sum() == 125
    edge = crop_cube(arr, (2, 2, 2), fill=-1000)
    assert (edge == -1000).sum() > 0 and (edge == 0).sum() > 0
    seg = np.zeros((64, 64, 64), np.uint8)
    seg[30:35, 30:35, 30:35] = 1      # 靠近中心
    seg[0:3, 0:3, 0:3] = 1            # 角落的鄰近結節
    m = central_component(seg)
    assert m.sum() == 125 and m[32, 32, 32]
    far = np.zeros((64, 64, 64), np.uint8)
    far[0:3, 0:3, 0:3] = 1
    assert not central_component(far).any()


def test_registration_recovers_translation(tmp_path):
    import SimpleITK as sitk

    from scripts.nlst_longitudinal_extract import apply_affine, register_prior_to_current

    rng = np.random.default_rng(0)
    z, y, x = np.mgrid[:96, :96, :96]
    base = np.full((96, 96, 96), -1000, np.float32)
    body = ((x - 48) ** 2 / 40**2 + (y - 48) ** 2 / 30**2 + (z - 48) ** 2 / 44**2) < 1
    base[body] = 40
    for _ in range(25):  # 肺內的「血管」與結節，讓 MI 有結構可對
        cz, cy, cx = rng.integers(20, 76, 3)
        base[(x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 < rng.integers(4, 30)] = -500
    shift = np.array([3.0, -4.0, 5.0])  # 前次 → 本次，LPS mm（x, y, z）
    prior = sitk.GetImageFromArray(base)
    prior.SetSpacing((2.0, 2.0, 2.0))
    cur = sitk.Resample(prior, prior, sitk.TranslationTransform(3, tuple(-shift)), sitk.sitkLinear, -1000)
    sitk.WriteImage(prior, str(tmp_path / "p.mhd"))
    sitk.WriteImage(cur, str(tmp_path / "c.mhd"))
    tf = register_prior_to_current(tmp_path / "p.mhd", tmp_path / "c.mhd", shrink=1)
    pt = np.array([[96.0, 96.0, 96.0]])
    assert np.linalg.norm(apply_affine(tf, pt) - (pt + shift)) < 2.0


def test_tracks_script_end_to_end(tmp_path):
    import json
    import subprocess
    import sys

    nod = pd.DataFrame({
        "PatientID": ["p", "p", "p"], "exam": [0, 1, 1], "exam_day": [0, 365, 365],
        "SeriesInstanceUID": ["s0", "s1", "s1"], "x": [0.0, 5.0, 80.0], "y": [0.0] * 3, "z": [0.0] * 3,
        "long_d": [6, 8, 5], "short_d": [6, 8, 5], "mean_d": [6.0, 8.0, 5.0], "volume": [100.0, 200.0, 60.0],
        "det_prob": [0.9, 0.9, 0.9], "seg_voxels": [100, 200, 60], "malignancy": [0.1, 0.3, 0.05],
        "emb_row": ["e:0", "e:1", "e:2"],
    })
    nod.to_csv(tmp_path / "nodules.csv", index=False)
    tf = {"PatientID": "p", "prior_exam": 0, "exam": 1, "metric": -0.6, "o0": 5.0, "o1": 0.0, "o2": 0.0}
    tf.update({f"m{i}{j}": float(i == j) for i in range(3) for j in range(3)})
    pd.DataFrame([tf]).to_csv(tmp_path / "transforms.csv", index=False)
    subprocess.run([sys.executable, "-m", "scripts.nlst_longitudinal_tracks", "--run_dir", str(tmp_path),
                    "--out", str(tmp_path / "o"), "--ctabc", str(tmp_path / "missing.parquet")], check=True)
    t = pd.read_csv(tmp_path / "o" / "tracks.csv")
    grown = t[(t.exam == 1) & (t.x == 5.0)].iloc[0]
    assert not grown.is_new and abs(grown.vdt_days - 365) < 1e-6 and grown.lungrads_growth
    assert t[(t.exam == 1) & (t.x == 80.0)].is_new.item()
    qc = json.load(open(tmp_path / "o" / "qc.json"))
    assert qc["registrations_ok"] == 1 and qc["matched_to_prior_pct"] == 50.0


def test_radiologist_labels_and_exam_metrics():
    from scripts.nlst_followup_refine import exam_metrics, radiologist_labels

    c = pd.DataFrame({"dicom_patient_id": ["a", "a", "b", "b"], "study_yr": [1, 1, 1, 2],
                      "sct_ab_code": ["51", "64", "51", "51"], "sct_ab_preexist": [1, 1, 2, 2],
                      "sct_ab_gwth": ["", "", "2", "1"]})
    lab = radiologist_labels(c).set_index(["PatientID", "exam"])
    assert lab.loc[("a", 1)].r_new and not lab.loc[("a", 1)].r_grew
    assert lab.loc[("b", 1)].r_grew and not lab.loc[("b", 2)].r_grew
    m = exam_metrics(pd.Series([0.9, 0.1, 0.8]), pd.Series([True, True, False]), 0.5)
    assert m["both"] == 1 and m["sens"] == 0.5 and m["ppv"] == 0.5


def test_large_nodule_labels_and_safety_threshold():
    from scripts.nlst_followup_refine import radiologist_max_diameter, safety_threshold

    c = pd.DataFrame({"dicom_patient_id": ["a", "a", "a", "b"], "study_yr": [0, 0, 1, 0],
                      "sct_ab_desc": ["51", "51", "64", "51"], "sct_long_dia": ["8", "22", "40", "5"]})
    r = radiologist_max_diameter(c).set_index(["PatientID", "exam"]).r_max_d
    assert r[("a", 0)] == 22 and r[("b", 0)] == 5 and ("a", 1) not in r.index  # 代碼 64 不算結節
    score = pd.Series([0.9, 0.8, 0.3, 0.05, 0.6])
    lab = pd.Series([True, True, True, True, False])
    assert safety_threshold(score, lab, 0.75) == 0.3      # 4 顆真的大結節保留 3 顆
    assert (score[lab] >= safety_threshold(score, lab, 1.0)).all()
