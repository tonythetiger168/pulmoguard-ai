import numpy as np
import pandas as pd

from scripts.nlst_cohort import kernel_is_soft, label_patients, pick_series


def test_label_patients_one_year_window():
    prsn = pd.DataFrame({
        "dicom_patient_id": ["1", "2", "3", "4"],
        "scr_days0": ["10", "10", "10", "10"],
        "candx_days": ["200", "900", None, None],
        "canc_free_days": ["0", "0", "2000", "500"],
        "age": 60, "gender": 1, "cigsmok": 1, "de_stag": None, "de_type": None,
    })
    lab = label_patients(prsn).set_index("PatientID").label.to_dict()
    assert lab == {"1": 1, "2": -1, "3": 0, "4": -1}


def test_pick_series_prefers_thin_soft():
    s = pd.DataFrame({
        "PatientID": ["a", "a", "a"], "SliceThickness": ["2.0", "2.0", "5.0"],
        "instanceCount": [150, 160, 60], "ConvolutionKernel": ["['B30f']", "['B50f']", "['B30f']"],
        "SeriesInstanceUID": ["soft", "sharp", "thick"],
    })
    assert pick_series(s).SeriesInstanceUID.tolist() == ["soft"]
    assert kernel_is_soft("['STANDARD']") and not kernel_is_soft("['BONE']")


def test_extract_block_centre_and_padding():
    import pytest
    pytest.importorskip("monai")
    from scripts.nlst_validate import extract_block
    a = np.arange(10 ** 3, dtype=np.int16).reshape(10, 10, 10)
    b = extract_block(a, np.array([5.0, 5.0, 5.0]))
    assert b.shape == (81, 81, 81) and b[40, 40, 40] == a[5, 5, 5] and b[0, 0, 0] == -1000
