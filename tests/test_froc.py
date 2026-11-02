"""FROC/CPM 計算單元測試"""
import pandas as pd

from src.eval.froc import froc


def _ann():
    return pd.DataFrame({"seriesuid": ["a", "b"], "coordX": [0.0, 0.0], "coordY": [0.0, 0.0],
                         "coordZ": [0.0, 0.0], "diameter_mm": [10.0, 10.0]})


def test_perfect_detector():
    dets = pd.DataFrame({"seriesuid": ["a", "b"], "coordX": [1.0, 0.0], "coordY": [0.0, 2.0],
                         "coordZ": [0.0, 0.0], "probability": [0.9, 0.8]})
    r = froc(dets, _ann(), n_scans=2)
    assert r["cpm"] == 1.0


def test_duplicate_hits_not_fp_and_fp_ranked():
    dets = pd.DataFrame({
        "seriesuid": ["a", "a", "b", "b"],
        "coordX": [0.0, 1.0, 50.0, 0.0], "coordY": [0.0] * 4, "coordZ": [0.0] * 4,
        "probability": [0.9, 0.5, 0.95, 0.3],   # b 的 FP 分數高於 b 的 TP
    })
    r = froc(dets, _ann(), n_scans=2)
    # 1 個 FP（0.95）：0.125/0.25 FP/scan 時 0 個 FP 允許 → 只有分數 > 0.95 的 TP（無）
    assert r["sensitivity"][0.25] == 0.0
    # 1 FP/scan → 允許 2 個 FP → 兩顆都找到
    assert r["sensitivity"][1] == 1.0


def test_excluded_findings_ignored():
    dets = pd.DataFrame({"seriesuid": ["a"], "coordX": [30.0], "coordY": [0.0],
                         "coordZ": [0.0], "probability": [0.99]})
    ex = pd.DataFrame({"seriesuid": ["a"], "coordX": [30.0], "coordY": [0.0],
                       "coordZ": [0.0], "diameter_mm": [6.0]})
    r = froc(dets, _ann(), n_scans=2, excluded=ex)
    assert r["sensitivity"][0.125] == 0.0  # 沒有 TP，但也沒有 FP
