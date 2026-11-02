"""chain_fp_filter：融合分數與端到端打分"""
import numpy as np
import pandas as pd

from scripts import chain_fp_filter, train_baseline
from src.datasets.luna16 import load_meta
from src.preprocess.pipeline import process_luna16
from tests.test_e2e_smoke import _make_fake_luna


def _toy():
    ann = pd.DataFrame({"seriesuid": ["a", "b"], "coordX": [0.0, 0.0], "coordY": [0.0, 0.0],
                        "coordZ": [0.0, 0.0], "diameter_mm": [10.0, 10.0]})
    # 每例 1 個真結節（分數低）+ 4 個高分假陽性
    rows = []
    for uid in ("a", "b"):
        rows.append({"seriesuid": uid, "coordX": 0.0, "coordY": 0.0, "coordZ": 0.0, "probability": 0.5})
        for k in range(4):
            rows.append({"seriesuid": uid, "coordX": 50.0 + k * 10, "coordY": 0.0, "coordZ": 0.0,
                         "probability": 0.9})
    return pd.DataFrame(rows), ann


def test_perfect_filter_lifts_low_fp_sensitivity():
    dets, ann = _toy()
    p_cls = np.where(dets.coordX == 0.0, 0.99, 0.01)
    res = chain_fp_filter.evaluate_fusions(dets, p_cls, ann, n_scans=2, excluded=None)
    assert res["det"]["sensitivity"][0.125] == 0.0       # 偵測器單獨：低 FP 區間抓不到
    assert res["product"]["sensitivity"][0.125] == 1.0   # 接上過濾器後全部抓到
    assert res["product"]["cpm"] > res["det"]["cpm"]


def test_missing_filter_score_keeps_detector_score():
    dets, ann = _toy()
    p_cls = np.full(len(dets), np.nan)
    res = chain_fp_filter.evaluate_fusions(dets, p_cls, ann, n_scans=2, excluded=None)
    assert res["product"]["cpm"] == res["det"]["cpm"]


def test_score_detections_end_to_end(tmp_path):
    luna = tmp_path / "luna16"
    _make_fake_luna(luna, n_series=4)
    proc = tmp_path / "processed"
    process_luna16(str(luna), str(proc), [0], workers=1)
    out = tmp_path / "run"
    train_baseline.main(train_baseline.parse_args([
        "--meta", str(proc / "meta.csv"), "--npy_dir", str(proc / "npy"),
        "--candidates", str(luna / "candidates_V2.csv"), "--out", str(out),
        "--patch_size", "32", "--epochs", "1", "--batch_size", "4", "--num_workers", "0",
    ]))
    cand = pd.read_csv(luna / "candidates_V2.csv")
    dets = cand.assign(probability=0.5)[["seriesuid", "coordX", "coordY", "coordZ", "probability"]]
    dets = pd.concat([dets, dets.head(1).assign(seriesuid="not.preprocessed")], ignore_index=True)
    p = chain_fp_filter.score_detections(dets, load_meta(str(proc / "meta.csv")), str(proc / "npy"),
                                         str(out / "best.pt"), patch_size=32, batch_size=8)
    assert np.isnan(p[-1]) and np.all((p[:-1] >= 0) & (p[:-1] <= 1))
