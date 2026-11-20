"""LUNA25 nodule blocks：取樣置中、病人層級切分、端到端煙霧測試（假資料）"""
import json

import numpy as np
import pandas as pd

from src.datasets.luna25 import Luna25BlockDataset, inner_split, load_annotations, make_folds, resample_patch


def _fake_luna25(tmp_path, n_patients=24, per_patient=3):
    (tmp_path / "image").mkdir()
    (tmp_path / "metadata").mkdir()
    rows, rng = [], np.random.default_rng(0)
    for p in range(n_patients):
        for j in range(per_patient):
            aid = f"P{p}_{j}"
            label = int(p % 4 == 0)          # 約 25% 惡性；同病人標籤相同
            block = np.full((64, 128, 128), -900, dtype=np.int16)
            r = 6 if label else 3            # 惡性結節較大，讓模型有東西可學
            block[32 - r:32 + r, 64 - r:64 + r, 64 - r:64 + r] = 40
            np.save(tmp_path / "image" / f"{aid}.npy", block)
            np.save(tmp_path / "metadata" / f"{aid}.npy",
                    {"origin": np.zeros(3), "spacing": np.array([1.25, 0.7, 0.7]), "transform": np.eye(3)})
            rows.append({"AnnotationID": aid, "PatientID": f"P{p}", "label": label,
                         "SeriesInstanceUID": f"S{p}", "Age_at_StudyDate": int(rng.integers(55, 75))})
    csv = tmp_path / "ann.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv


def test_resample_keeps_center():
    block = np.full((64, 128, 128), -1000, dtype=np.float32)
    block[28:36, 58:70, 58:70] = 100.0     # 約 10 mm 立方體
    out = resample_patch(block, (1.25, 0.7, 0.7), size_px=32, size_mm=40)
    assert out.shape == (32, 32, 32)
    assert out[15:17, 15:17, 15:17].mean() > 0     # 結節仍在 patch 中央
    assert out[0, 0, 0] == -1000


def test_dataset_shapes_and_aug(tmp_path):
    csv = _fake_luna25(tmp_path, n_patients=4)
    df = load_annotations(str(csv))
    ds = Luna25BlockDataset(df, str(tmp_path), size_px=32, augment=True)
    s = ds[0]
    assert s["image"].shape == (1, 32, 32, 32)
    assert 0.0 <= float(s["image"].min()) and float(s["image"].max()) <= 1.0


def test_folds_are_patient_level(tmp_path):
    # 48 位病人、12 位惡性：舊版 sklearn（1.7）的分層較粗，6 位惡性分 4 折可能出現沒有惡性的 fold
    df = load_annotations(str(_fake_luna25(tmp_path, n_patients=48)))
    df = make_folds(df, 4)
    assert df.groupby("group").fold.nunique().max() == 1
    assert all(df[df.fold == k].label.sum() > 0 for k in range(4))
    tr, va = inner_split(df[df.fold != 0])
    assert set(tr.group).isdisjoint(va.group) and len(va) > 0


def test_train_luna25_end_to_end(tmp_path):
    from scripts.train_luna25 import main, parse_args
    csv = _fake_luna25(tmp_path)
    out = tmp_path / "run"
    main(parse_args(["--csv", str(csv), "--blocks", str(tmp_path), "--out", str(out), "--folds", "2",
                     "--epochs", "1", "--batch_size", "4", "--num_workers", "1", "--size_px", "32"]))
    s = json.load(open(out / "summary.json"))
    assert s["n_nodules"] == 72 and len(s["per_fold"]) == 2
    assert (out / "oof_predictions.csv").exists()
    # 第二次執行會跳過已完成的 fold（spot pod 續跑）
    main(parse_args(["--csv", str(csv), "--blocks", str(tmp_path), "--out", str(out), "--folds", "2",
                     "--epochs", "1", "--batch_size", "4", "--num_workers", "1", "--size_px", "32"]))


def test_train_luna25_limit(tmp_path):
    """--limit 煙霧測試：分層取樣後仍保留 label 欄"""
    from scripts.train_luna25 import main, parse_args
    csv = _fake_luna25(tmp_path)
    out = tmp_path / "run"
    main(parse_args(["--csv", str(csv), "--blocks", str(tmp_path), "--out", str(out), "--folds", "2",
                     "--epochs", "1", "--batch_size", "4", "--num_workers", "1", "--size_px", "32",
                     "--limit", "40"]))
    assert json.load(open(out / "summary.json"))["n_nodules"] == 38   # 惡性只有 18 顆，全取；良性取 20
