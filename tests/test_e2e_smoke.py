"""端到端煙霧測試：合成 LUNA16 結構 → 前處理 → 訓練 1 epoch（CPU，約數十秒）"""
import json

import numpy as np
import pandas as pd
import SimpleITK as sitk

from scripts import train_baseline
from src.preprocess.pipeline import process_luna16


def _make_fake_luna(root, n_series=6):
    sub = root / "subset0"
    sub.mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_series):
        uid = f"1.2.3.{i}"
        arr = np.full((30, 64, 64), -1000, dtype=np.int16)
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing((0.8, 0.8, 1.5))
        img.SetOrigin((-25.0, -25.0, -20.0))
        for k in range(8):
            c = (int(rng.integers(10, 54)), int(rng.integers(10, 54)), int(rng.integers(5, 25)))
            w = img.TransformIndexToPhysicalPoint(c)
            label = int(k < 2)
            if label:
                z, y, x = c[2], c[1], c[0]
                arr[z - 2:z + 3, y - 3:y + 4, x - 3:x + 4] = 50
            rows.append({"seriesuid": uid, "coordX": w[0], "coordY": w[1], "coordZ": w[2], "class": label})
        out = sitk.GetImageFromArray(arr)
        out.CopyInformation(img)
        sitk.WriteImage(out, str(sub / f"{uid}.mhd"))
    pd.DataFrame(rows).to_csv(root / "candidates_V2.csv", index=False)


def test_preprocess_and_train_one_epoch(tmp_path):
    luna = tmp_path / "luna16"
    _make_fake_luna(luna)
    proc = tmp_path / "processed"
    metas = process_luna16(str(luna), str(proc), [0], workers=1)
    assert len(metas) == 6 and (proc / "meta.csv").exists()

    out = tmp_path / "run"
    cfg = train_baseline.parse_args([
        "--meta", str(proc / "meta.csv"), "--npy_dir", str(proc / "npy"),
        "--candidates", str(luna / "candidates_V2.csv"), "--out", str(out),
        "--patch_size", "32", "--epochs", "1", "--batch_size", "4", "--num_workers", "0",
    ])
    train_baseline.main(cfg)
    m = json.loads((out / "metrics.json").read_text())
    assert m["epochs_run"] == 1 and (out / "last.pt").exists()

    # 續跑：從 last.pt 接續
    cfg.resume, cfg.epochs = True, 2
    train_baseline.main(cfg)
    assert json.loads((out / "metrics.json").read_text())["epochs_run"] == 2
