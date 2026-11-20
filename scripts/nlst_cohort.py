"""
nlst_cohort.py — NLST 病人層級外部驗證的世代（cohort）挑選，資料全來自 IDC 公開索引與臨床表

定義（T0 篩檢 CT → 一年內是否確診肺癌）：
  - 陽性：T0 篩檢後 0–365 天內確診肺癌（nlst_prsn.candx_days − scr_days0）
  - 陰性：從未確診肺癌，且無癌追蹤 ≥ 3 年（canc_free_days ≥ 1095）
  - 每位病人取一個 T0（1999 年）CT series：層厚 ≤ 2.5 mm、≥ 80 張，優先最薄、軟組織 kernel、張數多
  - 排除所有出現在 LUNA25 的病人（LUNA25 的 CT 全部來自 NLST，避免訓練/驗證重疊）
  - 陰性隨機抽 NEG_PER_POS × 陽性數（上限 --max_neg），固定亂數種子

注意：這是「LUNA25 未見過的 NLST 病人」，同一個試驗、同一批掃描儀，不等於真正的外院外部驗證。

用法：
    python -m scripts.nlst_cohort --luna25_csv data/luna25/LUNA25_Public_Training_Development_Data.csv \
        --out runs/nlst/cohort.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SOFT_KERNELS = {"STANDARD", "B30F", "B31F", "B20F", "B30S", "C", "FC01", "FC02", "FC10", "FC11", "SOFT"}


def luna25_patients(csv_path: str) -> set[str]:
    df = pd.read_csv(csv_path)
    col = next(c for c in ("PatientID", "patient_id", "PatientId", "pid") if c in df.columns)
    return set(df[col].astype(str).str.strip())


def kernel_is_soft(k) -> bool:
    s = str(k).upper().replace("[", "").replace("]", "").replace("'", "").replace('"', "").strip()
    return s in SOFT_KERNELS


def pick_series(s: pd.DataFrame) -> pd.DataFrame:
    s = s.copy()
    s["SliceThickness"] = pd.to_numeric(s.SliceThickness, errors="coerce")
    s = s[(s.SliceThickness <= 2.5) & (s.instanceCount >= 80)]
    s["soft"] = s.ConvolutionKernel.map(kernel_is_soft)
    s = s.sort_values(["PatientID", "SliceThickness", "soft", "instanceCount"],
                      ascending=[True, True, False, False])
    return s.groupby("PatientID").head(1)


def label_patients(prsn: pd.DataFrame) -> pd.DataFrame:
    p = prsn.copy()
    for c in ("scr_days0", "candx_days", "canc_free_days"):
        p[c] = pd.to_numeric(p[c], errors="coerce")
    p["PatientID"] = p.dicom_patient_id.astype(str)
    d = p.candx_days - p.scr_days0
    p["label"] = np.where((d >= 0) & (d <= 365), 1,
                          np.where(p.candx_days.isna() & (p.canc_free_days >= 3 * 365), 0, -1))
    p["days_to_dx"] = d
    return p[["PatientID", "label", "days_to_dx", "age", "gender", "cigsmok", "de_stag", "de_type"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--luna25_csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--neg_per_pos", type=float, default=3.0)
    ap.add_argument("--max_neg", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from idc_index import IDCClient
    c = IDCClient()
    c.fetch_index("ct_index")
    c.fetch_index("clinical_index")
    s = c.sql_query("""
        SELECT i.PatientID, i.StudyInstanceUID, i.SeriesInstanceUID, CAST(i.StudyDate AS VARCHAR) AS StudyDate,
               i.series_size_MB, i.instanceCount, ct.SliceThickness, ct.ConvolutionKernel
        FROM index i JOIN ct_index ct USING (SeriesInstanceUID)
        WHERE i.collection_id = 'nlst' AND i.Modality = 'CT'""")
    s = s[s.StudyDate.str.startswith("1999")]          # NLST 日期以 1999 = T0 年
    series = pick_series(s)

    clin = sorted(Path.home().glob(".local/share/IDC/*/clinical_data/nlst_prsn"))
    if not clin:
        raise FileNotFoundError("IDC clinical table nlst_prsn not found after fetch_index('clinical_index')")
    prsn = pd.read_parquet(clin[-1])
    m = series.merge(label_patients(prsn), on="PatientID")
    m = m[m.label >= 0]

    excl = luna25_patients(args.luna25_csv)
    n_before = m.groupby("label").size().to_dict()
    m = m[~m.PatientID.isin(excl)]
    pos = m[m.label == 1]
    neg = m[m.label == 0]
    n_neg = int(min(len(neg), args.max_neg, round(args.neg_per_pos * len(pos))))
    neg = neg.sample(n_neg, random_state=args.seed)
    cohort = pd.concat([pos, neg]).sort_values("PatientID").reset_index(drop=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cohort.drop(columns=["soft"]).to_csv(args.out, index=False)
    print(f"T0 thin-slice CT with label: {n_before} | LUNA25 patients excluded: "
          f"{len(excl)} listed | cohort: {len(pos)} pos + {len(neg)} neg = {len(cohort)} scans, "
          f"{cohort.series_size_MB.sum() / 1024:.1f} GB")


if __name__ == "__main__":
    main()
