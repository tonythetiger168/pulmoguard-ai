"""
nlst_longitudinal_cohort.py — NLST 縱向（T0/T1/T2）世代：同一位病人每年各挑一個薄層 CT series

每列 = 一位病人的一次篩檢（exam 0/1/2），附上：
  - exam_day：該次篩檢距 T0 的天數（nlst_prsn.scr_days0/1/2）
  - days_to_dx：該次篩檢到確診肺癌的天數（沒確診 = NaN）
  - followup_days：沒確診者，該次篩檢到最後一次確認無癌的天數（canc_free_days − scr_daysk）
  - event_{1..6}y：該次篩檢後 N 年內確診 = 1；未確診且追蹤 ≥ N 年 = 0；追蹤不足 = −1（評估時排除）
  - grew_flag：放射科醫師在 nlst_ctabc 記錄該次有異常「比前次長大」（sct_ab_gwth = 2）
  - in_luna25：病人出現在 LUNA25（LUNA25 全來自 NLST；測試集要排除）

抽樣：≥ 2 年都有薄層 CT 的病人中，6 年內確診者全收；非癌症者中有「長大」或「既有結節」紀錄的抽 --n_growth，
其他隨機抽 --n_random。固定亂數種子。

用法：
    python -m scripts.nlst_longitudinal_cohort --luna25_csv data/luna25/LUNA25_Public_Training_Development_Data.csv \
        --out runs/nlst_long/cohort.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.nlst_cohort import luna25_patients, pick_series

HORIZONS = (1, 2, 3, 4, 5, 6)
YEAR_TO_EXAM = {"1999": 0, "2000": 1, "2001": 2}


def exam_labels(prsn: pd.DataFrame) -> pd.DataFrame:
    """每位病人展開成三列（exam 0/1/2），算該次篩檢起的事件時間與各年限標籤。"""
    p = prsn.copy()
    for c in ("scr_days0", "scr_days1", "scr_days2", "candx_days", "canc_free_days"):
        p[c] = pd.to_numeric(p[c], errors="coerce")
    p["PatientID"] = p.dicom_patient_id.astype(str)
    rows = []
    for k in (0, 1, 2):
        e = p[["PatientID", "candx_days", "canc_free_days", "scr_days0",
               "age", "gender", "cigsmok", "de_stag", "de_type"]].copy()
        e["scr_day"] = p[f"scr_days{k}"]
        e["exam"] = k
        rows.append(e)
    x = pd.concat(rows, ignore_index=True)
    x["exam_day"] = x.scr_day - x.scr_days0
    x["days_to_dx"] = x.candx_days - x.scr_day
    x["followup_days"] = np.where(x.candx_days.isna(), x.canc_free_days - x.scr_day, x.days_to_dx)
    for h in HORIZONS:
        t = 365.25 * h
        x[f"event_{h}y"] = np.where(x.days_to_dx.notna() & (x.days_to_dx >= 0) & (x.days_to_dx <= t), 1,
                                    np.where(x.followup_days >= t, 0, -1))
    # 確診在此次篩檢之前 = 這次已不是篩檢情境
    x = x[~(x.days_to_dx < 0)]
    return x.drop(columns=["scr_days0", "candx_days", "canc_free_days"])


def growth_flags(ctabc: pd.DataFrame) -> pd.DataFrame:
    c = ctabc.copy()
    c["PatientID"] = c.dicom_patient_id.astype(str)
    c["exam"] = pd.to_numeric(c.study_yr, errors="coerce")
    c["grew"] = c.sct_ab_gwth.astype(str).str.strip().isin(["2", "2.0"])
    c["preexist"] = pd.to_numeric(c.sct_ab_preexist, errors="coerce") == 2
    return c.groupby(["PatientID", "exam"]).agg(grew_flag=("grew", "any"), preexist_flag=("preexist", "any")).reset_index()


def sample_patients(per_patient: pd.DataFrame, n_growth: int, n_random: int, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    cancer = per_patient[per_patient.cancer_6y]
    rest = per_patient[~per_patient.cancer_6y]
    g = rest[rest.any_growth_or_preexist]
    g = g.sample(min(n_growth, len(g)), random_state=rng)
    r = rest.drop(g.index)
    r = r.sample(min(n_random, len(r)), random_state=rng)
    return pd.concat([cancer.assign(stratum="cancer"), g.assign(stratum="growth"), r.assign(stratum="random")])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--luna25_csv", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_growth", type=int, default=1000)
    ap.add_argument("--n_random", type=int, default=1000)
    ap.add_argument("--max_patients", type=int, default=None, help="試跑用：最後再隨機截到這麼多人（分層保留）")
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
    s["exam"] = s.StudyDate.str[:4].map(YEAR_TO_EXAM)
    s = s[s.exam.notna()]
    series = pd.concat([pick_series(g).assign(exam=int(e)) for e, g in s.groupby("exam")], ignore_index=True)

    clin = sorted(Path.home().glob(".local/share/IDC/*/clinical_data"))
    if not clin:
        raise FileNotFoundError("IDC clinical tables not found after fetch_index('clinical_index')")
    prsn = pd.read_parquet(clin[-1] / "nlst_prsn")
    ctabc = pd.read_parquet(clin[-1] / "nlst_ctabc")

    m = series.merge(exam_labels(prsn), on=["PatientID", "exam"]).merge(
        growth_flags(ctabc), on=["PatientID", "exam"], how="left")
    m[["grew_flag", "preexist_flag"]] = m[["grew_flag", "preexist_flag"]].fillna(False).astype(bool)
    n_exam = m.groupby("PatientID").exam.nunique()
    m = m[m.PatientID.isin(n_exam[n_exam >= 2].index)]

    per = m.groupby("PatientID").agg(cancer_6y=("event_6y", lambda v: (v == 1).any()),
                                     any_growth_or_preexist=("grew_flag", "any"),
                                     pre=("preexist_flag", "any"))
    per["any_growth_or_preexist"] |= per.pop("pre")
    picked = sample_patients(per, args.n_growth, args.n_random, args.seed)
    if args.max_patients and len(picked) > args.max_patients:
        frac = args.max_patients / len(picked)
        picked = picked.groupby("stratum").sample(frac=frac, random_state=args.seed)

    cohort = m.merge(picked[["stratum"]], left_on="PatientID", right_index=True)
    excl = luna25_patients(args.luna25_csv) if args.luna25_csv else set()
    cohort["in_luna25"] = cohort.PatientID.isin(excl)
    cohort = cohort.drop(columns=["soft"], errors="ignore").sort_values(["PatientID", "exam"]).reset_index(drop=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(args.out, index=False)
    print(f"patients: {cohort.PatientID.nunique()} ({picked.stratum.value_counts().to_dict()}) | "
          f"scans: {len(cohort)} | in LUNA25: {cohort[cohort.in_luna25].PatientID.nunique()} patients | "
          f"{cohort.series_size_MB.sum() / 1024:.1f} GB")


if __name__ == "__main__":
    main()
