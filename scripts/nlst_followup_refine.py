"""
nlst_followup_refine.py — 讓「新發結節」與「變大」的判定更接近 NLST 放射科醫師（CPU，不需要影像）

輸入：nlst_longitudinal_tracks 的 tracks.csv，加上 extract 各分片的輸出資料夾（nodules.csv、transforms.csv、
embeddings/*.npy，用來取 LUNA25 的 512 維特徵）。

步驟：
  1) 補配對：可比較的 is_new 結節，若前次 20 mm 內有「本次沒被配走」的結節，且 LUNA25 特徵餘弦相似度 ≥ --min_cos、
     大小差在 2 倍內 → 視為同一顆（10 mm 的距離門檻漏掉的），重算體積變化、VDT、Lung-RADS 生長
  2) 弱監督分類器（多例學習）：醫師只在「檢查」層級記錄有沒有新結節、有沒有結節長大（nlst_ctabc，代碼 51
     非鈣化結節）。每顆候選的標籤 = 所屬檢查的醫師標籤，檢查分數 = 候選分數最大值。
       new_prob ：is_new 候選是真的新結節的機率（特徵：偵測分數、惡性度、大小、形狀、前次最近候選距離、特徵 PCA）
       grow_prob：有前次配對的結節真的長大的機率（特徵：體積與直徑變化、VDT、前次大小、配對距離、前後特徵差）
  3) 病人隨機對半分成 dev / test：dev 上 5-fold 選門檻（檢查層級 F1 最大），用全部 dev 訓練後只在 test 報告
  4) 大結節過濾器：平均徑 ≥ 15 mm 的候選常是偵測器把大的正常構造當成結節。標籤 = 該次檢查醫師有沒有記錄
     ≥ 10 mm 的非鈣化結節（nlst_ctab 代碼 51 的長徑），同樣多例學習、同樣 dev / test
輸出：tracks_refined.csv（新增 rematched、new_prob、grow_prob、is_new_refined、growth_refined、growth_confident、
      large_prob、large_real（< 15 mm 一律 True））、refine.json
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from src.eval.nodule_growth import lungrads_growth, vdt_days

PCA_DIM = 16


def radiologist_labels(ctabc: pd.DataFrame) -> pd.DataFrame:
    b = ctabc.copy()
    b["PatientID"] = b.dicom_patient_id.astype(str)
    b["exam"] = pd.to_numeric(b.study_yr, errors="coerce")
    b = b[b.sct_ab_code.astype(str).str.strip() == "51"]   # 非鈣化結節
    return b.groupby(["PatientID", "exam"]).agg(
        r_new=("sct_ab_preexist", lambda s: bool((pd.to_numeric(s, errors="coerce") == 1).any())),
        r_grew=("sct_ab_gwth", lambda s: bool((s.astype(str).str.strip() == "2").any()))).reset_index()


def load_embeddings(run_dirs: list[str], tracks: pd.DataFrame) -> np.ndarray:
    """emb_row = '檔名:列'，各分片檔名重複，所以先用 PatientID 判斷屬於哪個分片。"""
    owner, cache = {}, {}
    for d in run_dirs:
        for p in pd.read_csv(Path(d) / "cohort.csv", dtype={"PatientID": str}).PatientID.unique():
            owner[p] = d
    out = np.full((len(tracks), 512), np.nan, np.float32)
    for i, (pid, er) in enumerate(zip(tracks.PatientID, tracks.emb_row)):
        if pid not in owner or not isinstance(er, str):
            continue
        f, k = er.split(":")
        key = (owner[pid], f)
        if key not in cache:
            cache[key] = np.load(Path(owner[pid]) / "embeddings" / f).astype(np.float32)
        out[i] = cache[key][int(k)]
    return out


def load_transforms(run_dirs: list[str]) -> dict:
    tf = pd.concat([pd.read_csv(Path(d) / "transforms.csv", dtype={"PatientID": str}) for d in run_dirs])
    from scripts.nlst_longitudinal_tracks import make_transforms
    return make_transforms(tf)


def rematch(t: pd.DataFrame, emb: np.ndarray, transforms: dict, max_mm: float, min_cos: float) -> pd.DataFrame:
    t = t.copy()
    t["rematched"] = False
    en = emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(1e-6)
    for (pid, ex), g in t[t.comparable.astype(bool) & (t.exam > 0)].groupby(["PatientID", "exam"]):
        new_idx = g.index[g.is_new.astype(bool)]
        if not len(new_idx):
            continue
        pex = t[(t.PatientID == pid) & (t.exam < ex)].exam.max()
        tf = transforms.get((pid, int(pex), int(ex)))
        prev = t[(t.PatientID == pid) & (t.exam == pex)]
        if tf is None or prev.empty:
            continue
        taken = set(g.nodule_id[~g.is_new.astype(bool)])
        cand = prev[~prev.nodule_id.isin(taken)]
        if cand.empty:
            continue
        pp = tf(cand[["x", "y", "z"]].to_numpy(float))
        for i in new_idx:
            d = np.linalg.norm(pp - t.loc[i, ["x", "y", "z"]].to_numpy(float), axis=1)
            cos = en[cand.index] @ en[i]
            ratio = np.maximum(cand.mean_d.to_numpy(), t.loc[i, "mean_d"]) / np.maximum(
                np.minimum(cand.mean_d.to_numpy(), t.loc[i, "mean_d"]), 1e-3)
            ok = (d <= max_mm) & (cos >= min_cos) & (ratio <= 2.0) & ~cand.nodule_id.isin(taken).to_numpy()
            if not ok.any():
                continue
            j = int(np.argmin(np.where(ok, d, np.inf)))
            c = cand.iloc[j]
            taken.add(c.nodule_id)
            dt = t.loc[i, "exam_day"] - c.exam_day
            t.loc[i, ["nodule_id", "prior_volume", "prior_mean_d", "days_since_prior", "match_dist_mm"]] = \
                [c.nodule_id, c.volume, c.mean_d, dt, d[j]]
            t.loc[i, "volume_change_pct"] = (t.loc[i, "volume"] / c.volume - 1) * 100 if c.volume > 0 else np.nan
            t.loc[i, "vdt_days"] = vdt_days(c.volume, t.loc[i, "volume"], dt)
            t.loc[i, "lungrads_growth"] = lungrads_growth(c.mean_d, t.loc[i, "mean_d"], dt)
            t.loc[i, ["is_new", "is_new_confident", "rematched"]] = [False, False, True]
    return t


def features(t: pd.DataFrame, emb: np.ndarray, prior_emb: np.ndarray, pca: PCA) -> tuple[pd.DataFrame, pd.DataFrame]:
    z = pca.transform(np.nan_to_num(emb))
    zp = pca.transform(np.nan_to_num(prior_emb))
    base = pd.DataFrame({
        "det_prob": t.det_prob, "malignancy": t.malignancy, "mean_d": t.mean_d, "log_vol": np.log1p(t.volume),
        "elong": t.long_d / t.short_d.clip(lower=0.5), "prior_nearest_mm": t.get("prior_nearest_mm"),
        "exam": t.exam}, index=t.index)
    for k in range(z.shape[1]):
        base[f"pc{k}"] = z[:, k]
    grow = base.drop(columns=["prior_nearest_mm"]).copy()
    grow["prior_mean_d"] = t.prior_mean_d
    grow["d_diam"] = t.mean_d - t.prior_mean_d
    grow["vol_chg"] = t.volume_change_pct.clip(-100, 1000)
    grow["inv_vdt"] = (1.0 / t.vdt_days.replace(0, np.nan)).clip(-0.05, 0.05).fillna(0.0)
    grow["match_dist"] = t.match_dist_mm
    for k in range(z.shape[1]):
        grow[f"dpc{k}"] = z[:, k] - zp[:, k]
    return base, grow


def mil_fit_predict(Xtr, ytr, Xte):
    clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=250, l2_regularization=1.0,
                                         class_weight="balanced", random_state=0)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1], clf


def exam_metrics(exam_score: pd.Series, exam_label: pd.Series, thr: float) -> dict:
    o, r = exam_score >= thr, exam_label.astype(bool)
    tp = int((o & r).sum())
    return {"exams": int(len(r)), "rad": int(r.sum()), "ours": int(o.sum()), "both": tp,
            "sens": round(tp / max(1, r.sum()), 3), "ppv": round(tp / max(1, o.sum()), 3),
            "f1": round(2 * tp / max(1, o.sum() + r.sum()), 3)}


def best_threshold(score: pd.Series, label: pd.Series) -> float:
    grid = np.unique(np.quantile(score, np.linspace(0.5, 0.995, 120)))
    f1 = [exam_metrics(score, label, g)["f1"] for g in grid]
    return float(grid[int(np.argmax(f1))])


def safety_threshold(score: pd.Series, label: pd.Series, min_sens: float) -> float:
    """最高的門檻，使 label 為真的檢查至少 min_sens 被保留（篩檢寧可多留，不要漏掉真的大結節）。"""
    pos = np.sort(score[label.astype(bool)].to_numpy())
    if not len(pos):
        return 0.0
    return float(pos[int(np.floor((1 - min_sens) * len(pos)))])


def run_task(name, rows: pd.Index, X: pd.DataFrame, t: pd.DataFrame, labels: pd.DataFrame, lab: str,
             dev_pts: set, folds: int) -> tuple[pd.Series, dict]:
    """rows：參與的候選。回傳每顆候選在 test 病人上的機率（dev 病人用 OOF）與 test 指標。"""
    d = t.loc[rows, ["PatientID", "exam"]].merge(labels, on=["PatientID", "exam"], how="left")
    y = d[lab].fillna(False).astype(int).to_numpy()
    is_dev = d.PatientID.isin(dev_pts).to_numpy()
    Xr = X.loc[rows].reset_index(drop=True)
    prob = np.zeros(len(rows))
    dev_i = np.where(is_dev)[0]
    for tr, te in GroupKFold(folds).split(dev_i, y[dev_i], d.PatientID.iloc[dev_i]):
        prob[dev_i[te]], _ = mil_fit_predict(Xr.iloc[dev_i[tr]], y[dev_i[tr]], Xr.iloc[dev_i[te]])
    test_i = np.where(~is_dev)[0]
    prob[test_i], _ = mil_fit_predict(Xr.iloc[dev_i], y[dev_i], Xr.iloc[test_i])
    d["prob"] = prob
    return pd.Series(prob, index=rows, name=f"{name}_prob"), d


def evaluate(cand: pd.DataFrame, exams: pd.DataFrame, labels: pd.DataFrame, lab: str, dev_pts: set) -> dict:
    """候選分數 → 檢查分數（最大值；沒有候選的檢查 = 0）；dev 上選門檻，test 上報告。"""
    s = cand.groupby(["PatientID", "exam"]).prob.max().rename("score")
    e = exams.merge(s, on=["PatientID", "exam"], how="left").merge(labels, on=["PatientID", "exam"], how="left")
    e["score"] = e.score.fillna(0.0)
    e[lab] = e[lab].fillna(False)
    dev = e.PatientID.isin(dev_pts)
    thr = best_threshold(e.score[dev], e[lab][dev])
    return {"threshold": thr, "dev": exam_metrics(e.score[dev], e[lab][dev], thr),
            "test": exam_metrics(e.score[~dev], e[lab][~dev], thr)}, thr


def baseline(t: pd.DataFrame, exams: pd.DataFrame, labels: pd.DataFrame, col: str, lab: str, dev_pts: set,
             min_d: float = 4.0) -> dict:
    o = t[t[col].astype(bool) & (t.mean_d >= min_d)].groupby(["PatientID", "exam"]).size().rename("n")
    e = exams.merge(o, on=["PatientID", "exam"], how="left").merge(labels, on=["PatientID", "exam"], how="left")
    e["score"] = (e.n.fillna(0) > 0).astype(float)
    e[lab] = e[lab].fillna(False)
    test = ~e.PatientID.isin(dev_pts)
    return exam_metrics(e.score[test], e[lab][test], 0.5)


def radiologist_max_diameter(ctab: pd.DataFrame) -> pd.DataFrame:
    """每次篩檢醫師記錄的最大非鈣化結節長徑（mm，代碼 51）；沒記錄 = 0。"""
    a = ctab.copy()
    a["PatientID"] = a.dicom_patient_id.astype(str)
    a["exam"] = pd.to_numeric(a.study_yr, errors="coerce")
    a["ld"] = pd.to_numeric(a.sct_long_dia, errors="coerce")
    a = a[a.sct_ab_desc.astype(str).str.strip() == "51"]
    return a.groupby(["PatientID", "exam"]).ld.max().rename("r_max_d").reset_index()


LARGE_MM = 15.0
LARGE_KEEP_SENS = 0.95  # 安全門檻：dev 上醫師記錄 ≥ 15 mm 的檢查至少保留 95%
LARGE_TRUE_MM = 10.0   # 醫師記錄 ≥ 10 mm 才算「這次檢查真的有大結節」（我們量的平均徑和醫師的長徑有落差，放寬一點）


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--run_dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ctabc", default=None)
    ap.add_argument("--ctab", default=None)
    ap.add_argument("--rematch_mm", type=float, default=20.0)
    ap.add_argument("--min_cos", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    t = pd.read_csv(a.tracks, dtype={"PatientID": str})
    for c in ("is_new", "is_new_confident", "lungrads_growth", "comparable"):
        t[c] = t[c].astype(str).str.lower().isin(["true", "1"])
    ctabc = pd.read_parquet(a.ctabc or sorted(glob.glob(str(Path.home() / ".local/share/IDC/*/clinical_data/nlst_ctabc")))[-1])
    labels = radiologist_labels(ctabc)
    emb = load_embeddings(a.run_dirs, t)
    transforms = load_transforms(a.run_dirs)
    pts = np.array(sorted(t.PatientID.unique()))
    rng = np.random.default_rng(a.seed)
    dev_pts = set(rng.choice(pts, len(pts) // 2, replace=False))
    exams = t[t.comparable & (t.exam > 0)][["PatientID", "exam"]].drop_duplicates()

    res = {"exams_comparable": int(len(exams)), "test_patients": int(len(pts) - len(dev_pts))}
    res["before"] = {"new": baseline(t, exams, labels, "is_new", "r_new", dev_pts),
                     "new_confident": baseline(t, exams, labels, "is_new_confident", "r_new", dev_pts),
                     "grew": baseline(t, exams, labels, "lungrads_growth", "r_grew", dev_pts, 0.0)}

    t = rematch(t, emb, transforms, a.rematch_mm, a.min_cos)
    res["rematched"] = int(t.rematched.sum())
    res["after_rematch"] = {"new": baseline(t, exams, labels, "is_new", "r_new", dev_pts),
                            "grew": baseline(t, exams, labels, "lungrads_growth", "r_grew", dev_pts, 0.0)}

    # 前次配對結節的特徵（給變大分類器算前後差）
    key = t.set_index(["PatientID", "nodule_id", "exam"]).index
    pos = {k: i for i, k in enumerate(key)}
    prev_exam = t.groupby("PatientID").exam.transform(lambda s: s.map(
        {e: (sorted(set(s))[sorted(set(s)).index(e) - 1] if sorted(set(s)).index(e) else -1) for e in set(s)}))
    prior_emb = np.full_like(emb, np.nan)
    for i, (p, n, pe) in enumerate(zip(t.PatientID, t.nodule_id, prev_exam)):
        j = pos.get((p, n, pe))
        if j is not None:
            prior_emb[i] = emb[j]
    ok = ~np.isnan(emb).any(1)
    pca = PCA(PCA_DIM, random_state=0).fit(emb[ok & t.PatientID.isin(dev_pts).to_numpy()])
    base, grow = features(t, emb, prior_emb, pca)

    comp = t.comparable & (t.exam > 0)
    new_rows = t.index[comp & t.is_new & (t.mean_d >= 4.0)]
    grow_rows = t.index[comp & ~t.is_new & t.prior_volume.notna()]
    new_p, new_c = run_task("new", new_rows, base, t, labels, "r_new", dev_pts, 5)
    grow_p, grow_c = run_task("grow", grow_rows, grow, t, labels, "r_grew", dev_pts, 5)
    res["new_classifier"], thr_new = evaluate(new_c, exams, labels, "r_new", dev_pts)
    res["grow_classifier"], thr_grow = evaluate(grow_c, exams, labels, "r_grew", dev_pts)

    # 3) 大結節（≥ 15 mm）過濾器：所有檢查（含 T0）。標籤 = 醫師這次有記錄 ≥ 10 mm 的非鈣化結節
    ctab = pd.read_parquet(a.ctab or sorted(glob.glob(str(Path.home() / ".local/share/IDC/*/clinical_data/nlst_ctab")))[-1])
    big_lab = radiologist_max_diameter(ctab)
    big_lab["r_large"] = big_lab.r_max_d >= LARGE_TRUE_MM
    big_rows = t.index[t.mean_d >= LARGE_MM]
    big_X = base.drop(columns=["prior_nearest_mm"])
    big_p, big_c = run_task("large", big_rows, big_X, t, big_lab[["PatientID", "exam", "r_large"]], "r_large", dev_pts, 5)
    big_exams = t.loc[big_rows, ["PatientID", "exam"]].drop_duplicates()
    res["large_classifier"], thr_big = evaluate(big_c, big_exams, big_lab[["PatientID", "exam", "r_large"]],
                                                "r_large", dev_pts)
    res["large_before"] = {"exams_with_candidate_ge15mm": int(len(big_exams)),
                           "radiologist_ge10mm_share": round(float(big_exams.merge(big_lab, how="left")
                                                                   .r_large.fillna(False).mean()), 3)}
    # F1 門檻（上面）會漏掉約三成醫師也記錄 ≥ 15 mm 的大結節，不能拿來改報告。實際用「安全門檻」：
    # dev 上醫師記錄 ≥ 15 mm 的檢查至少保留 LARGE_KEEP_SENS
    s = big_c.groupby(["PatientID", "exam"]).prob.max().rename("score").reset_index()
    s = big_exams.merge(s, how="left").merge(big_lab, how="left")
    s["r15"] = s.r_max_d.fillna(0) >= LARGE_MM
    s["r10"] = s.r_max_d.fillna(0) >= LARGE_TRUE_MM
    dev = s.PatientID.isin(dev_pts)
    thr_safe = safety_threshold(s.score[dev], s.r15[dev], LARGE_KEEP_SENS)
    res["large_safe"] = {"threshold": thr_safe, "min_dev_sens_rad_ge15mm": LARGE_KEEP_SENS}
    for name, m in (("dev", dev), ("test", ~dev)):
        k = s.score[m] >= thr_safe
        res["large_safe"][name] = {
            "exams": int(m.sum()), "kept": int(k.sum()), "dropped": int((~k).sum()),
            "dropped_no_radiologist_nodule": int((~k & s.r_max_d[m].isna()).sum()),
            "sens_rad_ge15mm": round(float(k[s.r15[m]].mean()), 3), "sens_rad_ge10mm": round(float(k[s.r10[m]].mean()), 3),
            "ppv_rad_ge10mm_kept": round(float(s.r10[m][k].mean()), 3),
            "ppv_rad_ge10mm_before": round(float(s.r10[m].mean()), 3)}
    t["large_prob"] = big_p.reindex(t.index)
    t["large_real"] = t.mean_d.lt(LARGE_MM) | (t.large_prob.fillna(0) >= thr_safe)

    t["new_prob"] = new_p.reindex(t.index)
    t["grow_prob"] = grow_p.reindex(t.index)
    t["is_new_refined"] = t.new_prob.fillna(0) >= thr_new
    t["growth_refined"] = t.grow_prob.fillna(0) >= thr_grow
    # 簡單可解釋的版本：Lung-RADS 直徑 +1.5 mm 且體積 +30%（2026-11-15 test：F1 0.49，比單用直徑 0.46 好）
    t["growth_confident"] = t.lungrads_growth.astype(bool) & (t.volume_change_pct > 30)
    t["split"] = np.where(t.PatientID.isin(dev_pts), "dev", "test")
    t.drop(columns=["emb_row"], errors="ignore").to_csv(out / "tracks_refined.csv", index=False)
    json.dump(res, open(out / "refine.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
