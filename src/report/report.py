"""
report.py — 由 Study + Lung-RADS 分級產生結構化、可追溯的報告

每一句 Sentence 都帶 sources（指向 Study 的欄位路徑，如 nodules[0].diameter_mm），
因此可以逐句核對「這個數字是哪個模型輸出的」。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .lungrads import MANAGEMENT, assess_study, is_growing
from .schema import Study

TYPE_LABEL = {"solid": ("solid", "實性"), "part_solid": ("part-solid", "部分實性"),
              "ground_glass": ("ground-glass", "毛玻璃")}
SOURCE_LABEL = {"detector_box": ("detector bounding box", "偵測框估計"),
                "segmentation": ("segmentation", "分割量測"),
                "tracking": ("tracking pipeline", "追蹤管線量測"), "manual": ("manual measurement", "人工量測")}

DISCLAIMER = {
    "en": "AI-generated draft for research use. Not a diagnosis. Must be reviewed and signed by a radiologist.",
    "zh": "本報告為 AI 產生之研究用草稿，並非診斷，須經放射科醫師審閱簽核。",
}


@dataclass
class Sentence:
    en: str
    zh: str
    sources: list[str] = field(default_factory=list)


def _fmt_xyz(c) -> str:
    return f"({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f})"


def build_report(study: Study, malignancy_review: float = 0.5) -> dict:
    exam_cat, per = assess_study(study, malignancy_review)
    findings: list[Sentence] = []
    if not study.nodules:
        findings.append(Sentence(
            f"No nodule candidates above detector threshold {study.det_threshold:.2f}.",
            f"偵測器在門檻 {study.det_threshold:.2f} 以上未發現結節候選。", ["det_threshold", "nodules"]))
    for i, (n, a) in enumerate(zip(study.nodules, per)):
        p = f"nodules[{i}]"
        ten, tzh = TYPE_LABEL[n.nodule_type]
        sen, szh = SOURCE_LABEL.get(n.diameter_source, (n.diameter_source, n.diameter_source))
        assumed_en = " (type assumed; detector does not classify nodule type)" if n.type_source == "assumed" else ""
        assumed_zh = "（型態為預設值，偵測器無法分辨型態）" if n.type_source == "assumed" else ""
        findings.append(Sentence(
            f"Nodule {n.nodule_id}: {ten}{assumed_en}, mean diameter {n.diameter_mm:.1f} mm ({sen}), "
            f"at {_fmt_xyz(n.center_mm)} mm, detector confidence {n.det_prob:.2f}.",
            f"結節 {n.nodule_id}：{tzh}{assumed_zh}，平均直徑 {n.diameter_mm:.1f} mm（{szh}），"
            f"座標 {_fmt_xyz(n.center_mm)} mm，偵測信心 {n.det_prob:.2f}。",
            [f"{p}.nodule_type", f"{p}.type_source", f"{p}.diameter_mm", f"{p}.diameter_source",
             f"{p}.center_mm", f"{p}.det_prob"]))
        if not study.baseline:
            if not n.comparable:
                findings.append(Sentence(
                    f"Nodule {n.nodule_id}: no usable prior-exam registration; compared as baseline.",
                    f"結節 {n.nodule_id}：無法與前次影像對位，以基線規則判讀。", [f"{p}.comparable"]))
            elif n.is_new and n.new_confident is False:
                near_en = near_zh = ""
                srcs = [f"{p}.is_new", f"{p}.new_confident"]
                if n.prior_nearest_mm is not None:
                    near_en = f"; a prior-exam candidate lies {n.prior_nearest_mm:.1f} mm away"
                    near_zh = f"；前次影像在 {n.prior_nearest_mm:.1f} mm 處有候選"
                    srcs.append(f"{p}.prior_nearest_mm")
                findings.append(Sentence(
                    f"Nodule {n.nodule_id}: possibly new (low-confidence match{near_en}); compare with the prior "
                    f"exam manually. Graded by size only.",
                    f"結節 {n.nodule_id}：可能為新出現（配對信心低{near_zh}），請人工對照前次影像；僅依大小分級。",
                    srcs))
            elif n.is_new:
                findings.append(Sentence(
                    f"Nodule {n.nodule_id}: new since the prior exam ({n.days_since_prior:.0f} days ago).",
                    f"結節 {n.nodule_id}：前次檢查（{n.days_since_prior:.0f} 天前）未見，為新出現。",
                    [f"{p}.is_new", f"{p}.days_since_prior"]))
                if n.diameter_mm >= 6:
                    findings.append(Sentence(
                        f"Nodule {n.nodule_id}: new nodule of {n.diameter_mm:.1f} mm; verify on the prior images, "
                        f"because automatic matching can miss an existing nodule.",
                        f"結節 {n.nodule_id}：新出現且達 {n.diameter_mm:.1f} mm，請對照前次影像確認，"
                        f"自動配對可能漏配既有結節。", [f"{p}.is_new", f"{p}.diameter_mm"]))
            elif n.prior_diameter_mm is not None:
                delta = n.diameter_mm - n.prior_diameter_mm
                vol_en = vol_zh = ""
                srcs = [f"{p}.prior_diameter_mm", f"{p}.diameter_mm", f"{p}.days_since_prior"]
                if n.volume_change_pct is not None:
                    vol_en = f", volume change {n.volume_change_pct:+.0f}%"
                    vol_zh = f"，體積變化 {n.volume_change_pct:+.0f}%"
                    srcs.append(f"{p}.volume_change_pct")
                if n.vdt_days is not None and n.vdt_days > 0 and delta > 0:
                    vol_en += f", volume doubling time {n.vdt_days:.0f} days"
                    vol_zh += f"，體積倍增時間 {n.vdt_days:.0f} 天"
                    srcs.append(f"{p}.vdt_days")
                findings.append(Sentence(
                    f"Nodule {n.nodule_id}: prior mean diameter {n.prior_diameter_mm:.1f} mm "
                    f"{n.days_since_prior:.0f} days ago, change {delta:+.1f} mm{vol_en}.",
                    f"結節 {n.nodule_id}：{n.days_since_prior:.0f} 天前平均直徑 {n.prior_diameter_mm:.1f} mm，"
                    f"變化 {delta:+.1f} mm{vol_zh}。", srcs))
                if n.rematched:
                    findings.append(Sentence(
                        f"Nodule {n.nodule_id}: matched to the prior nodule by image appearance (not by position "
                        f"alone); confirm the pairing.",
                        f"結節 {n.nodule_id}：依影像特徵與前次結節配對（非僅依位置），請確認配對是否正確。",
                        [f"{p}.rematched"]))
                if is_growing(n) and n.growth_confident is not None:
                    findings.append(Sentence(
                        f"Nodule {n.nodule_id}: " + ("volume increase of 30% or more supports the growth."
                                                     if n.growth_confident else
                                                     "growth is by diameter only; volume increase is under 30%, "
                                                     "so measurement noise is possible."),
                        f"結節 {n.nodule_id}：" + ("體積增加 30% 以上，支持增大判定。" if n.growth_confident else
                                                  "僅直徑達增大標準，體積增加未達 30%，可能是量測誤差。"),
                        [f"{p}.growth_confident", f"{p}.volume_change_pct"]))
            if (n.comparable and not n.is_new and n.upstream_growth_flag is not None
                    and n.upstream_growth_flag != is_growing(n)):
                findings.append(Sentence(
                    f"Nodule {n.nodule_id}: growth call disagrees with the tracking pipeline flag; check manually.",
                    f"結節 {n.nodule_id}：增大判定與追蹤管線的旗標不一致，請人工確認。",
                    [f"{p}.upstream_growth_flag", f"{p}.prior_diameter_mm"]))
        if n.volume_mm3 is not None:
            seg_vol = n.diameter_source == "segmentation"
            findings.append(Sentence(
                f"Nodule {n.nodule_id}: {'segmented ' if seg_vol else ''}volume {n.volume_mm3:.0f} mm3.",
                f"結節 {n.nodule_id}：{'分割' if seg_vol else ''}體積 {n.volume_mm3:.0f} mm³。", [f"{p}.volume_mm3"]))
        if n.nodule_type == "part_solid" and n.solid_component_mm is not None:
            findings.append(Sentence(f"Nodule {n.nodule_id}: solid component {n.solid_component_mm:.1f} mm.",
                                     f"結節 {n.nodule_id}：實性成分 {n.solid_component_mm:.1f} mm。",
                                     [f"{p}.solid_component_mm"]))
        findings.append(Sentence(f"Nodule {n.nodule_id}: Lung-RADS {a.category} ({a.rule}).",
                                 f"結節 {n.nodule_id}：Lung-RADS {a.category}（{a.rule}）。",
                                 [f"{p}.{a.measure_used}", f"{p}.nodule_type"]))
        if n.malignancy_prob is not None:
            findings.append(Sentence(
                f"Nodule {n.nodule_id}: AI malignancy risk score {n.malignancy_prob:.2f} (0-1, not a diagnosis).",
                f"結節 {n.nodule_id}：AI 惡性風險分數 {n.malignancy_prob:.2f}（0–1，非診斷）。",
                [f"{p}.malignancy_prob"]))
        if a.flag_4x_review:
            findings.append(Sentence(
                f"Nodule {n.nodule_id}: AI risk score {n.malignancy_prob:.2f} >= {malignancy_review:.2f}; "
                f"radiologist to assess for suspicious features (possible 4X).",
                f"結節 {n.nodule_id}：AI 風險分數 {n.malignancy_prob:.2f} ≥ {malignancy_review:.2f}，"
                f"請放射科醫師評估是否有可疑特徵（可能為 4X）。",
                [f"{p}.malignancy_prob", "malignancy_review_threshold"]))

    mgmt = MANAGEMENT[exam_cat if exam_cat in MANAGEMENT else "2"]
    impression = Sentence(f"Lung-RADS {exam_cat}. {mgmt['en']}", f"Lung-RADS {exam_cat}。{mgmt['zh']}",
                          ["exam_category"] + [f"nodules[{i}]" for i, a in enumerate(per) if a.category == exam_cat])

    if study.baseline:
        lim0 = Sentence("Baseline exam: growth and new-nodule rules need a prior exam and are not applied.",
                        "基線檢查：增大與新生結節規則需要前次影像，未套用。", ["baseline"])
    else:
        lim0 = Sentence("Follow-up rules are applied to solid nodules only; nodule matching across exams is "
                        "automatic and new nodules may be detector false positives.",
                        "追蹤規則只套用於實性結節；跨次檢查的結節配對為自動完成，新出現的結節可能是偵測器誤報。",
                        ["baseline", "nodules[*].is_new"])
    limitations = [
        lim0,
        Sentence("Perifissural/juxtapleural benign features, airway and cystic lesions, and S findings "
                 "are not assessed.", "未評估肺裂旁/胸膜旁良性特徵、氣道與囊性病灶，以及 S 修飾碼的其他發現。", []),
    ]
    if any(n.type_source == "assumed" for n in study.nodules):
        limitations.append(Sentence("Nodule type is assumed solid; part-solid or ground-glass nodules may be "
                                    "mis-categorized.", "結節型態預設為實性，部分實性或毛玻璃結節可能分級不準。",
                                    ["nodules[*].type_source"]))
    if study.models.segmentation and "MOCK" in study.models.segmentation:
        limitations.append(Sentence("Segmentation values are MOCK placeholders.", "分割數值為模擬（MOCK）資料。",
                                    ["models.segmentation"]))

    return {
        "study_id": study.study_id,
        "baseline": study.baseline,
        "exam_category": exam_cat,
        "management": {"en": mgmt["en"], "zh": mgmt["zh"], "followup_months": mgmt["months"]},
        "nodule_assessments": [asdict(a) for a in per],
        "findings": [asdict(s) for s in findings],
        "impression": asdict(impression),
        "limitations": [asdict(s) for s in limitations],
        "malignancy_review_threshold": malignancy_review,
        "disclaimer": DISCLAIMER,
        "inputs": study.to_dict(),
    }


def to_markdown(rep: dict, lang: str = "zh") -> str:
    t = {"zh": ("Lung-RADS 結構化報告（草稿）", "檢查", "發現", "印象與建議", "限制", "依據欄位"),
         "en": ("Lung-RADS Structured Report (draft)", "Study", "Findings", "Impression", "Limitations",
                "sources")}[lang]
    lines = [f"# {t[0]}", "", f"**{t[1]}**: `{rep['study_id']}`", "", f"## {t[2]}", ""]
    lines += [f"- {s[lang]}  <sub>{t[5]}: {', '.join(s['sources'])}</sub>" for s in rep["findings"]]
    imp = rep["impression"]
    lines += ["", f"## {t[3]}", "", f"**{imp[lang]}**  <sub>{t[5]}: {', '.join(imp['sources'])}</sub>", ""]
    lines += [f"## {t[4]}", ""] + [f"- {s[lang]}" for s in rep["limitations"]]
    if rep.get("patient"):
        p = rep["patient"]
        head = "給病人的說明" if lang == "zh" else "Patient explanation"
        lines += ["", f"## {head}", "", p[lang]["text"], "",
                  f"<sub>{p[lang]['backend']}{' — ' + p[lang]['guard'] if p[lang].get('guard') else ''}</sub>"]
    lines += ["", f"> {rep['disclaimer'][lang]}", ""]
    return "\n".join(lines)
