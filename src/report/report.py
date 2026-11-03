"""
report.py — 由 Study + Lung-RADS 分級產生結構化、可追溯的報告

每一句 Sentence 都帶 sources（指向 Study 的欄位路徑，如 nodules[0].diameter_mm），
因此可以逐句核對「這個數字是哪個模型輸出的」。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .lungrads import MANAGEMENT, assess_study
from .schema import Study

TYPE_LABEL = {"solid": ("solid", "實性"), "part_solid": ("part-solid", "部分實性"),
              "ground_glass": ("ground-glass", "毛玻璃")}
SOURCE_LABEL = {"detector_box": ("detector bounding box", "偵測框估計"),
                "segmentation": ("segmentation", "分割量測"), "manual": ("manual measurement", "人工量測")}

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
        if n.volume_mm3 is not None:
            findings.append(Sentence(f"Nodule {n.nodule_id}: segmented volume {n.volume_mm3:.0f} mm3.",
                                     f"結節 {n.nodule_id}：分割體積 {n.volume_mm3:.0f} mm³。", [f"{p}.volume_mm3"]))
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

    limitations = [
        Sentence("Baseline rules only; growth and new-nodule rules need a prior exam and are not applied.",
                 "僅套用基線規則；增大與新生結節規則需要前次影像，尚未套用。", ["baseline"]),
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
