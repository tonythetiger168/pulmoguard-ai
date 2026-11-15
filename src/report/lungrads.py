"""
lungrads.py — 規則式 Lung-RADS v2022 分級（基線篩檢 + 實性結節的追蹤檢查）

只依「量測數字 + 型態」分級，不依 AI 惡性度自動升級：
AI 惡性度高時只會加上「建議放射科醫師評估是否為 4X」的旗標，最終判讀屬於醫師。

追蹤檢查（實性結節）：新生 <4/4–6/6–8/≥8 mm → 2/3/4A/4B；增大（平均直徑增加 >1.5 mm）<8 mm → 4A、
≥8 mm → 4B；穩定結節套用 v2022 的階梯式降級（3 類穩定 ≥6 個月 → 2；4A 穩定 ≥3 個月 → 3、≥8 個月 → 2）。
部分實性/毛玻璃結節的追蹤規則尚未實作，會退回基線規則並標註。

尚未涵蓋（報告中會列為限制）：肺裂旁/胸膜旁良性特徵、
支氣管內結節、氣道/囊性病灶、S 修飾碼（顯著的其他發現）。
參考：ACR Lung-RADS v2022 Assessment Categories。
"""
from __future__ import annotations

from dataclasses import dataclass

from .schema import Nodule, Study

CATEGORY_ORDER = ["0", "1", "2", "3", "4A", "4B", "4X"]

# v2022 實性結節體積門檻（mm³），有分割體積時優先使用
SOLID_VOL_MM3 = (113.0, 268.0, 1767.0)

MANAGEMENT = {
    "1": {"months": 12, "en": "Continue annual screening with LDCT in 12 months.",
          "zh": "12 個月後回到每年一次的低劑量電腦斷層（LDCT）篩檢。"},
    "2": {"months": 12, "en": "Continue annual screening with LDCT in 12 months.",
          "zh": "12 個月後回到每年一次的低劑量電腦斷層（LDCT）篩檢。"},
    "3": {"months": 6, "en": "LDCT in 6 months.",
          "zh": "6 個月後追蹤低劑量電腦斷層（LDCT）。"},
    "4A": {"months": 3, "en": "LDCT in 3 months; PET/CT may be considered if there is a solid nodule "
                              "or solid component of 8 mm or larger.",
           "zh": "3 個月後追蹤低劑量電腦斷層；若實性結節或實性成分達 8 mm 以上，可考慮 PET/CT。"},
    "4B": {"months": None, "en": "Diagnostic chest CT with or without contrast; PET/CT may be considered if "
                                 "there is a solid component of 8 mm or larger; tissue sampling and/or "
                                 "referral for further clinical evaluation.",
           "zh": "安排診斷性胸部電腦斷層（可含顯影劑）；實性成分達 8 mm 以上可考慮 PET/CT；"
                 "視情況進行切片或轉介進一步評估。"},
}


@dataclass
class NoduleAssessment:
    nodule_id: int
    category: str
    rule: str  # 觸發的規則（人類可讀，供追溯）
    measure_used: str  # 用了哪個欄位做判斷
    flag_4x_review: bool = False
    excluded: bool = False  # 大結節誤報過濾器排除：不計入整體分級


def _solid(n: Nodule) -> tuple[str, str, str]:
    if n.volume_mm3 is not None and n.diameter_source == "segmentation":
        v = n.volume_mm3
        a, b, c = SOLID_VOL_MM3
        cat = "2" if v < a else "3" if v < b else "4A" if v < c else "4B"
        return cat, f"solid, volume {v:.0f} mm3 (thresholds 113/268/1767 mm3)", "volume_mm3"
    d = round(n.diameter_mm, 1)
    cat = "2" if d < 6 else "3" if d < 8 else "4A" if d < 15 else "4B"
    return cat, f"solid, mean diameter {d:.1f} mm (thresholds 6/8/15 mm)", "diameter_mm"


def _part_solid(n: Nodule) -> tuple[str, str, str]:
    d = round(n.diameter_mm, 1)
    if d < 6:
        return "2", f"part-solid, total diameter {d:.1f} mm < 6 mm", "diameter_mm"
    s = n.solid_component_mm
    if s is None:
        # 沒有實性成分量測：保守地以 3 起算，並要求醫師量測
        return "3", f"part-solid, total {d:.1f} mm >= 6 mm, solid component not measured", "diameter_mm"
    s = round(s, 1)
    cat = "3" if s < 6 else "4A" if s < 8 else "4B"
    return cat, f"part-solid, solid component {s:.1f} mm (thresholds 6/8 mm)", "solid_component_mm"


def _ground_glass(n: Nodule) -> tuple[str, str, str]:
    d = round(n.diameter_mm, 1)
    cat = "2" if d < 30 else "3"
    return cat, f"ground-glass, mean diameter {d:.1f} mm (threshold 30 mm)", "diameter_mm"


GROWTH_MM = 1.5  # v2022：平均直徑 12 個月內增加 >1.5 mm
STABLE_STEP_DAYS = {"3": [(150, "2")], "4A": [(240, "2"), (80, "3")]}  # 階梯式降級門檻（天）


def is_growing(n: Nodule) -> bool:
    return (n.prior_diameter_mm is not None
            and round(n.diameter_mm - n.prior_diameter_mm, 1) > GROWTH_MM)


def _solid_followup(n: Nodule) -> tuple[str, str, str]:
    d = round(n.diameter_mm, 1)
    if n.is_new and n.new_confident is False:
        cat, rule, used = _solid(n)
        return cat, rule + "; possible new (low confidence match), baseline size rule applied", "is_new"
    if n.is_new:
        cat = "2" if d < 4 else "3" if d < 6 else "4A" if d < 8 else "4B"
        return cat, f"solid, NEW, mean diameter {d:.1f} mm (new thresholds 4/6/8 mm)", "is_new"
    if n.prior_diameter_mm is None:
        cat, rule, used = _solid(n)
        return cat, rule + "; prior size unavailable, baseline rule applied", used
    delta = round(d - n.prior_diameter_mm, 1)
    days = n.days_since_prior or 0
    if delta > GROWTH_MM:
        cat = "4A" if d < 8 else "4B"
        return cat, (f"solid, GROWING {n.prior_diameter_mm:.1f} -> {d:.1f} mm (+{delta:.1f} mm in "
                     f"{days:.0f} days; growth threshold >1.5 mm)"), "prior_diameter_mm"
    size_cat, size_rule, _ = _solid(n)
    for min_days, stepped in STABLE_STEP_DAYS.get(size_cat, []):
        if days >= min_days:
            return stepped, (f"solid, stable {n.prior_diameter_mm:.1f} -> {d:.1f} mm over {days:.0f} days; "
                             f"size category {size_cat} stepped down to {stepped}"), "prior_diameter_mm"
    return size_cat, f"{size_rule}; stable ({delta:+.1f} mm over {days:.0f} days)", "prior_diameter_mm"


def assess_nodule(n: Nodule, malignancy_review: float = 0.5, baseline: bool = True) -> NoduleAssessment:
    if not baseline and n.comparable and n.nodule_type == "solid":
        cat, rule, used = _solid_followup(n)
    else:
        fn = {"solid": _solid, "part_solid": _part_solid, "ground_glass": _ground_glass}[n.nodule_type]
        cat, rule, used = fn(n)
        if not baseline:
            why = ("no usable prior registration" if not n.comparable
                   else "follow-up rules for this nodule type not implemented")
            rule += f"; {why}, baseline rule applied"
    flag = (cat in ("3", "4A", "4B") and n.malignancy_prob is not None
            and n.malignancy_prob >= malignancy_review)
    return NoduleAssessment(n.nodule_id, cat, rule, used, flag)


def _event_based(n: Nodule, baseline: bool) -> bool:
    """分級是否來自新生（確定）或增大事件，而非單純大小。"""
    if baseline or not n.comparable or n.nodule_type != "solid":
        return False
    if n.is_new:
        return n.new_confident is not False
    return is_growing(n)


def max_category(cats: list[str]) -> str:
    return max(cats, key=CATEGORY_ORDER.index) if cats else "1"


def assess_study(study: Study, malignancy_review: float = 0.5) -> tuple[str, list[NoduleAssessment]]:
    per = []
    for n in study.nodules:
        a = assess_nodule(n, malignancy_review, study.baseline)
        if n.large_real is False and not _event_based(n, study.baseline):
            a.excluded = True
            a.flag_4x_review = False
            a.rule += "; rejected by large-nodule false-positive filter, excluded from exam category"
        per.append(a)
    return max_category([a.category for a in per if not a.excluded]), per
