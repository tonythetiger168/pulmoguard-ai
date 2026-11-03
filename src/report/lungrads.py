"""
lungrads.py — 規則式 Lung-RADS v2022 分級（基線篩檢）

只依「量測數字 + 型態」分級，不依 AI 惡性度自動升級：
AI 惡性度高時只會加上「建議放射科醫師評估是否為 4X」的旗標，最終判讀屬於醫師。

尚未涵蓋（報告中會列為限制）：追蹤檢查的新生/增大規則、肺裂旁/胸膜旁良性特徵、
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


def assess_nodule(n: Nodule, malignancy_review: float = 0.5) -> NoduleAssessment:
    fn = {"solid": _solid, "part_solid": _part_solid, "ground_glass": _ground_glass}[n.nodule_type]
    cat, rule, used = fn(n)
    flag = (cat in ("3", "4A", "4B") and n.malignancy_prob is not None
            and n.malignancy_prob >= malignancy_review)
    return NoduleAssessment(n.nodule_id, cat, rule, used, flag)


def max_category(cats: list[str]) -> str:
    return max(cats, key=CATEGORY_ORDER.index) if cats else "1"


def assess_study(study: Study, malignancy_review: float = 0.5) -> tuple[str, list[NoduleAssessment]]:
    if not study.baseline:
        raise NotImplementedError("目前只支援基線篩檢；追蹤比較（新生/增大）規則尚未實作")
    per = [assess_nodule(n, malignancy_review) for n in study.nodules]
    return max_category([a.category for a in per]), per
