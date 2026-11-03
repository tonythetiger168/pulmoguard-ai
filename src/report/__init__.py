"""Lung-RADS 結構化報告原型：影像模型輸出 → 規則分級 → 可追溯報告 →（選配）LLM 病人說明。"""
from .lungrads import assess_nodule, assess_study
from .patient import guard, openai_compatible, patient_explanations
from .report import build_report, to_markdown
from .schema import ModelInfo, Nodule, Study

__all__ = ["Nodule", "Study", "ModelInfo", "assess_nodule", "assess_study", "build_report", "to_markdown",
           "patient_explanations", "openai_compatible", "guard"]
