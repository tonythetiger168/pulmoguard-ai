"""
patient.py — 給病人看的中英白話說明：規則模板 + 選配 LLM 潤稿 + 防幻覺守門

流程：
1. 先用規則模板產生一定正確的說明（template_text）。
2. 若有設定 LLM，讓它「只改寫」模板，輸入只有結構化事實與模板文字。
3. guard() 檢查 LLM 輸出：
   - 出現的每個數字都必須在模板或報告中出現過（不能新增尺寸、機率、月數、顆數）
   - 必須保留 Lung-RADS 分級，且不能出現其他分級
   - 不能出現確診/排除癌症等斷言
   任一條不通過 → 回退到模板，並記錄原因。

LLM 後端採 OpenAI 相容的 /v1/chat/completions，所以 vLLM、llama.cpp server、Ollama，
以及之後 HeliosLM 若提供相容的 HTTP 服務，都可以直接接上。
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import Callable

from .lungrads import CATEGORY_ORDER

MEANING = {
    "1": ("no lung nodules were found", "沒有發現肺結節"),
    "2": ("the findings look benign and the chance of cancer is very low",
          "看起來是良性表現，是癌症的機會很低"),
    "3": ("the findings are probably benign, but a shorter follow-up scan is recommended to make sure",
          "大多可能是良性，但建議提早追蹤以確認沒有變化"),
    "4A": ("the findings are suspicious and need a closer follow-up",
           "有可疑的地方，需要較密集的追蹤"),
    "4B": ("the findings are suspicious and further tests are recommended",
           "有可疑的地方，建議進一步檢查"),
}

BANNED = [
    r"確診", r"確定是癌", r"確定不是癌", r"排除癌症", r"沒有癌症", r"一定是", r"保證",
    r"\bdiagnosed\b", r"\bconfirm(?:s|ed)? (?:cancer|malignan)", r"\brules? out\b", r"\bno cancer\b",
    r"\bdefinitely\b", r"\bguarantee",
]

NUM_RE = re.compile(r"\d+(?:\.\d+)?")
CAT_RE = re.compile(r"Lung-?RADS\s*(?:category\s*|第\s*)?([0-4][ABX]?)", re.IGNORECASE)


def template_text(rep: dict, lang: str) -> str:
    cat = rep["exam_category"]
    nods = rep["inputs"]["nodules"]
    mgmt = rep["management"][lang]
    meaning = MEANING.get(cat, MEANING["2"])[0 if lang == "en" else 1]
    if lang == "en":
        if not nods:
            body = "This low-dose CT scan did not find any lung nodules that needed attention."
        else:
            biggest = max(n["diameter_mm"] for n in nods)
            body = (f"This low-dose CT scan found {len(nods)} small spot(s) in the lungs (nodules); "
                    f"the largest is about {biggest:.1f} mm.")
        return (f"{body} Using the international Lung-RADS system, this scan is Lung-RADS {cat}, which means "
                f"{meaning}. Recommended next step: {mgmt} The AI scores are only a reference for your doctor; "
                f"please discuss these results with your doctor.")
    if not nods:
        body = "這次低劑量電腦斷層沒有發現需要注意的肺結節。"
    else:
        biggest = max(n["diameter_mm"] for n in nods)
        body = f"這次低劑量電腦斷層在肺部發現 {len(nods)} 顆小結節，最大的約 {biggest:.1f} mm。"
    return (f"{body}依照國際通用的 Lung-RADS 分級，這次屬於 Lung-RADS {cat}，意思是{meaning}。"
            f"建議下一步：{mgmt}AI 分數只是提供醫師參考，請和您的醫師討論這份結果。")


def _norm(x: str) -> str:
    return str(float(x)).rstrip("0").rstrip(".") if "." in x else str(int(x))


def allowed_numbers(rep: dict, lang: str) -> set[str]:
    texts = [template_text(rep, lang), rep["impression"][lang], rep["management"][lang]]
    texts += [s[lang] for s in rep["findings"]]
    return {_norm(m) for t in texts for m in NUM_RE.findall(t)} | {"0", "1"}


def guard(text: str, rep: dict, lang: str) -> list[str]:
    """回傳違規清單；空清單代表通過。"""
    errs = []
    ok = allowed_numbers(rep, lang)
    extra = sorted({_norm(m) for m in NUM_RE.findall(text)} - ok)
    if extra:
        errs.append(f"numbers not in model outputs: {extra}")
    cats = {c.upper() for c in CAT_RE.findall(text)}
    if rep["exam_category"] not in cats:
        errs.append(f"missing Lung-RADS {rep['exam_category']}")
    wrong = cats - {rep["exam_category"]}
    if wrong & set(CATEGORY_ORDER):
        errs.append(f"other Lung-RADS categories mentioned: {sorted(wrong)}")
    for b in BANNED:
        if re.search(b, text, re.IGNORECASE):
            errs.append(f"banned phrase: {b}")
    if len(text) > 3 * len(template_text(rep, lang)) + 200:
        errs.append("too long")
    return errs


SYSTEM_PROMPT = (
    "You rewrite a lung cancer screening result for a patient. You may ONLY rephrase the DRAFT to be warmer and "
    "clearer. Do not add, remove or change any finding, number, size, count, category, probability or follow-up "
    "interval. Do not diagnose, do not say cancer is present or absent. Keep 'Lung-RADS <category>' verbatim. "
    "Reply with the rewritten text only, in the same language as the DRAFT."
)


def build_messages(rep: dict, lang: str) -> list[dict]:
    facts = {"exam_category": rep["exam_category"], "management": rep["management"][lang],
             "nodules": [{k: n[k] for k in ("nodule_id", "diameter_mm", "nodule_type")}
                         for n in rep["inputs"]["nodules"]]}
    user = f"FACTS (read-only):\n{json.dumps(facts, ensure_ascii=False)}\n\nDRAFT:\n{template_text(rep, lang)}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


LLMFn = Callable[[list[dict]], str]


def openai_compatible(base_url: str | None = None, model: str | None = None,
                      api_key: str | None = None, timeout: float = 60.0) -> LLMFn:
    """OpenAI 相容 chat completions。環境變數：PULMOGUARD_LLM_BASE_URL / _MODEL / _API_KEY。"""
    import requests

    base = (base_url or os.environ.get("PULMOGUARD_LLM_BASE_URL", "http://localhost:8000/v1")).rstrip("/")
    mdl = model or os.environ.get("PULMOGUARD_LLM_MODEL", "local")
    key = api_key or os.environ.get("PULMOGUARD_LLM_API_KEY", "none")

    def call(messages: list[dict]) -> str:
        r = requests.post(f"{base}/chat/completions", timeout=timeout,
                          headers={"Authorization": f"Bearer {key}"},
                          json={"model": mdl, "messages": messages, "temperature": 0.2})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    call.backend_name = f"llm:{mdl}@{base}"  # type: ignore[attr-defined]
    return call


def patient_explanations(rep: dict, llm: LLMFn | None = None, langs=("zh", "en")) -> dict:
    out = {}
    for lang in langs:
        tmpl = template_text(rep, lang)
        if llm is None:
            out[lang] = {"text": tmpl, "backend": "template", "guard": None}
            continue
        name = getattr(llm, "backend_name", "llm")
        try:
            text = llm(build_messages(rep, lang))
        except Exception as e:  # 網路/服務錯誤 → 回退
            out[lang] = {"text": tmpl, "backend": "template (fallback)", "guard": f"llm error: {e}"}
            continue
        errs = guard(text, rep, lang)
        if errs:
            out[lang] = {"text": tmpl, "backend": "template (fallback)", "guard": "; ".join(errs),
                         "rejected_llm_text": text}
        else:
            out[lang] = {"text": text, "backend": name, "guard": "passed"}
    return out
