"""
make_lungrads_report.py — 影像管線輸出 → Lung-RADS 結構化報告（JSON + 中英 Markdown）

CPU 即可執行，不需要 GPU。

用法（範例資料）：
    python -m scripts.make_lungrads_report --detections examples/lungrads/detections.json \
        --malignancy examples/lungrads/candidate_scores.csv --seg examples/lungrads/segmentation_mock.json \
        --out runs/reports

接上 LLM 潤稿（任何 OpenAI 相容服務：vLLM、llama.cpp、Ollama、之後的 HeliosLM server）：
    PULMOGUARD_LLM_BASE_URL=http://localhost:11434/v1 PULMOGUARD_LLM_MODEL=qwen2.5:7b-instruct \
        python -m scripts.make_lungrads_report ... --llm openai
LLM 輸出若新增數字、改了分級或出現確診字眼，會自動回退到規則模板。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.report import build_report, openai_compatible, patient_explanations, to_markdown
from src.report.adapters import build_study, load_detector_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections", required=True, help="MONAI 偵測器結果 JSON")
    ap.add_argument("--malignancy", help="LUNA25 candidate_scores.csv")
    ap.add_argument("--seg", help="分割結果 JSON（可為 mock）")
    ap.add_argument("--out", default="runs/reports")
    ap.add_argument("--det_threshold", type=float, default=0.5)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--malignancy_review", type=float, default=0.5, help="AI 分數達此值時提示醫師評估 4X")
    ap.add_argument("--llm", choices=["none", "openai"], default="none")
    a = ap.parse_args()

    dets = load_detector_json(a.detections)
    mal = pd.read_csv(a.malignancy, dtype={"seriesuid": str}) if a.malignancy else None
    seg_all = json.loads(Path(a.seg).read_text()) if a.seg else None
    llm = openai_compatible() if a.llm == "openai" else None
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for uid, d in dets.items():
        # 分割 JSON：單一病例 {"seriesuid": uid, "lesions": [...]} 或多病例 {"<uid>": {"lesions": [...]}}
        seg = None
        if seg_all:
            seg = seg_all if seg_all.get("seriesuid") == uid else seg_all.get(uid)
        study = build_study(uid, d, det_threshold=a.det_threshold, top_k=a.top_k, malignancy=mal, seg=seg)
        rep = build_report(study, a.malignancy_review)
        rep["patient"] = patient_explanations(rep, llm)
        (out / f"{uid}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
        for lang in ("zh", "en"):
            (out / f"{uid}.{lang}.md").write_text(to_markdown(rep, lang))
        print(f"{uid}: Lung-RADS {rep['exam_category']} ({len(study.nodules)} nodules) -> {out}/{uid}.*")


if __name__ == "__main__":
    main()
