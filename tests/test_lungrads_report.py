"""Lung-RADS 報告原型：分級規則、可追溯性、LLM 守門（全部 CPU、無網路）。"""
import json
import subprocess
import sys

import pandas as pd
import pytest

from src.report import Nodule, Study, assess_nodule, build_report, guard, patient_explanations, to_markdown
from src.report.adapters import build_study, load_detector_json


def nod(d, **kw):
    return Nodule(nodule_id=1, center_mm=(0.0, 0.0, 0.0), diameter_mm=d, diameter_source="detector_box",
                  det_prob=0.9, **kw)


@pytest.mark.parametrize("d,cat", [(4.0, "2"), (5.94, "2"), (5.96, "3"), (6.0, "3"), (7.9, "3"),
                                   (8.0, "4A"), (14.9, "4A"), (15.0, "4B"), (30.0, "4B")])
def test_solid_diameter_thresholds(d, cat):
    assert assess_nodule(nod(d)).category == cat


@pytest.mark.parametrize("v,cat", [(100, "2"), (113, "3"), (267, "3"), (268, "4A"), (1767, "4B")])
def test_solid_volume_thresholds(v, cat):
    n = nod(7.0, volume_mm3=v)
    n.diameter_source = "segmentation"
    assert assess_nodule(n).category == cat


def test_part_solid_and_ggn():
    assert assess_nodule(nod(5.0, nodule_type="part_solid")).category == "2"
    assert assess_nodule(nod(10.0, nodule_type="part_solid", solid_component_mm=4.0)).category == "3"
    assert assess_nodule(nod(10.0, nodule_type="part_solid", solid_component_mm=7.0)).category == "4A"
    assert assess_nodule(nod(12.0, nodule_type="part_solid", solid_component_mm=8.0)).category == "4B"
    assert assess_nodule(nod(25.0, nodule_type="ground_glass")).category == "2"
    assert assess_nodule(nod(30.0, nodule_type="ground_glass")).category == "3"


def test_ai_score_never_changes_category_only_flags():
    a = assess_nodule(nod(6.5, malignancy_prob=0.95))
    assert a.category == "3" and a.flag_4x_review
    assert not assess_nodule(nod(4.0, malignancy_prob=0.95)).flag_4x_review  # 類別 2 不提示


def test_empty_study_is_category_1():
    rep = build_report(Study(study_id="x"))
    assert rep["exam_category"] == "1" and rep["management"]["followup_months"] == 12


def test_every_sentence_is_traceable():
    s = Study(study_id="s", nodules=[nod(9.0, malignancy_prob=0.7)])
    rep = build_report(s)
    for sent in rep["findings"] + [rep["impression"]]:
        assert sent["sources"] and sent["en"] and sent["zh"]
        for src in sent["sources"]:
            if src.startswith("nodules["):
                i = int(src[8:src.index("]")])
                field = src.split(".", 1)[1] if "." in src else None
                assert field is None or field in rep["inputs"]["nodules"][i]
    assert rep["exam_category"] == "4A"


def make_rep():
    rep = build_report(Study(study_id="s", nodules=[nod(9.0, malignancy_prob=0.7)]))
    return rep


def test_guard_passes_faithful_rewrite():
    rep = make_rep()
    ok = "您好，這次檢查在肺部看到 1 顆結節，約 9.0 mm，屬於 Lung-RADS 4A。建議 3 個月後再追蹤一次低劑量電腦斷層。"
    assert guard(ok, rep, "zh") == []


@pytest.mark.parametrize("bad,why", [
    ("Lung-RADS 4A. The nodule is 12 mm.", "numbers"),
    ("Lung-RADS 4A, follow up in 9 months... wait 24 months.", "numbers"),
    ("This scan is Lung-RADS 2, nothing to worry about.", "missing"),
    ("Lung-RADS 4A; you have been diagnosed with early cancer.", "banned"),
    ("屬於 Lung-RADS 4A，可以排除癌症。", "banned"),
])
def test_guard_rejects_hallucinations(bad, why):
    lang = "zh" if any("一" <= c <= "鿿" for c in bad) else "en"
    errs = guard(bad, make_rep(), lang)
    assert errs and any(why in e for e in errs), errs


def test_llm_fallback_to_template_on_violation():
    rep = make_rep()
    out = patient_explanations(rep, llm=lambda msgs: "Lung-RADS 4A. Your 20 mm tumour is cancer, diagnosed.")
    for lang in ("zh", "en"):
        assert out[lang]["backend"] == "template (fallback)"
        assert "rejected_llm_text" in out[lang]


def test_llm_error_falls_back():
    def boom(msgs):
        raise ConnectionError("down")
    assert patient_explanations(make_rep(), llm=boom)["en"]["backend"] == "template (fallback)"


def test_llm_prompt_only_contains_facts():
    seen = []
    patient_explanations(make_rep(), llm=lambda m: seen.append(m) or "x", langs=("en",))
    user = seen[0][1]["content"]
    assert "DRAFT" in user and "4A" in user and "9.0" in user


def test_adapters_join_malignancy_and_seg(tmp_path):
    det = tmp_path / "d.json"
    det.write_text(json.dumps({"validation": [{"image": "/x/u1.mhd",
                                               "box": [[1, 2, 3, 10, 8, 9], [50, 50, 50, 4, 4, 4]],
                                               "label_scores": [0.9, 0.3]}]}))
    dets = load_detector_json(det)
    mal = pd.DataFrame([{"seriesuid": "u1", "coordX": 1.5, "coordY": 2, "coordZ": 3, "probability": 0.9,
                         "malignancy": 0.66}])
    seg = {"mock": True, "lesions": [{"center_mm": [1, 2, 3], "diam_mm": 8.4, "volume_mm3": 300}]}
    s = build_study("u1", dets["u1"], malignancy=mal, seg=seg)
    assert len(s.nodules) == 1  # 0.3 低於門檻
    n = s.nodules[0]
    assert n.malignancy_prob == 0.66 and n.diameter_source == "segmentation" and n.volume_mm3 == 300
    assert "MOCK" in s.models.segmentation
    assert "MOCK" in to_markdown(build_report(s), "en")


def test_cli_on_examples(tmp_path):
    subprocess.run([sys.executable, "-m", "scripts.make_lungrads_report",
                    "--detections", "examples/lungrads/detections.json",
                    "--malignancy", "examples/lungrads/candidate_scores.csv",
                    "--seg", "examples/lungrads/segmentation_mock.json", "--out", str(tmp_path)], check=True)
    rep = json.loads((tmp_path / "demo-0001.json").read_text())
    assert rep["exam_category"] == "4B"
    assert (tmp_path / "demo-0002.zh.md").read_text().count("Lung-RADS 2") >= 1
