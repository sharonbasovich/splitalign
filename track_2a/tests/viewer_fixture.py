"""Synthetic evidence fixture for browser QA of the static viewer.

Builds a self-contained copy of ``viewer/`` whose evidence.js contains the
awkward cases the real exporter can produce: a normal item, a FAILED item
(id/lang/failed/error/provenance only — no text), an item without recorded
text, an EMPTY mode, a partial run with a cap, a known configuration defect
and separate inference-vs-exporter provenance. Mock labels only; nothing
here is model output.

    PYTHONPATH=src python -m tests.viewer_fixture /tmp/viewer_qa
    python3 -m http.server -d /tmp/viewer_qa 8765
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

TRACK = Path(__file__).resolve().parents[1]


def evidence() -> dict:
    prov = {"splitalign_version": "0.1.0", "prompt_version": "splitalign-prompts-v1",
            "backend": "apertus", "model": "swiss-ai/Apertus-v1.5-8B",
            "mode": "splitalign", "split": "dev/val",
            "note": "FIXTURE — synthetic labels, not model output"}
    ok_item = {
        "id": "fixture_de_0", "lang": "de",
        "text_a": "The council approved the budget on Monday .",
        "text_b": "Der Rat hat das Budget am Dienstag genehmigt .",
        "labels_a": [0, 0, 0, 0, 0, 0, 1, -1],
        "labels_b": [0, 0, 0, 0, 0, 0, 1, 0, -1],
        "gold_labels_a": [0, 0, 0, 0, 0, 0, 1, -1],
        "gold_labels_b": [0, 0, 0, 0, 0, 0, 1, 0, -1],
        "segments_a": [{"index": 0, "start": 0, "end": 8,
                        "text": "The council approved the budget on Monday ."}],
        "segments_b": [{"index": 0, "start": 0, "end": 9,
                        "text": "Der Rat hat das Budget am Dienstag genehmigt ."}],
        "ops": [{"op": "1:1", "a_start": 0, "a_end": 1, "b_start": 0, "b_end": 1}],
        "judgments": [{"difference": 2, "spans_a": ["Monday"], "spans_b": ["Dienstag"],
                       "ok": True, "repairs": 0, "cached": False, "lang_b": "de"}],
        "judge_lang": "de",
        "stats": {"parse_failures": 0, "repairs": 0, "span_matched": 2,
                  "span_unmatched": 0, "ops": 1, "unparseable_similarity": 0},
        "provenance": prov,
    }
    failed_item = {"id": "fixture_fr_1", "lang": "fr", "failed": True,
                   "error": "ApertusUnavailable: HTTP 429 after 3 attempts",
                   "provenance": dict(prov, note="failed item: no prediction")}
    notext_item = {"id": "fixture_it_2", "lang": "it", "labels_a": [], "labels_b": [],
                   "ops": [], "judgments": [], "provenance": prov}
    run = {
        "run_id": "FIXTURE-00000000", "started_utc": "2026-01-01T00:00:00Z",
        "source": {"commit": None, "dirty": None,
                   "note": "fixture: no source recorded"},
        "prompt_version": "splitalign-prompts-v1",
        "known_defects": ["judge prompt language: fixture copy of the v1 defect "
                          "notice (fr/it judged with a German-language prompt)"],
        "call_log": "no run-scoped log: fixture",
        "n_split": {"de": 34, "fr": 34, "it": 34},
        "modes": {
            "splitalign": {"intended": {"de": 1, "fr": 1, "it": 1}, "n_intended": 3,
                           "produced": True, "summary_present": True,
                           "completed": {"de": 1, "fr": 0, "it": 1}, "n_completed": 2,
                           "lang_status": {"de": "complete", "fr": "started_no_output",
                                           "it": "complete"},
                           "budget_capped": "max_requests=4 reached", "caution": None,
                           "new_api_requests": 4, "new_api_tokens": 1234,
                           "coverage_of_split": {"de": 0.0294, "fr": 0.0, "it": 0.0294}},
            "baseline": {"intended": {"de": 1, "fr": 1, "it": 1}, "n_intended": 3,
                         "produced": False, "summary_present": False,
                         "caution": "intended in manifest but no run summary",
                         "coverage_of_split": {"de": 0.0, "fr": 0.0, "it": 0.0}},
        },
        "modes_produced": ["splitalign"],
        "modes_intended_not_produced": ["baseline"],
        "matched": None, "partial": True,
        "comparison_note": "fixture: no comparison",
    }
    return {
        "generated_utc": "2026-01-01T00:00:01Z",
        "exporter": {"splitalign_version": "0.1.0",
                     "prompt_version": "splitalign-prompts-v2-lang",
                     "note": "version of the code that WROTE this file"},
        "inference": {"prompt_version": "splitalign-prompts-v1",
                      "model": "swiss-ai/Apertus-v1.5-8B",
                      "splitalign_version": "0.1.0", "backend": "apertus",
                      "conflict": {}, "recorded_from": "items"},
        "splitalign_version": "0.1.0", "prompt_version": "splitalign-prompts-v1",
        "backend": "apertus", "model": "swiss-ai/Apertus-v1.5-8B", "mock": False,
        "split": "dev/val", "run": run,
        "evaluation": {"splitalign": {
            "per_language": {"de": {"spearman": 0.25, "n_samples": 1},
                             "fr": {"spearman": None, "invalid_reason": "no predictions",
                                    "n_samples": 0},
                             "it": {"spearman": None, "invalid_reason": "constant labels",
                                    "n_samples": 1}},
            "macro_spearman": None,
            "macro_spearman_invalid_reason": "fr, it undefined"}},
        "limitations": ["FIXTURE: synthetic labels for viewer QA, not model output."],
        "items_by_mode": {"splitalign": [ok_item, failed_item, notext_item],
                          "baseline": []},
    }


def build(dest: Path) -> Path:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("index.html", "style.css", "app.js"):
        shutil.copy2(TRACK / "viewer" / name, dest / name)
    (dest / "evidence.js").write_text(
        "window.EVIDENCE = " + json.dumps(evidence(), ensure_ascii=False,
                                          allow_nan=False) + ";\n")
    return dest


if __name__ == "__main__":
    print(build(Path(sys.argv[1])))
