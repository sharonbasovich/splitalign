"""Language propagation, run-scoped call logs, calibrate fail-closed,
exporter-vs-inference provenance. Mock backend / synthetic only."""
import argparse
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from splitalign import PROMPT_VERSION, PROMPT_VERSION_BY_KIND
from splitalign import apertus as _ap
from splitalign import run as _run
from splitalign.cache import DiskCache
from splitalign.pipeline import make_judge
from splitalign.viewer_export import inference_provenance

from viewer_fixture import build as build_fixture  # noqa: E402

LANG_NAME = {"de": "German", "fr": "French", "it": "Italian"}


@pytest.fixture
def mock_env(tmp_path, monkeypatch):
    out = tmp_path / "out"
    monkeypatch.setattr(_run, "OUT_DIR", out)
    monkeypatch.setattr(_run, "RESULTS_DIR", out / "results")
    monkeypatch.setattr(_run, "PRED_DIR", out / "results" / "predictions")
    monkeypatch.setattr(_run, "DETAIL_DIR", out / "results" / "details")
    monkeypatch.setattr(_run, "EVIDENCE_PATH", out / "viewer" / "evidence.js")
    monkeypatch.setattr(_run, "CONFIG_PATH", out / "results" / "calibration.json")
    monkeypatch.delenv("SPLITALIGN_BACKEND", raising=False)
    return out


def _args(**kw):
    base = dict(split="val", lang="all", limit=1, offset=0, seed=0,
                backend="mock", bootstrap=0, require_full=False,
                max_requests=500, max_tokens=10**9, cfg=None)
    base.update(kw)
    return argparse.Namespace(**base)


def _spy_mock(monkeypatch):
    calls = []
    orig = _ap.MockApertusClient.complete

    def spy(self, messages, **kw):
        calls.append([dict(m) for m in messages])
        return orig(self, messages, **kw)
    monkeypatch.setattr(_ap.MockApertusClient, "complete", spy)
    return calls


def test_lang_reaches_every_judge_prompt_via_cli_pipeline(mock_env, monkeypatch):
    """Full CLI/pipeline level: de, fr and it items each get a judge prompt
    naming THEIR language, and every judgment/detail records it."""
    calls = _spy_mock(monkeypatch)
    rc = _run.main(["pipeline", "--lang", "all", "--limit", "1",
                    "--backend", "mock", "--bootstrap", "0"])
    assert rc == 0
    run_dir = next((mock_env / "results" / "runs").iterdir())
    details = json.loads((run_dir / "details_splitalign_mock_dev_val.json").read_text())
    seen = set()
    for det in details:
        assert det["judge_lang"] == det["lang"]
        for j in det["judgments"]:
            if j is not None:
                assert j["lang_b"] == det["lang"]
        seen.add(det["lang"])
    assert seen == {"de", "fr", "it"}
    judge_calls = [c for c in calls if '"kind": "judge_tag"' in c[-1]["content"]]
    assert judge_calls
    by_lang = {l: 0 for l in LANG_NAME}
    for msgs in judge_calls:
        lb = next(l for l in LANG_NAME if f'"lang_b": "{l}"' in msgs[-1]["content"])
        assert LANG_NAME[lb] in msgs[0]["content"], "system prompt must name the target language"
        assert LANG_NAME[lb] in msgs[-1]["content"]
        for other, name in LANG_NAME.items():
            if other != lb:
                assert name not in msgs[0]["content"]
        by_lang[lb] += 1
    assert all(n > 0 for n in by_lang.values()), by_lang


def test_judge_cache_identity_includes_language_and_v2_key(tmp_path):
    """Cache identity: language is part of the judge key, the judge kind is
    keyed under the bumped version, and an entry stored under the old v1 key
    is never served to the corrected code."""
    assert PROMPT_VERSION_BY_KIND["judge_pair"] == PROMPT_VERSION == "splitalign-prompts-v2-lang"
    assert PROMPT_VERSION_BY_KIND["pair_similarity"] == "splitalign-prompts-v1"
    j, _ = make_judge("mock", tmp_path, "dev/val", "x", 0, run_id="t")
    a, b = "The council approved the budget.", "Le conseil a approuvé le budget."
    j_de = j.judge_pair(a, b, lang_a="en", lang_b="de")
    j_fr = j.judge_pair(a, b, lang_a="en", lang_b="fr")
    assert j_de.lang_b == "de" and j_fr.lang_b == "fr"
    assert j_fr.cached is False, "fr must not reuse the de judgment"
    assert j.judge_pair(a, b, lang_a="en", lang_b="fr").cached is True
    # an old v1-keyed entry (what historical runs wrote) is invisible now
    from splitalign import prompts
    req = prompts.judge_request(a, b, "en", "it")
    cache = DiskCache(tmp_path / "cache")
    old_key = cache.key(backend="mock", model="MOCK-NOT-APERTUS",
                        prompt_version="splitalign-prompts-v1", kind="judge_pair",
                        split="dev/val", item_id="x", payload={"req": req, "r": 0})
    cache.put(old_key, {"text": json.dumps({"semantic_difference": 5,
                                            "differing_spans": {"side_a": [], "side_b": []}}),
                        "backend": "mock", "model": "MOCK-NOT-APERTUS"})
    j_it = j.judge_pair(a, b, lang_a="en", lang_b="it")
    assert j_it.cached is False and j_it.difference != 5


def test_predict_item_requires_lang(tmp_path):
    from splitalign.pipeline import predict_item
    j, _ = make_judge("mock", tmp_path, "dev/val", "x", 0)
    with pytest.raises(TypeError):
        predict_item({"id": "x", "text_a": "a b", "text_b": "c d",
                      "labels_a": [0, 0], "labels_b": [0, 0]}, j)  # no lang


def test_call_log_is_run_scoped_including_cache_hits(mock_env):
    """Two runs: each has its own calls.jsonl, every record (cached ones
    included) carries that run's id, and no new unscoped record is written."""
    runs_root = mock_env / "results" / "runs"
    assert _run.cmd_pipeline(_args(lang="de")) == 0
    first = next(runs_root.iterdir())
    assert _run.cmd_pipeline(_args(lang="de")) == 0
    second_dir = next(d for d in runs_root.iterdir() if d != first)
    runs = [first, second_dir]
    assert not (mock_env / "results" / "calls.jsonl").exists()
    for rd in runs:
        recs = [json.loads(l) for l in (rd / "calls.jsonl").read_text().splitlines()]
        assert recs and all(r["run_id"] == rd.name for r in recs)
        assert all(r["backend"] == "mock" for r in recs)
    second = [json.loads(l) for l in (runs[1] / "calls.jsonl").read_text().splitlines()]
    assert any(r["cached"] for r in second), "second run should hit the shared cache"
    assert all(r["run_id"] == runs[1].name for r in second if r["cached"])
    info = _run._run_info(runs[0], "dev/val", "mock", "dev_val",
                          {"splitalign": [], "baseline": []})
    assert info["call_log"].startswith("run-scoped")
    assert info["known_defects"] == []


def test_calibrate_apertus_fails_closed_before_any_client(mock_env, monkeypatch):
    """calibrate --backend apertus --max-requests 0: exits before creating a
    client or judge, so zero outbound attempts are even possible."""
    def boom(*a, **k):
        raise AssertionError("network/client path must not be reached")
    monkeypatch.setattr(_ap.ApertusClient, "from_env", boom)
    monkeypatch.setattr(_ap.urllib.request, "urlopen", boom)
    monkeypatch.setattr(_run, "make_judge", boom)
    monkeypatch.setenv("APERTUS_API_KEY", "test-key")
    monkeypatch.setenv("APERTUS_API_BASE", "http://fake.local/v1")
    monkeypatch.setenv("APERTUS_MODEL", "fake-model")
    with pytest.raises(SystemExit) as ei:
        _run.cmd_calibrate(_args(backend="apertus", max_requests=0, lang="de"))
    assert "mock only" in str(ei.value)
    assert not (mock_env / "results" / "calibration.json").exists()


def test_calibrate_mock_passes_lang_and_tags_log(mock_env, monkeypatch):
    calls = _spy_mock(monkeypatch)
    assert _run.cmd_calibrate(_args(backend="mock", lang="fr", limit=1)) == 0
    judge_calls = [c for c in calls if '"kind": "judge_tag"' in c[-1]["content"]]
    assert judge_calls and all("French" in c[0]["content"] for c in judge_calls)
    recs = [json.loads(l) for l in
            (mock_env / "results" / "calls.jsonl").read_text().splitlines()]
    assert recs and all(str(r["run_id"]).startswith("calibrate-") for r in recs)


def test_known_defects_for_historical_prompt_versions():
    assert _run._known_defects("splitalign-prompts-v1")
    assert _run._known_defects(None)
    assert _run._known_defects(PROMPT_VERSION) == []


def test_inference_provenance_is_not_exporter_version():
    items = {"splitalign": [
        {"id": "a", "provenance": {"prompt_version": "splitalign-prompts-v1",
                                   "model": "swiss-ai/Apertus-v1.5-8B",
                                   "splitalign_version": "0.1.0", "backend": "apertus"}},
        {"id": "b", "failed": True, "provenance": {"prompt_version": "splitalign-prompts-v1",
                                                   "backend": "apertus"}}]}
    inf = inference_provenance(items, {"run_id": "r", "prompt_version": "splitalign-prompts-v1"})
    assert inf["prompt_version"] == "splitalign-prompts-v1"
    assert inf["model"] == "swiss-ai/Apertus-v1.5-8B"
    assert inf["conflict"] == {}
    assert inference_provenance({"m": [{"id": "x"}]})["prompt_version"] is None
    mixed = inference_provenance({"m": [{"provenance": {"prompt_version": "v1"}},
                                        {"provenance": {"prompt_version": "v2"}}]})
    assert mixed["prompt_version"] is None and mixed["conflict"] == {"prompt_version": ["v1", "v2"]}


def test_inference_provenance_manifest_vs_items_disagreement_is_a_conflict():
    # Regression: a manifest value that disagrees with the per-item records
    # must surface in ``conflict`` (not just null the headline field).
    items = {"splitalign": [{"id": "a", "provenance": {
        "prompt_version": "splitalign-prompts-v1", "model": "m"}}]}
    inf = inference_provenance(
        items, {"run_id": "r", "prompt_version": "splitalign-prompts-v2-lang",
                "model": "m"})
    assert inf["prompt_version"] is None
    assert inf["conflict"] == {"prompt_version": [
        "splitalign-prompts-v1", "splitalign-prompts-v2-lang"]}
    assert inf["model"] == "m"
    # manifest-only value (items unrecorded) is adopted, no conflict
    only_man = inference_provenance({"m": [{"id": "x"}]},
                                    {"prompt_version": "splitalign-prompts-v1"})
    assert only_man["prompt_version"] == "splitalign-prompts-v1"
    assert only_man["conflict"] == {}


def test_export_viewer_keeps_exporter_and_inference_separate(mock_env):
    """A re-export of an existing run must not relabel its prompt version."""
    assert _run.cmd_pipeline(_args(lang="de")) == 0
    rd = next((mock_env / "results" / "runs").iterdir())
    man_p = rd / "manifest.json"
    man = json.loads(man_p.read_text())
    man["prompt_version"] = "splitalign-prompts-v1"       # simulate historical run
    man_p.write_text(json.dumps(man))
    for det in rd.glob("details_*_mock_dev_val.json"):
        d = json.loads(det.read_text())
        for it in d:
            it["provenance"]["prompt_version"] = "splitalign-prompts-v1"
        det.write_text(json.dumps(d))
    ns = argparse.Namespace(split="val", backend="mock", run=str(rd), mode="splitalign")
    assert _run.cmd_export_viewer(ns) == 0
    ev = json.loads((mock_env / "viewer" / "evidence.js").read_text()
                    .removeprefix("window.EVIDENCE = ").rstrip().rstrip(";"))
    assert ev["exporter"]["prompt_version"] == PROMPT_VERSION
    assert ev["prompt_version"] == "splitalign-prompts-v1"
    assert ev["inference"]["prompt_version"] == "splitalign-prompts-v1"
    assert ev["run"]["known_defects"], "v1 run must carry the language-defect notice"
    assert any("judge prompt language" in l for l in ev["limitations"])


def test_viewer_fixture_is_strict_json(tmp_path):
    d = build_fixture(tmp_path / "v")
    txt = (d / "evidence.js").read_text()
    def bad(c):
        raise AssertionError(c)
    ev = json.loads(txt.removeprefix("window.EVIDENCE = ").rstrip().rstrip(";"),
                    parse_constant=bad)
    assert ev["items_by_mode"]["baseline"] == []
    assert any(it.get("failed") for it in ev["items_by_mode"]["splitalign"])
    assert (d / "app.js").exists() and (d / "index.html").exists()
