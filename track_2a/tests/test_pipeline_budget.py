"""End-to-end cmd_pipeline budget/artifact-lifecycle tests.

Synthetic endpoint only (monkeypatched urlopen) — no real API calls.
Proves: one shared ApiBudget governs BOTH methods (the second method
cannot reset it), retries consume the same cap, and every artifact of a
capped run is written into one immutable run scope with an intended-ID
manifest — never mixed with earlier runs.
"""
import io
import json
import urllib.error

import pytest

from splitalign import apertus as _ap
from splitalign import run as _run


class _FakeResp:
    def read(self):
        return json.dumps({
            "model": "fake-model", "choices": [{"message": {"content":
                "{\"sentence1\":[],\"sentence2\":[]}"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10},
        }).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture
def pipeline_env(tmp_path, monkeypatch):
    """Redirect all run output to tmp and use a fake apertus endpoint."""
    out = tmp_path / "out"
    monkeypatch.setattr(_run, "OUT_DIR", out)
    monkeypatch.setattr(_run, "RESULTS_DIR", out / "results")
    monkeypatch.setattr(_run, "PRED_DIR", out / "results" / "predictions")
    monkeypatch.setattr(_run, "DETAIL_DIR", out / "results" / "details")
    monkeypatch.setattr(_run, "EVIDENCE_PATH", out / "viewer" / "evidence.js")
    monkeypatch.setattr(_run, "CONFIG_PATH", out / "results" / "calibration.json")
    monkeypatch.setenv("APERTUS_API_KEY", "test-key")
    monkeypatch.setenv("APERTUS_API_BASE", "http://fake.local/v1")
    monkeypatch.setenv("APERTUS_MODEL", "fake-model")
    monkeypatch.setenv("APERTUS_RPS", "0")
    monkeypatch.setattr(_ap.time, "sleep", lambda *_: None)
    return out


def _args(**kw):
    import argparse
    base = dict(split="val", lang="de", limit=1, offset=0, seed=0,
                backend="apertus", bootstrap=0, require_full=False,
                max_requests=4, max_tokens=10**9, cfg=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_second_method_cannot_reset_budget(pipeline_env, monkeypatch):
    """Splitalign exhausts the shared cap; baseline must NOT get a fresh one."""
    monkeypatch.setattr(_ap.urllib.request, "urlopen",
                        lambda *a, **k: _FakeResp())
    args = _args()
    _run.cmd_pipeline(args)

    budget = args._budget
    # one budget object total, shared across both methods
    assert budget.requests <= budget.max_requests
    assert budget.requests > 0
    # baseline never got a fresh allowance: total attempts <= cap, not 2*cap
    assert budget.requests <= 4

    run_dirs = list((_run.RESULTS_DIR / "runs").iterdir())
    assert len(run_dirs) == 1
    rd = run_dirs[0]
    man = json.loads((rd / "manifest.json").read_text())
    # both modes declared intended IDs in ONE manifest
    assert set(man["modes"]) == {"splitalign", "baseline"}
    for mode in ("splitalign", "baseline"):
        s = json.loads((rd / f"run_summary_{mode}_apertus_dev_val.json")
                       .read_text())
        assert s["budget_capped"], f"{mode} must report the cap honestly"
        assert s["new_api_requests"] == budget.requests  # shared counter
    # matched-ID eval file exists in the same immutable scope
    assert (rd / "eval_matched_apertus_dev_val.json").exists()


def test_429_halts_run_no_retry(pipeline_env, monkeypatch):
    """A 429 is one unknown-cost attempt: recorded, NO retry, run halts
    fail-closed — both methods share the single ledger."""
    def boom(*a, **k):
        raise urllib.error.HTTPError("u", 429, "", {}, io.BytesIO(b"x"))
    monkeypatch.setattr(_ap.urllib.request, "urlopen", boom)
    args = _args(max_requests=3)
    _run.cmd_pipeline(args)
    # one attempt per method (each halts at its first dispatch), no retries
    assert args._budget.requests == 2
    assert args._budget.attempts_no_usage == 2
    rd = next((_run.RESULTS_DIR / "runs").iterdir())
    s = json.loads((rd / "run_summary_splitalign_apertus_dev_val.json")
                   .read_text())
    assert s["budget_capped"] and "UNKNOWN" in s["budget_capped"]
    assert s["attempts_without_usage"] == 1   # disclosed, at splitalign stop
    b = json.loads((rd / "run_summary_baseline_apertus_dev_val.json")
                   .read_text())
    assert b["budget_capped"] and b["attempts_without_usage"] == 2


def test_runs_do_not_mix(pipeline_env, monkeypatch):
    """Two runs produce two scopes; a later capped run can't pollute the first."""
    monkeypatch.setattr(_ap.urllib.request, "urlopen",
                        lambda *a, **k: _FakeResp())
    _run.cmd_pipeline(_args(max_requests=10**9, max_tokens=10**9))
    _run.cmd_pipeline(_args(max_requests=1))   # tiny cap -> immediate partial
    rds = sorted((_run.RESULTS_DIR / "runs").iterdir())
    assert len(rds) == 2
    # first run's predictions are untouched by the second, capped run
    preds1 = sorted(p.name for p in rds[0].glob("*_admin_*.jsonl"))
    preds2 = sorted(p.name for p in rds[1].glob("*_admin_*.jsonl"))
    assert preds1, "first run must have produced files"
    assert set(preds2) <= set(preds1) or preds2 != preds1


def _strict(p):
    return json.loads(p.read_text(),
                      parse_constant=lambda c: (_ for _ in ()).throw(
                          AssertionError(f"non-strict JSON {c} in {p}")))


def _assert_truthful_scope(rd, modes=("splitalign", "baseline"),
                           langs=("de", "fr", "it")):
    """Every planned language has completed+remaining == intended, both
    modes have summaries, matched eval exists, all JSON strict."""
    man = _strict(rd / "manifest.json")
    assert set(man["modes"]) == set(modes)
    assert man["source"] is not None and "prompt_version" in man
    assert set(man["gold_sha256"]) == set(langs)
    for mode in modes:
        s = _strict(rd / f"run_summary_{mode}_apertus_dev_val.json")
        intended = man["modes"][mode]["intended_ids"]
        for l in langs:
            assert set(s["completed_ids"][l]) | set(s["remaining_ids"][l]) \
                == set(intended[l]), (mode, l)
            assert not (set(s["completed_ids"][l]) & set(s["remaining_ids"][l]))
            assert s["lang_status"][l] in (
                "complete", "partial", "started_no_output", "not_started")
            assert (s["lang_status"][l] == "complete") == (
                len(s["completed_ids"][l]) == len(intended[l]))
        assert s["n_completed"] == sum(len(v) for v in s["completed_ids"].values())
        _strict(rd / f"eval_{mode}_apertus_dev_val.json")
    m = _strict(rd / "eval_matched_apertus_dev_val.json")
    assert set(m["per_language"]) == set(langs)
    for p in rd.glob("*.json"):
        _strict(p)
    return man, m


def test_lang_all_cap_before_first_item(pipeline_env, monkeypatch):
    """Cap already exhausted (0 requests allowed): nothing runs, yet every
    language of BOTH methods has truthful not_started state and the
    matched eval + evidence are still written."""
    monkeypatch.setattr(_ap.urllib.request, "urlopen",
                        lambda *a, **k: _FakeResp())
    args = _args(lang="all", max_requests=0)
    _run.cmd_pipeline(args)
    rd = next((_run.RESULTS_DIR / "runs").iterdir())
    man, m = _assert_truthful_scope(rd)
    for mode in ("splitalign", "baseline"):
        s = _strict(rd / f"run_summary_{mode}_apertus_dev_val.json")
        assert s["n_completed"] == 0 and s["budget_capped"]
        assert s["lang_status"] == {"de": "started_no_output",
                                    "fr": "not_started", "it": "not_started"}
        e = _strict(rd / f"eval_{mode}_apertus_dev_val.json")
        assert e["macro_spearman"] is None and e["macro_spearman_invalid_reason"]
    assert m["macro_matched"] == {"splitalign": None, "baseline": None}
    assert args._budget.requests == 0
    ev = _run.EVIDENCE_PATH.read_text()
    assert "NaN" not in ev
    evj = json.loads(ev[ev.index("{"):].rstrip().rstrip(";"))
    assert evj["run"]["run_id"] == rd.name and evj["run"]["partial"] is True
    assert evj["run"]["modes"]["baseline"]["budget_capped"]


def test_lang_all_cap_mid_first_language(pipeline_env, monkeypatch):
    """Token cap trips inside de: de partial, fr/it not_started for
    splitalign; baseline exhausted before its first item; one shared budget."""
    monkeypatch.setattr(_ap.urllib.request, "urlopen",
                        lambda *a, **k: _FakeResp())
    probe = _args(lang="de", limit=1, max_requests=10**9)
    _run._predict(probe, "splitalign")
    per_item = probe._budget.requests
    assert per_item > 1
    # fresh scope: allow a bit more than one de item, less than two
    for d in (_run.RESULTS_DIR / "runs").iterdir():
        for f in d.iterdir():
            f.unlink()
        d.rmdir()
    import shutil
    shutil.rmtree(_run.RESULTS_DIR / "cache", ignore_errors=True)
    args = _args(lang="all", limit=2, max_requests=per_item + 1)
    _run.cmd_pipeline(args)
    rd = next((_run.RESULTS_DIR / "runs").iterdir())
    _assert_truthful_scope(rd)
    sa = _strict(rd / "run_summary_splitalign_apertus_dev_val.json")
    assert sa["lang_status"]["de"] == "partial"
    assert sa["completed_ids"]["de"] and len(sa["remaining_ids"]["de"]) == 1
    assert sa["lang_status"]["fr"] == "not_started"
    assert sa["lang_status"]["it"] == "not_started"
    assert sum(1 for _ in (rd / "splitalign_apertus_admin_de.jsonl").open()) == 1
    assert not (rd / "splitalign_apertus_admin_fr.jsonl").exists()
    bl = _strict(rd / "run_summary_baseline_apertus_dev_val.json")
    assert bl["budget_capped"] and bl["n_completed"] == 0
    assert bl["lang_status"] == {"de": "started_no_output",
                                 "fr": "not_started", "it": "not_started"}
    assert sa["new_api_requests"] == bl["new_api_requests"] == per_item + 1
    assert args._budget.requests == per_item + 1  # retries would count too


def test_lang_all_cap_mid_second_language(pipeline_env, monkeypatch):
    """Cap trips in fr: de complete, fr started/partial, it not_started."""
    monkeypatch.setattr(_ap.urllib.request, "urlopen",
                        lambda *a, **k: _FakeResp())
    probe = _args(lang="de", limit=1, max_requests=10**9)
    _run._predict(probe, "splitalign")
    per_de = probe._budget.requests
    for d in (_run.RESULTS_DIR / "runs").iterdir():
        for f in d.iterdir():
            f.unlink()
        d.rmdir()
    import shutil
    shutil.rmtree(_run.RESULTS_DIR / "cache", ignore_errors=True)
    args = _args(lang="all", limit=1, max_requests=per_de + 1)
    _run.cmd_pipeline(args)
    rd = next((_run.RESULTS_DIR / "runs").iterdir())
    _assert_truthful_scope(rd)
    sa = _strict(rd / "run_summary_splitalign_apertus_dev_val.json")
    assert sa["lang_status"]["de"] == "complete"
    assert sa["lang_status"]["fr"] in ("partial", "started_no_output")
    assert sa["lang_status"]["it"] == "not_started"
    assert (rd / "splitalign_apertus_admin_de.jsonl").exists()
    e = _strict(rd / "eval_splitalign_apertus_dev_val.json")
    assert e["macro_spearman"] is None          # strict: fr/it undefined
    assert e["per_language"]["de"]["spearman"] is not None or \
        e["per_language"]["de"]["invalid_reason"]
    bl = _strict(rd / "run_summary_baseline_apertus_dev_val.json")
    assert bl["budget_capped"] and bl["n_completed"] == 0
    m = _strict(rd / "eval_matched_apertus_dev_val.json")
    assert all(v["n_matched"] == 0 for v in m["per_language"].values())


def test_lang_all_500_halts_run_fail_closed(pipeline_env, monkeypatch):
    """A 500 is an unknown-cost attempt: run halts fail-closed after ONE
    dispatch per method (no retry), zero outputs, attempt disclosed."""
    def boom(*a, **k):
        raise urllib.error.HTTPError("u", 500, "", {}, io.BytesIO(b"x"))
    monkeypatch.setattr(_ap.urllib.request, "urlopen", boom)
    args = _args(lang="all", limit=1, max_requests=10**9)
    _run.cmd_pipeline(args)
    rd = next((_run.RESULTS_DIR / "runs").iterdir())
    _assert_truthful_scope(rd)
    sa = _strict(rd / "run_summary_splitalign_apertus_dev_val.json")
    assert sa["n_completed"] == 0 and sa["budget_capped"]
    assert sa["lang_status"] == {"de": "started_no_output",
                               "fr": "not_started", "it": "not_started"}
    assert sa["attempts_without_usage"] == 1
    bl = _strict(rd / "run_summary_baseline_apertus_dev_val.json")
    assert bl["n_completed"] == 0 and bl["budget_capped"]
    assert bl["lang_status"]["de"] == "started_no_output"
    # exactly one dispatched attempt per method, both unknown cost
    assert args._budget.requests == 2 and args._budget.attempts_no_usage == 2


def test_export_viewer_requires_explicit_run(pipeline_env, monkeypatch):
    """No implicit newest-run promotion; explicit run embeds identity."""
    monkeypatch.setattr(_ap.urllib.request, "urlopen",
                        lambda *a, **k: _FakeResp())
    _run.cmd_pipeline(_args(max_requests=10**9))
    rd = next((_run.RESULTS_DIR / "runs").iterdir())
    import argparse
    with pytest.raises(SystemExit, match="requires --run"):
        _run.cmd_export_viewer(argparse.Namespace(
            split="val", backend="apertus", run=None, mode="splitalign"))
    _run.cmd_export_viewer(argparse.Namespace(
        split="val", backend="apertus", run=rd.name, mode="splitalign"))
    ev = _run.EVIDENCE_PATH.read_text()
    evj = json.loads(ev[ev.index("{"):].rstrip().rstrip(";"))
    assert evj["run"]["run_id"] == rd.name
    assert evj["run"]["partial"] is False
    assert evj["run"]["modes_intended_not_produced"] == []
    assert evj["run"]["matched"]["n_matched"] == {"de": 1}
    assert evj["run"]["source"] is not None
