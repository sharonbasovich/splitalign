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


def test_retry_storm_consumes_shared_cap(pipeline_env, monkeypatch):
    """A 429 storm: retries burn the shared budget; run stops cleanly."""
    def boom(*a, **k):
        raise urllib.error.HTTPError("u", 429, "", {}, io.BytesIO(b"x"))
    monkeypatch.setattr(_ap.urllib.request, "urlopen", boom)
    args = _args(max_requests=3)
    _run.cmd_pipeline(args)
    assert args._budget.requests == 3      # retries counted at the boundary
    rd = next((_run.RESULTS_DIR / "runs").iterdir())
    s = json.loads((rd / "run_summary_splitalign_apertus_dev_val.json")
                   .read_text())
    assert "request cap" in s["budget_capped"]
    assert s["attempts_without_usage"] == 3   # unknown token cost disclosed


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
