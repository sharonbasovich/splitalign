"""Attempt-boundary logging regressions — synthetic fixtures only.

Every record must carry a unique run/method/attempt identity and tell the
truth about the outbound boundary: predispatch budget/bound blocks are NOT
outbound attempts (dispatched=False, no usage claim), failed dispatches are
unknown-cost (usage_status="unknown"), and provider usage that arrives with
a bound-violation exception is preserved in full (never dropped/clipped).
"""
import json
import urllib.error

import pytest

from splitalign import apertus as _ap
from splitalign.apertus import (ApertusClient, ApiBudget, ApertusUnavailable,
                                BoundNotConfigured, BudgetExceeded, CallLogger,
                                TokenBoundSpec)
from splitalign.cache import DiskCache
from splitalign.judge import Judge
from test_reservation import _Resp, _ok_resp, TEST_SPEC, MSG

KIND = "judge_tag"


def _judge(tmp_path, client, budget, *, method="splitalign"):
    j = Judge(client=client, backend="apertus",
              cache=DiskCache(tmp_path / "cache"),
              logger=CallLogger(tmp_path / "calls.jsonl", run_id="testrun"),
              split="dev/test", item_id="synthetic_0", seed=0)
    j.budget = budget
    j.method = method
    return j


def _client(budget, spec=TEST_SPEC):
    c = ApertusClient("http://fake", "k", "m", rps=0, bound_spec=spec)
    c.budget = budget
    return c


def _recs(tmp_path):
    return [json.loads(l)
            for l in (tmp_path / "calls.jsonl").read_text().splitlines()]


def _invoke(j, max_tokens=1):
    return j._invoke(kind=KIND, messages=MSG,
                     payload_for_key={"req": 1}, max_tokens=max_tokens)


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(_ap.time, "sleep", lambda *_: None)


def test_cap_block_is_predispatch_not_attempt(no_sleep, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(_ap.urllib.request, "urlopen",
                        lambda *a, **k: calls.append(1) or _ok_resp())
    j = _judge(tmp_path, _client(ApiBudget(max_requests=0)),
               ApiBudget(max_requests=0))
    with pytest.raises(BudgetExceeded):
        _invoke(j)
    assert calls == []
    (r,) = _recs(tmp_path)
    assert r["dispatched"] is False
    assert r["usage_status"] is None
    assert r["ok"] is False and r["error"] == "BudgetExceeded"
    assert r["attempt_id"].startswith("testrun-") and \
        r["attempt_id"].endswith("-1") and r["method"] == "splitalign"


def test_missing_bound_config_is_predispatch_block(no_sleep, monkeypatch,
                                                   tmp_path):
    calls = []
    monkeypatch.setattr(_ap.urllib.request, "urlopen",
                        lambda *a, **k: calls.append(1) or _ok_resp())
    c = ApertusClient("http://fake", "k", "m", rps=0, bound_spec=None)
    j = _judge(tmp_path, c, None)
    with pytest.raises(BoundNotConfigured):
        _invoke(j)
    assert calls == []
    (r,) = _recs(tmp_path)
    assert r["dispatched"] is False and r["usage_status"] is None
    assert r["error"] == "BoundNotConfigured"


def test_bound_violation_preserves_known_usage(no_sleep, monkeypatch,
                                             tmp_path):
    monkeypatch.setattr(_ap.urllib.request, "urlopen",
                        lambda *a, **k: _ok_resp(pt=1000, ct=1))
    budget = ApiBudget(max_requests=100, max_tokens=10**6)
    j = _judge(tmp_path, _client(budget), budget)
    with pytest.raises(ApertusUnavailable) as exc:
        _invoke(j)
    assert exc.value.usage == (1000, 1)
    (r,) = _recs(tmp_path)
    assert r["dispatched"] is True and r["usage_status"] == "known"
    # actual measured over-bound usage preserved in full — never clipped
    assert r["prompt_tokens"] == 1000 and r["completion_tokens"] == 1
    assert r["reserved_tokens"] == 256 + 8 + 1 + 1   # bound for "a" + mt=1
    assert r["error"] == "ApertusUnavailable" and r["ok"] is False
    assert budget.measured_tokens == 1001 and budget.bound_violations == 1


def test_successful_dispatch_logged_with_reservation(no_sleep, monkeypatch,
                                                     tmp_path):
    monkeypatch.setattr(_ap.urllib.request, "urlopen",
                        lambda *a, **k: _ok_resp(pt=10, ct=5))
    budget = ApiBudget(max_requests=100, max_tokens=10**6)
    j = _judge(tmp_path, _client(budget), budget)
    text, cached, res = _invoke(j, max_tokens=1)
    assert not cached and res.prompt_tokens == 10
    (r,) = _recs(tmp_path)
    assert r["dispatched"] is True and r["usage_status"] == "known"
    assert r["reserved_tokens"] == 256 + 8 + 1 + 1
    assert r["prompt_tokens"] == 10 and r["completion_tokens"] == 5
    assert r["ok"] is True and r["error"] is None


def test_unknown_cost_dispatch_marked_unknown(no_sleep, monkeypatch,
                                              tmp_path):
    def boom(*a, **k):
        raise urllib.error.URLError("down")
    monkeypatch.setattr(_ap.urllib.request, "urlopen", boom)
    budget = ApiBudget(max_requests=100, max_tokens=10**6)
    j = _judge(tmp_path, _client(budget), budget)
    with pytest.raises(ApertusUnavailable):
        _invoke(j)
    (r,) = _recs(tmp_path)
    assert r["dispatched"] is True and r["usage_status"] == "unknown"
    assert r["prompt_tokens"] is None and r["completion_tokens"] is None
    assert r["reserved_tokens"] == 256 + 8 + 1 + 1
    assert budget.attempts_no_usage == 1 and budget.poisoned


def test_attempt_ids_unique_across_logger_instances(no_sleep, monkeypatch,
                                                    tmp_path):
    """make_judge builds ONE CallLogger per item appending to the same
    run-scoped calls.jsonl — attempt_id must be unique across instances."""
    monkeypatch.setattr(_ap.urllib.request, "urlopen",
                        lambda *a, **k: _ok_resp(pt=1, ct=1))
    budget = ApiBudget(max_requests=100, max_tokens=10**6)
    shared = tmp_path / "calls.jsonl"
    for iid in ("synthetic_a", "synthetic_b"):
        c = _client(budget)
        j = Judge(client=c, backend="apertus",
                  cache=DiskCache(tmp_path / "cache"),
                  logger=CallLogger(shared, run_id="testrun"),
                  split="dev/test", item_id=iid, seed=0)
        j.budget = budget
        _invoke(j)
    recs = _recs(tmp_path)
    assert len({r["attempt_id"] for r in recs}) == 2
    assert all(r["run_id"] == "testrun" for r in recs)


def test_attempt_ids_unique_across_records(no_sleep, monkeypatch, tmp_path):
    monkeypatch.setattr(_ap.urllib.request, "urlopen",
                        lambda *a, **k: _ok_resp(pt=1, ct=1))
    budget = ApiBudget(max_requests=100, max_tokens=10**6)
    j = _judge(tmp_path, _client(budget), budget)
    _invoke(j)
    _invoke(j)      # identical payload -> cache hit, still a logged record
    j._invoke(kind=KIND, messages=MSG,
              payload_for_key={"req": 2}, max_tokens=1)   # new dispatch
    ids = [r["attempt_id"] for r in _recs(tmp_path)]
    assert len(ids) == len(set(ids)) == 3
