"""Atomic predispatch reservation + unknown-cost fail-closed tests.

Synthetic endpoint only (monkeypatched urlopen) — no real API calls.
Proves: reservation happens before dispatch; unknown-cost attempts
(missing/malformed usage, HTTP error, timeout, malformed response) are
recorded once and halt with NO retry; reservations can't exceed caps even
concurrently; and the cost-ledger audit classifies without double counting.
"""
import io
import json
import threading
import urllib.error

import pytest

from splitalign.apertus import (ApertusClient, ApiBudget, ApertusUnavailable,
                                BudgetExceeded)
from splitalign import apertus as _ap


class _Resp:
    """Fake HTTP response with a caller-supplied body."""
    def __init__(self, body):
        if isinstance(body, str):
            body = body.encode()
        self._b = body

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _ok_resp(pt=10, ct=5):
    return _Resp(json.dumps({
        "model": "m", "choices": [{"message": {"content": "ok"}}],
        "usage": {"prompt_tokens": pt, "completion_tokens": ct}}))


def _client(budget):
    c = ApertusClient("http://fake", "k", "m", rps=0)
    c.budget = budget
    return c


MSG = [{"role": "user", "content": "a"}]
BOUND = ApertusClient._prompt_token_bound(MSG)   # 1 byte + 256 = 257


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(_ap.time, "sleep", lambda *_: None)


# -- reservation boundaries -------------------------------------------------

def test_one_token_left_vs_bound_blocks_dispatch(no_sleep, monkeypatch):
    """Remaining capacity 1 token < reservation bound => ZERO dispatch."""
    calls = []
    monkeypatch.setattr(_ap.urllib.request, "urlopen",
                        lambda *a, **k: calls.append(1) or _ok_resp())
    b = ApiBudget(max_requests=10**9, max_tokens=BOUND - 1)
    with pytest.raises(BudgetExceeded):
        _client(b).complete(MSG, max_tokens=0)
    assert calls == [] and b.requests == 0


def test_exact_fit_dispatches_then_next_blocked(no_sleep, monkeypatch):
    """Bound == remaining capacity: dispatch + reconcile; next is blocked."""
    calls = []
    monkeypatch.setattr(_ap.urllib.request, "urlopen",
                        lambda *a, **k: calls.append(1) or _ok_resp(10, 5))
    b = ApiBudget(max_requests=10**9, max_tokens=BOUND + 15)
    c = _client(b)
    c.complete(MSG, max_tokens=15)          # bound 257+15 fits exactly
    assert len(calls) == 1 and b.requests == 1
    assert b.measured_tokens == 15 and b.tokens == 15
    c.complete(MSG, max_tokens=0)           # 15 + 257 = cap exactly: fits
    assert len(calls) == 2 and b.tokens == 30
    with pytest.raises(BudgetExceeded):
        c.complete(MSG, max_tokens=0)       # 30 + 257 > 272
    assert len(calls) == 2


def test_request_cap_zero_dispatches_nothing(no_sleep, monkeypatch):
    calls = []
    monkeypatch.setattr(_ap.urllib.request, "urlopen",
                        lambda *a, **k: calls.append(1) or _ok_resp())
    b = ApiBudget(max_requests=0, max_tokens=10**9)
    with pytest.raises(BudgetExceeded):
        _client(b).complete(MSG)
    assert calls == [] and b.requests == 0


def test_request_cap_one_allows_max_one_including_repair(no_sleep, monkeypatch):
    """Cap 1: exactly one dispatch ever — a follow-up (e.g. a repair) is
    blocked at reservation before leaving the client."""
    calls = []
    monkeypatch.setattr(_ap.urllib.request, "urlopen",
                        lambda *a, **k: calls.append(1) or _ok_resp())
    b = ApiBudget(max_requests=1, max_tokens=10**9)
    c = _client(b)
    c.complete(MSG)
    with pytest.raises(BudgetExceeded):
        c.complete(MSG)                      # repair or second call alike
    assert len(calls) == 1 and b.requests == 1


# -- unknown-cost usage validation -------------------------------------------

def _usage_payload(**over):
    u = {"prompt_tokens": 10, "completion_tokens": 5}
    u.update(over)
    return {"model": "m", "choices": [{"message": {"content": "ok"}}],
            "usage": u}


@pytest.mark.parametrize("payload", [
    {"model": "m", "choices": [{"message": {"content": "ok"}}]},   # no usage
    {**_usage_payload(), "usage": None},                          # null usage
    _usage_payload(prompt_tokens=None),                           # one field null
    _usage_payload(completion_tokens=None),                       # other missing
    {k: v for k, v in _usage_payload().items() if k != "usage"},  # no usage key
    _usage_payload(prompt_tokens=-1),                             # negative
    _usage_payload(prompt_tokens=True),                           # bool
    _usage_payload(completion_tokens=False),                      # bool
    _usage_payload(prompt_tokens="10"),                           # string
    _usage_payload(completion_tokens=2.5),                        # float
    _usage_payload(prompt_tokens=None, completion_tokens=None),
])
def test_bad_usage_one_attempt_unknown_halt(no_sleep, monkeypatch, payload):
    """Missing/malformed usage => 1 recorded attempt, unknown cost, halt —
    and no second dispatch is possible or attempted."""
    calls = []
    monkeypatch.setattr(_ap.urllib.request, "urlopen",
                        lambda *a, **k: calls.append(1) or _Resp(
                            json.dumps(payload)))
    b = ApiBudget(max_requests=10**9, max_tokens=10**9)
    c = _client(b)
    with pytest.raises(ApertusUnavailable, match="usage"):
        c.complete(MSG, max_tokens=0)
    assert len(calls) == 1            # exactly one dispatch, no retry
    assert b.requests == 1 and b.attempts_no_usage == 1
    assert b.measured_tokens == 0
    assert b.tokens == BOUND          # reservation retained, not released


@pytest.mark.parametrize("body", [
    "this is not json",
    json.dumps({"choices": []}),                              # missing choice
    json.dumps({"choices": [{"message": {}}]}),               # missing content
    json.dumps({"choices": [{"message": {"content": 42}}]}),  # non-str content
    json.dumps([]),                                           # not an object
])
def test_malformed_response_attempt_retained_no_second(no_sleep, monkeypatch,
                                                       body):
    """Unparseable body / missing choices / missing content => attempt is
    recorded with unknown cost and the call raises; no second attempt."""
    calls = []
    monkeypatch.setattr(_ap.urllib.request, "urlopen",
                        lambda *a, **k: calls.append(1) or _Resp(body))
    b = ApiBudget(max_requests=10**9, max_tokens=10**9)
    with pytest.raises(ApertusUnavailable):
        _client(b).complete(MSG, max_tokens=0)
    assert len(calls) == 1 and b.requests == 1
    assert b.attempts_no_usage == 1 and b.measured_tokens == 0


def test_concurrent_reservations_cannot_exceed_cap():
    """Concurrent reserve() calls are atomic: total reservations <= cap."""
    b = ApiBudget(max_requests=5, max_tokens=10**9)
    wins = []
    def try_reserve():
        try:
            b.reserve(1)
            wins.append(1)
        except BudgetExceeded:
            pass
    ts = [threading.Thread(target=try_reserve) for _ in range(20)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(wins) == 5 and b.requests == 5


# -- ledger audit classification ---------------------------------------------

def test_ledger_audit_classifies_without_double_count(tmp_path):
    """audit_costs buckets real/mock/error/duplicate-cache rows; distinct
    dedupes exact repeats; estimated usage never joins measured totals."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "audit_costs", str(_repo_results() / "audit_costs.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    log = tmp_path / "calls.jsonl"
    apertus_ok = {"ts": "t1", "backend": "apertus", "cached": False, "ok": True,
                  "prompt_tokens": 100, "completion_tokens": 5}
    apertus_fail = {"ts": "t2", "backend": "apertus", "cached": False,
                    "ok": False}
    unknown_fail = {"ts": "t3", "cached": False, "ok": False}
    mock_rec = {"ts": "t4", "backend": "mock", "cached": False, "ok": True,
                "prompt_tokens": 999, "usage_estimated": True}
    cache_hit = {"ts": "t5", "backend": "apertus", "cached": True, "ok": True}
    log.write_text("\n".join(json.dumps(r) for r in
                             [apertus_ok, apertus_fail, unknown_fail, mock_rec,
                              cache_hit, cache_hit]))   # cache_hit duplicated
    rep = mod.audit_file(str(log))
    bk = rep["buckets"]
    assert bk["apertus_noncached_ok"]["prompt_tokens_distinct"] == 100
    assert bk["apertus_noncached_fail"]["distinct"] == 1
    assert bk["unknown_noncached"]["distinct"] == 1
    assert bk["mock_noncached"]["prompt_tokens_distinct"] == 999
    assert bk["cache_apertus"]["raw"] == 2 and bk["cache_apertus"]["distinct"] == 1
    assert bk["cache_apertus"]["dup_extra_rows"] == 1
    assert rep["n_raw"] == 6 and rep["n_distinct"] == 5


def _repo_results():
    from pathlib import Path
    return Path(__file__).resolve().parents[1] / "results"


def test_ledger_audit_matches_committed_numbers():
    """The committed audit script must reproduce the ledger's headline
    numbers exactly (regression guard against silent drift)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "audit_costs", str(_repo_results() / "audit_costs.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rep = mod.audit_file(str(_repo_results() / "calls.jsonl"))
    bk = rep["buckets"]
    assert rep["n_raw"] == 3788 and rep["n_distinct"] == 3725
    assert bk["apertus_noncached_ok"]["distinct"] == 1243
    assert bk["apertus_noncached_ok"]["prompt_tokens_distinct"] == 1295021
    assert bk["apertus_noncached_ok"]["completion_tokens_distinct"] == 361332
    assert bk["apertus_noncached_fail"]["distinct"] == 1
    assert bk["unknown_noncached"]["distinct"] == 2
    assert bk["mock_noncached"]["distinct"] == 1014
    assert bk["mock_noncached"]["prompt_tokens_distinct"] == 818985
    assert bk["cache_apertus"]["raw"] == 874
    assert bk["cache_apertus"]["distinct"] == 854


def test_viewer_schema_node_tests():
    """Run the v2/v3 viewer display tests (tests/viewer_schema_test.mjs).
    Skipped only when node is genuinely unavailable."""
    import shutil, subprocess
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    p = _repo_results().parent / "tests" / "viewer_schema_test.mjs"
    r = subprocess.run([node, str(p)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
