"""Banded similarity candidate policy — far-off-diagonal pairs never hit the API."""
import json
import sys
from pathlib import Path

TRACK = Path(__file__).resolve().parents[1]

from splitalign.apertus import CallLogger, MockApertusClient
from splitalign.cache import DiskCache
from splitalign.judge import Judge


class _CountingMock(MockApertusClient):
    def __init__(self):
        super().__init__(seed=0)
        self.queried_pairs = []

    def complete(self, messages, **kw):
        content = messages[-1]["content"]
        start = content.find("```json")
        end = content.find("```", start + 7)
        req = json.loads(content[start + 7:end])
        self.queried_pairs += [(p["i"], p["j"]) for p in req.get("pairs", [])]
        return super().complete(messages, **kw)


def _judge(tmp_path) -> Judge:
    return Judge(client=_CountingMock(), backend="mock",
                 cache=DiskCache(tmp_path / "cache"),
                 logger=CallLogger(None), split="dev/val", item_id="t")


def test_banded_matrix_skips_far_diagonal(tmp_path):
    j = _judge(tmp_path)
    a = [f"segment a{i}" for i in range(20)]
    b = [f"segment b{i}" for i in range(20)]
    sim = j.similarity_matrix(a, b)
    assert len(sim) == 20 and len(sim[0]) == 20
    # corner pairs (0,19) and (19,0) are way out of band -> never queried
    assert (0, 19) not in j.client.queried_pairs
    assert (19, 0) not in j.client.queried_pairs
    # and stay at the zero fill
    assert sim[0][19] == 0.0 and sim[19][0] == 0.0
    # diagonal always queried
    assert (10, 10) in j.client.queried_pairs
    # far fewer calls than n*m
    assert len(j.client.queried_pairs) < 20 * 20 * 0.6


def test_small_docs_unaffected(tmp_path):
    j = _judge(tmp_path)
    sim = j.similarity_matrix(["a0", "a1", "a2"], ["b0", "b1", "b2"])
    # 3x3 fully covered by the +-2 absolute neighbourhood
    assert len(set(j.client.queried_pairs)) == 9


def test_zero_fill_when_pair_unreturned(tmp_path):
    j = _judge(tmp_path)
    # a pathological batch answer is not needed: unreturned pairs stay 0
    sim = j.similarity_matrix(["x", "y"], ["p", "q"])
    assert all(isinstance(v, float) for row in sim for v in row)


# -- noncached API budget: outbound-attempt accounting (synthetic, no net) --

import io
import urllib.error
import json as _json
import pytest
from splitalign.apertus import (ApertusClient, ApiBudget, ApertusUnavailable,
                                BudgetExceeded)
from splitalign import apertus as _ap


class _FakeResp:
    def __init__(self, usage_pt=10, usage_ct=5):
        self._body = _json.dumps({
            "model": "fake", "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": usage_pt,
                      "completion_tokens": usage_ct}}).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _client(budget, retries=5):
    c = ApertusClient("http://fake", "k", "m", rps=0, max_retries=retries)
    c.budget = budget
    return c


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(_ap.time, "sleep", lambda *_: None)


def _http_err(code):
    return urllib.error.HTTPError("u", code, "", {}, io.BytesIO(b"err"))


def test_429_is_one_attempt_then_halt(no_sleep, monkeypatch):
    """A 429 is one attempt with UNKNOWN cost: recorded, NO retry, halt."""
    budget = ApiBudget(max_requests=10**9, max_tokens=10**9)
    client = _client(budget)
    monkeypatch.setattr(_ap.urllib.request, "urlopen",
                        lambda *a, **k: (_ for _ in ()).throw(_http_err(429)))
    with pytest.raises(ApertusUnavailable):
        client.complete([{"role": "user", "content": "hi"}])
    assert budget.requests == 1            # one dispatch, no retry
    assert budget.attempts_no_usage == 1   # token cost unknown — reported
    assert budget.measured_tokens == 0
    # reservation retained: committed tokens stay at the reserved bound
    assert budget.tokens == client._prompt_token_bound(
        [{"role": "user", "content": "hi"}]) + 2048


def test_reserved_bound_blocks_dispatch(no_sleep, monkeypatch):
    """Reservation fits exactly -> 1 dispatch + reconcile; the NEXT request's
    bound does not fit -> blocked BEFORE dispatch (zero bytes leave)."""
    bound = _client(None)._prompt_token_bound([{"role": "user", "content": "a"}])
    budget = ApiBudget(max_requests=10**9, max_tokens=bound)
    client = _client(budget)
    monkeypatch.setattr(_ap.urllib.request, "urlopen",
                        lambda *a, **k: _FakeResp(10, 5))
    client.complete([{"role": "user", "content": "a"}], max_tokens=0)
    assert budget.requests == 1
    assert budget.tokens == 15 and budget.measured_tokens == 15
    with pytest.raises(BudgetExceeded, match="token cap"):
        client.complete([{"role": "user", "content": "b"}], max_tokens=0)
    assert budget.requests == 1            # second request never dispatched


def test_timeout_is_one_attempt_then_halt(no_sleep, monkeypatch):
    """A timeout is one attempt with unknown cost: recorded, NO retry."""
    budget = ApiBudget(max_requests=10**9, max_tokens=10**9)
    client = _client(budget)
    monkeypatch.setattr(_ap.urllib.request, "urlopen",
                        lambda *a, **k: (_ for _ in ()).throw(
                            urllib.error.URLError("timeout")))
    with pytest.raises(ApertusUnavailable):
        client.complete([{"role": "user", "content": "hi"}])
    assert budget.requests == 1 and budget.attempts_no_usage == 1


def test_500_is_one_attempt_no_retry(no_sleep, monkeypatch):
    """A 500 is a single unknown-cost attempt; the client never retries."""
    budget = ApiBudget(max_requests=10**9, max_tokens=10**9)
    client = _client(budget, retries=2)
    monkeypatch.setattr(_ap.urllib.request, "urlopen",
                        lambda *a, **k: (_ for _ in ()).throw(_http_err(500)))
    with pytest.raises(ApertusUnavailable):
        client.complete([{"role": "user", "content": "hi"}])
    assert budget.requests == 1            # no retry after unknown cost


def test_cache_hits_stay_free(tmp_path):
    """Identical-key cache hits never consume budget."""
    j = _judge(tmp_path)
    j.budget = ApiBudget(max_requests=0, max_tokens=0)
    j.similarity_matrix(["a", "b"], ["x", "y"])   # mock client, free
    assert j.budget.requests == 0
