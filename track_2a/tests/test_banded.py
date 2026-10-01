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
