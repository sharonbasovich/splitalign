"""End-to-end on synthetic items + real dev items via the MOCK backend.
Mock outputs exercise plumbing; they are never performance evidence."""
import json
from pathlib import Path

import pytest

from splitalign.fetch_data import load_gold_items
from splitalign.pipeline import make_judge, predict_baseline_item, predict_item


def _synth_item():
    return {
        "id": "synthetic_0",
        "text_a": "Water policy . Switzerland protects its lakes .",
        "text_b": "Wasserpolitik . Die Schweiz schützt ihre Seen .",
    }


def test_predict_item_synthetic(tmp_path):
    judge, backend = make_judge("mock", tmp_path, "dev/train", "synthetic_0", 0)
    assert backend == "mock"
    out = predict_item(_synth_item(), judge)
    rec = out["record"]
    assert len(rec["labels_a"]) == len(rec["text_a"].split())
    assert len(rec["labels_b"]) == len(rec["text_b"].split())
    assert out["detail"]["coverage"]["a_complete"]
    assert out["detail"]["coverage"]["b_complete"]


def test_predict_deterministic(tmp_path):
    item = _synth_item()
    j1, _ = make_judge("mock", tmp_path / "c1", "dev/train", item["id"], 0)
    j2, _ = make_judge("mock", tmp_path / "c2", "dev/train", item["id"], 0)
    r1 = predict_item(item, j1)["record"]
    r2 = predict_item(item, j2)["record"]
    assert r1["labels_a"] == r2["labels_a"]
    assert r1["labels_b"] == r2["labels_b"]


def test_cache_hit_second_run(tmp_path):
    item = _synth_item()
    j1, _ = make_judge("mock", tmp_path, "dev/train", item["id"], 0)
    predict_item(item, j1)
    j2, _ = make_judge("mock", tmp_path, "dev/train", item["id"], 0)
    out = predict_item(item, j2)
    cached_flags = [j["cached"] for j in out["detail"]["judgments"] if j]
    assert cached_flags and all(cached_flags)


def test_baseline_synthetic(tmp_path):
    judge, _ = make_judge("mock", tmp_path, "dev/train", "synthetic_0", 0)
    out = predict_baseline_item(_synth_item(), judge)
    rec = out["record"]
    assert len(rec["labels_a"]) == len(rec["text_a"].split())


def test_real_dev_item_mock(tmp_path):
    items = load_gold_items("dev/val", "de")[:1]
    judge, _ = make_judge("mock", tmp_path, "dev/val", items[0]["id"], 0)
    out = predict_item(items[0], judge)
    assert len(out["record"]["labels_a"]) == len(items[0]["labels_a"])
    assert len(out["record"]["labels_b"]) == len(items[0]["labels_b"])


def test_apertus_missing_credentials_fails_honestly(tmp_path, monkeypatch):
    for v in ("APERTUS_API_BASE", "APERTUS_API_KEY", "APERTUS_MODEL"):
        monkeypatch.delenv(v, raising=False)
    from splitalign.apertus import MissingCredentials
    with pytest.raises(MissingCredentials) as e:
        make_judge("apertus", tmp_path, "dev/val", "x", 0)
    msg = str(e.value)
    assert "APERTUS_API_BASE" in msg and "APERTUS_API_KEY" in msg and "APERTUS_MODEL" in msg


def test_backend_resolution_single_source(monkeypatch):
    """env vs CLI must be a single source; contradictions are config errors."""
    import pytest
    from splitalign.run import resolve_backend
    monkeypatch.delenv("SPLITALIGN_BACKEND", raising=False)
    assert resolve_backend(None) == "mock"
    assert resolve_backend("apertus") == "apertus"
    monkeypatch.setenv("SPLITALIGN_BACKEND", "apertus")
    assert resolve_backend(None) == "apertus"      # env supplies the default
    assert resolve_backend("apertus") == "apertus"  # agreement is fine
    with pytest.raises(SystemExit):                 # contradiction is an error
        resolve_backend("mock")
    monkeypatch.setenv("SPLITALIGN_BACKEND", "bogus")
    with pytest.raises(SystemExit):
        resolve_backend(None)
