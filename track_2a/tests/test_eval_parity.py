"""Prove our original metric implementation matches the documented spec.

1. ``metricspec.parse_token_labels`` behaves per the documented reference
   harness semantics (order walk, current/next-position match, fallback).
2. ``map_label_from_positive_to_negative`` is the documented s -> 1 - s/5 map.
3. ``evaluate_predictions`` produces the same Spearman as a plain scipy
   computation on the same filtered label arrays.
4. No upstream code is shipped: the vendor tree must not exist.
"""
import json
import sys
from pathlib import Path

import pytest

TRACK = Path(__file__).resolve().parents[1]


def test_no_vendored_upstream_code():
    """Upstream evaluation sources are unlicensed — none may ship."""
    assert not (TRACK / "src" / "vendor").exists(), \
        "vendor tree must not exist: upstream eval code is unlicensed"


def test_parse_token_labels_semantics():
    from splitalign.metricspec import parse_token_labels
    toks = ["a", "b", "c"]
    # token match -> label applied; skip-ahead allowed; unknown skipped
    labs = parse_token_labels(toks, [["a", 5], ["x", 9], ["c", 3]], fallback_label=0.)
    assert labs == [5.0, 0.0, 3.0]
    # non-iterable -> all fallback
    assert parse_token_labels(toks, None, fallback_label=2.) == [2.0, 2.0, 2.0]
    # overlong predictions stop at len(tokens)
    labs = parse_token_labels(toks, [["a", 1], ["b", 1], ["c", 1], ["d", 1]], 0.)
    assert labs == [1.0, 1.0, 1.0]
    # bare-string entries match positions without assigning a label
    labs = parse_token_labels(["a", "b"], ["a", ["b", 4]], 0.)
    assert labs == [0.0, 4.0]
    # non-numeric label keeps the fallback (and does not advance)
    labs = parse_token_labels(["a", "b"], [["a", "oops"], ["b", 2]], 0.)
    assert labs == [0.0, 2.0]
    # out-of-schema entries (empty list, non-sequence, dict) are rejected
    # explicitly — the reference crashes/misbehaves on these; we skip.
    labs = parse_token_labels(["a", "b"], [[], 7, {"a": 1}, ["a", 1],
                                         ["b", 2]], 0.)
    assert labs == [1.0, 2.0]


def test_label_mapping():
    from splitalign.metricspec import map_label_from_positive_to_negative as m
    assert m(5) == 0.0 and m(0) == 1.0 and m(-1) == -1.0
    assert abs(m(4) - 0.2) < 1e-9 and abs(m(3) - 0.4) < 1e-9


def test_spearman_parity_with_scipy(tmp_path):
    """Our LangResult spearman == scipy.stats.spearmanr on filtered arrays."""
    pytest.importorskip("scipy")
    from scipy.stats import spearmanr
    from splitalign import guard
    from splitalign.evaluate import evaluate_predictions
    from splitalign.fetch_data import load_gold_items

    items = load_gold_items("dev/val", "de")[:3]
    gold_path = guard.gold_path("dev/val", "de")
    # fabricate deterministic *varying* predictions (not a model claim)
    recs = []
    for it in items:
        recs.append({"id": it["id"], "text_a": it["text_a"], "text_b": it["text_b"],
                     "labels_a": [round((i % 5) * 0.2, 2) for i in range(len(it["labels_a"]))],
                     "labels_b": [round((i % 5) * 0.2, 2) for i in range(len(it["labels_b"]))]})
    res = evaluate_predictions(recs, gold_path, "de", n_resamples=0)
    # independent recompute
    pl, gl = [], []
    for it in items:
        pl += [round((i % 5) * 0.2, 2) for i in range(len(it["labels_a"]))]; gl += it["labels_a"]
        if not all(v == -1 for v in it["labels_b"]):
            pl += [round((i % 5) * 0.2, 2) for i in range(len(it["labels_b"]))]; gl += it["labels_b"]
    fp = [p for p, g in zip(pl, gl) if g != -1]
    fg = [g for g in gl if g != -1]
    expected = spearmanr(fp, fg).statistic
    assert res.spearman == pytest.approx(expected, abs=1e-9)


def test_bootstrap_zero_omits_ci(tmp_path):
    """bootstrap=0 must not emit bounds (never show lo==hi as a fake CI)."""
    from splitalign import guard
    from splitalign.evaluate import evaluate_split
    from splitalign.fetch_data import load_gold_items
    # run-scoped synthetic fixture: gold labels replayed as predictions
    items = load_gold_items("dev/val", "de")[:3]
    pred = tmp_path / "run"
    pred.mkdir()
    with (pred / "splitalign_mock_admin_de.jsonl").open("w") as f:
        for it in items:
            f.write(json.dumps({"id": it["id"], "labels_a": it["labels_a"],
                                "labels_b": it["labels_b"]}) + "\n")
    res = evaluate_split(pred, "dev/val", langs=("de",), n_resamples=0,
                         prefix="splitalign_mock")
    assert res["per_language"]["de"]["n_samples"] == 3
    lang = res["per_language"]["de"]
    assert "spearman_lo" not in lang and "spearman_hi" not in lang


def test_nan_serialized_as_null_with_reason(tmp_path):
    """Constant predictions -> undefined correlation -> null + reason."""
    from splitalign import guard
    from splitalign.evaluate import evaluate_predictions, _dump
    from splitalign.fetch_data import load_gold_items
    items = load_gold_items("dev/val", "de")[:2]
    recs = [{"id": it["id"], "text_a": it["text_a"], "text_b": it["text_b"],
             "labels_a": [1.0] * len(it["labels_a"]),
             "labels_b": [1.0] * len(it["labels_b"])} for it in items]
    r = evaluate_predictions(recs, guard.gold_path("dev/val", "de"), "de",
                             n_resamples=0)
    d = _dump(r)
    assert d["spearman"] is None
    assert "invalid_reason" in d
    json.dumps(d)  # must be standards-compliant JSON (no NaN literal)


def test_coverage_reported(tmp_path):
    """Coverage fields expose incomplete prediction sets."""
    from splitalign import guard
    from splitalign.evaluate import evaluate_predictions
    from splitalign.fetch_data import load_gold_items
    items = load_gold_items("dev/val", "de")
    gold_path = guard.gold_path("dev/val", "de")
    recs = [{"id": items[0]["id"], "text_a": items[0]["text_a"],
             "text_b": items[0]["text_b"],
             "labels_a": items[0]["labels_a"], "labels_b": items[0]["labels_b"]}]
    res = evaluate_predictions(recs, gold_path, "de", n_resamples=0)
    assert res.n_gold_items == len(items)
    assert res.coverage == pytest.approx(1 / len(items))
    assert len(res.missing_ids) == len(items) - 1


def test_require_full_fails_on_absent_language(tmp_path):
    """--require-full: a required language with zero predictions fails."""
    from splitalign import guard
    from splitalign.evaluate import evaluate_split
    from splitalign.fetch_data import load_gold_items
    items = load_gold_items("dev/val", "de")
    p = tmp_path / "mod_x_admin_de.jsonl"
    with p.open("w") as f:
        for it in items:
            f.write(json.dumps({"id": it["id"], "text_a": it["text_a"],
                                "text_b": it["text_b"],
                                "labels_a": it["labels_a"],
                                "labels_b": it["labels_b"]},
                               ensure_ascii=False) + "\n")
    with pytest.raises(SystemExit, match="entire language"):
        evaluate_split(tmp_path, "dev/val", n_resamples=0,
                       prefix="mod_x", require_full_coverage=True)
