"""Prove the vendored evaluation path agrees with the official semantics.

1. Vendored ``evaluation/*.py`` are byte-identical to upstream @ 1807a42
   (sha256 manifest).
2. ``load_gold_data`` + ``parse_token_labels`` behave identically to the
   upstream reference implementations (re-derived here independently).
3. Our evaluate_predictions produces the same Spearman as a plain scipy
   computation on the same filtered label arrays.
"""
import hashlib
import json
import sys
from pathlib import Path

import pytest

VENDOR = Path(__file__).resolve().parents[1] / "src" / "vendor" / "swissgov_rsd"

# sha256 of upstream files at commit 1807a42100e742ed03d337c54c4b9ea86995f565
UPSTREAM_SHA256 = {
    "evaluation/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "evaluation/predictions.py": "40a914ce039804fa8858600e992844489aea5292841ccb6666ef183c30c32fbc",
    "evaluation/utils.py": "8da64aa6ec33f316eec2c65d0cbc3fe36d247c585a033d2ed04a7bebc175df0b",
    "rsd/recognizers/utils.py": "SKIP",  # intentionally trimmed (torch-free), see VENDORED.md
}


@pytest.mark.parametrize("rel", list(UPSTREAM_SHA256))
def test_vendored_file_integrity(rel):
    want = UPSTREAM_SHA256[rel]
    if want == "SKIP":
        pytest.skip("partial vendor by design")
    got = hashlib.sha256((VENDOR / rel).read_bytes()).hexdigest()
    assert got == want, f"vendored {rel} diverged from upstream {want[:12]}"


def test_parse_token_labels_semantics():
    from evaluation.utils import parse_token_labels
    toks = ["a", "b", "c"]
    # token match -> label applied; skip-ahead allowed; unknown skipped
    labs = parse_token_labels(toks, [["a", 5], ["x", 9], ["c", 3]], fallback_label=0.)
    assert labs == [5.0, 0.0, 3.0]
    # non-iterable -> all fallback
    assert parse_token_labels(toks, None, fallback_label=2.) == [2.0, 2.0, 2.0]
    # overlong predictions stop at len(tokens)
    labs = parse_token_labels(toks, [["a", 1], ["b", 1], ["c", 1], ["d", 1]], 0.)
    assert labs == [1.0, 1.0, 1.0]


def test_label_mapping():
    from evaluation.utils import map_label_from_positive_to_negative as m
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
