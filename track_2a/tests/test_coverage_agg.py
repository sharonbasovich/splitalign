"""Token-weighted coverage aggregation tests.

The 0.90 gate metric is sum(valid nonpunct tokens)/sum(matched nonpunct
tokens) across the whole method output — NOT the mean of per-document
ratios. These tests pin that down for unequal doc sizes, punctuation-heavy
docs and invalid-judgment fallbacks, and for the baseline distinction
between emitted list coverage and valid mapped token coverage.
"""
import json

from splitalign import run as _run


def _details(run_dir, mode, items, backend="apertus", split="val"):
    p = run_dir / f"details_{mode}_{backend}_{split}.json"
    p.write_text(json.dumps(items))


def test_unequal_sizes_token_weighted(tmp_path):
    """Two docs of very different sizes: the big doc must dominate the
    aggregate (token-weighted), not a 50/50 mean of ratios."""
    items = [
        # doc 1: 100 matched, all valid
        {"stats": {"matched_tokens": 100, "valid_matched_tokens": 100}},
        # doc 2: 2 matched, none valid (ratio 0.0)
        {"stats": {"matched_tokens": 2, "valid_matched_tokens": 0}},
    ]
    _details(tmp_path, "splitalign", items)
    agg = _run._method_diagnostics(tmp_path, "val", "apertus")["splitalign"]
    # weighted: 100/102 = 0.9804; mean-of-ratios would be 0.5
    assert agg["judge_valid_token_coverage"] == round(100 / 102, 4)
    assert agg["matched_tokens"] == 102
    assert agg["valid_matched_tokens"] == 100


def test_punct_heavy_invalid_fallback(tmp_path):
    """matched/valid counts already exclude punctuation (score.py); the
    aggregate just sums them. Invalid pairs land in fallback, not valid."""
    items = [
        {"stats": {"matched_tokens": 6, "valid_matched_tokens": 0,
                   "invalid_pairs": 1, "invalid_fallback_tokens": 6,
                   "dropped_punct_ids": 40}},
        {"stats": {"matched_tokens": 8, "valid_matched_tokens": 8}},
    ]
    _details(tmp_path, "splitalign", items)
    agg = _run._method_diagnostics(tmp_path, "val", "apertus")["splitalign"]
    assert agg["judge_valid_token_coverage"] == round(8 / 14, 4)
    assert agg["invalid_pairs"] == 1 and agg["invalid_fallback_tokens"] == 6
    assert agg["dropped_punct_ids"] == 40


def test_zero_matched_coverage_is_null(tmp_path):
    _details(tmp_path, "splitalign",
             [{"stats": {"matched_tokens": 0, "valid_matched_tokens": 0}}])
    agg = _run._method_diagnostics(tmp_path, "val", "apertus")["splitalign"]
    assert agg["judge_valid_token_coverage"] is None


def test_baseline_emitted_coverage_distinct(tmp_path):
    """Baseline reports emitted list coverage (fraction of tokens the model
    output a label for), a DIFFERENT metric from valid mapped coverage —
    must not be folded into judge_valid_token_coverage."""
    items = [
        {"emitted_coverage_a": 1.0, "emitted_coverage_b": 0.4, "repairs": 1},
        {"emitted_coverage_a": 0.5, "emitted_coverage_b": None, "repairs": 0},
    ]
    _details(tmp_path, "baseline", items)
    agg = _run._method_diagnostics(tmp_path, "val", "apertus")["baseline"]
    assert agg["emitted_coverage_a"] == 0.75
    assert agg["emitted_coverage_b"] == 0.4      # single-sided mean
    assert agg["repairs"] == 1
    assert "judge_valid_token_coverage" not in agg
