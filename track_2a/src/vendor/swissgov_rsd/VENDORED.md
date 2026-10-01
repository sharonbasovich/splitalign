# Vendored evaluation modules

Source: <https://github.com/ZurichNLP/SwissGov-RSD> @ commit
`1807a42100e742ed03d337c54c4b9ea86995f565`
License: MIT ((c) 2023 University of Zurich) — see `LICENSE` in this directory.

## Why vendored

Upstream `python -m scripts.evaluate_predictions_admin` cannot be imported
cleanly: `scripts/__init__.py` pulls in `project_labels.py`, which reads an
unrelated `.openai/key.txt` at import time. We therefore vendor only the
modules needed for official scoring and replicate the driver loop in
`splitalign/evaluate.py`. No scoring semantics were changed.

## Files

| Vendored path | Upstream path | Integrity |
|---|---|---|
| `evaluation/__init__.py` | `evaluation/__init__.py` | byte-identical (sha256-pinned) |
| `evaluation/predictions.py` | `evaluation/predictions.py` | byte-identical |
| `evaluation/utils.py` | `evaluation/utils.py` | byte-identical |
| `rsd/recognizers/utils.py` | `rsd/recognizers/utils.py` | **partial**: `DifferenceSample` and `tokenize` verbatim; torch-dependent helpers (`cos_sim`, `pairwise_dot_score`, …) elided because they are unused by evaluation and force a heavy import |

Pin verification: `tests/test_eval_parity.py::test_vendored_file_integrity`
asserts the exact upstream sha256 of each byte-identical file.

## What we did NOT vendor

`scripts/evaluate_predictions_admin.py` — inspected for semantics, not
imported (import side effect). Its per-language loop, `-1` gold filtering,
length-mismatch padding/truncation, `labels_b` all-`-1` skip, and
`nlpstats` correlate/bootstrap calls are reproduced in
`splitalign/evaluate.py` and proven equivalent on synthetic + dev fixtures
by `tests/test_eval_parity.py`.
