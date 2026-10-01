# SplitAlign — Technical Report

Hack Apertus Online 2026 · Track 2A (UZH SwissGov-RSD) · Team Waterloo Agent Lab · Entrant Sharon Basovich

Repository: https://github.com/sharonbasovich/splitalign · Code license: Apache-2.0
Report license: CC-BY-4.0 · Prediction labels: CDLA-Permissive-2.0 over CC-BY-4.0 source texts

**Status: exploratory.** Apertus SplitAlign results below are measured on 30 dev/val
documents (10 per language). The matched re-implemented-baseline comparison is pending
(see Results); no comparative performance claim is made until matched-ID evidence exists.

## 1. Summary

SplitAlign scores token-level cross-lingual semantic difference between aligned
Swiss administrative documents (de/fr/it). Instead of asking a hosted LLM to label
every token in one shot, it decomposes the task: segment → align sentences
cross-lingually via monotone dynamic programming → score aligned pairs with short
Apertus judgments (graded difference 0–5 plus verbatim differing spans) → map sentence
scores back onto the original whitespace-token arrays. Original token indices are
preserved throughout segmentation, so outputs match the official token-level label
space exactly, including asymmetric additions and omissions.

## 2. Architecture

- `segment.py` — sentence/paragraph segmentation that records each segment's token
  offset so every downstream score can be projected back to exact token positions.
- `align.py` — monotone DP alignment over cross-lingual sentence pairs with 1:0,
  0:1, 1:1, 1:2, and 2:1 (merge) operations; segment-pair similarity comes from
  batched Apertus calls under a **banded candidate policy** (|i/(n−1) − j/(m−1)|
  ≤ 0.2 or ±2 absolute) — measured 304,968 → 108,260 candidate pairs (35%) on the
  dev corpus; far-off-diagonal pairs score 0 without an API call.
- `judge.py` — structured-JSON Apertus judgments on aligned pairs: graded
  difference plus verbatim differing spans; robust parse with repair/salvage on
  malformed output; deterministic seed; all calls cached on
  backend|model|prompt_version|kind|split|item|payload keys and logged
  (redacted: hashes, token counts, latency — no prompt text, no secrets).
- `score.py` — maps per-op judgments onto token arrays: uniform sentence score
  plus span boosts on reported differing spans; explicit omission/addition labels.
- `guard.py` — held-out firewall: only `dev/train` and `dev/val` paths/IDs may be
  loaded; fail-closed on disallowed splits, symlinks, or unmanifested IDs.
  *The manifest is a mutable local development guard, not an integrity proof.*
- `evaluate.py` + `metricspec.py` — original Apache-2.0 implementation of the
  official token-level Spearman/Kendall metric and `1 − s/5` label mapping,
  written from the metric specification (zero vendored upstream code — upstream
  ships no repo-level LICENSE). Nonfinite values serialize as JSON `null` with an
  explicit `invalid_reason`; the primary macro is **strict**: null whenever any
  required language is missing or undefined, with a clearly labeled
  finite-language descriptive mean reported separately.
- `apertus.py` — OpenAI-compatible CSCS client: rate limiting, exponential
  backoff, retry accounting at the **outbound-attempt boundary** (every HTTP
  attempt including retries/failures counts against the shared cap), redacted
  call logging.
- `run.py` — CLI: `fetch-data`, `predict`, `baseline`, `pipeline`, `calibrate`,
  `evaluate`, `export-viewer`, `selftest`. Every run writes an immutable
  `results/runs/<ts>-<id>/` scope with a pre-inference intended-ID manifest;
  methods are compared only on the exact intersection of produced IDs.

## 3. Exact Apertus use

- Endpoint `https://api.inference.cscs.ch/v1`, model `swiss-ai/Apertus-v1.5-8B`
  (non-thinking), chat-completions API, `temperature=0`, fixed seed.
- Apertus performs (a) batched cross-lingual sentence-pair similarity used as DP
  scores and (b) the graded judgments; no fine-tuning or hidden-state access is
  assumed on the hosted API. k=1 self-consistency (k=3 reserved pending measured
  benefit vs cost).
- Budget: a shared `ApiBudget` caps NEW (noncached) requests AND tokens across
  both methods (`--max-requests 500`, `--max-tokens 600000` for the reported run);
  accounting sits at the outbound-attempt boundary so retries/failures count;
  a single in-flight bounded request is the only possible overshoot. Identical
  cache hits are free by construction.

## 4. Data & licensing

- ZurichNLP/swissgov-rsd — CC-BY-4.0 per the dataset's Hugging Face card.
  Only `dev/train` (fetch-allowlisted) and `dev/val` data are used; the held-out
  test split is never fetched, read, or evaluated in this build.
- Official `list_to_drop.txt` semantics: dev items with id suffixes
  {18,100,106,153,196} per language are excluded in one reported variant;
  results are reported both with and without that exclusion.
- Generated prediction labels are released under CDLA-Permissive-2.0 over the
  CC-BY-4.0 source texts; all code is original Apache-2.0.
- The 'baseline' method is our **re-implemented** whole-document
  token-annotation prompt (same response contract, not the upstream template).

## 5. Results (dev/val, 10 docs/language)

Run `20261001T211451Z-8de1aab7` — backend `apertus`, model `swiss-ai/Apertus-v1.5-8B`:

| Method | de | fr | it | Strict macro |
|---|---|---|---|---|
| SplitAlign | 0.254 | 0.267 | 0.224 | **0.248** |

Usage: 30/30 planned items completed; 97 new API requests, 72,624 new tokens
(payload-identical cache reuse from prior bounded runs; cap 500/600k not hit).
Bootstrap CIs omitted (n_resamples=0). The matched baseline comparison is
pending an authorized bounded run; earlier pre-fix diagnostics (including a
−0.023 figure and an all-NaN baseline file) are quarantined in
`results/runs/legacy-pre-runscope/` and are not current results.

For reference only (paper prior work, NOT our measurements): author-prompted LLM
predictions ≈ −0.9 macro Spearman, DiffAlign (mmBERT) 18.6, sentence-label
oracle 56.4 — oracles are upper bounds, not deployable baselines.

## 6. Limitations

- Sentence-level judgments map to uniform token scores except on reported
  differing spans; fine-grained in-sentence variation is approximated.
- Alignment is monotone; cross-lingual reordering is not modeled.
- Similarity banding never scores far-off-diagonal pairs (0.0 by policy);
  measured candidate coverage is reported above.
- Mock backend (`MOCK-NOT-APERTUS`) exists for offline plumbing tests only and
  is never reported as model performance.
- Held-out claims are scoped to this clean build; a narrow historical research
  audit (filenames/line counts plus one ~800-char DEV item) is disclosed in the
  repository, not claimed as "never touched".

## 7. Reproducibility

- Commit: see `results/runs/*/manifest.json` and git history (results run at
  7ad62c3..8480423).
- `make run` (root) → Docker build + `pipeline --backend mock --limit 3` on a
  clean checkout; real backend via `SPLITALIGN_BACKEND=apertus` +
  `APERTUS_API_KEY`/`APERTUS_API_BASE`/`APERTUS_MODEL` env vars — contradictory
  or missing config fails closed, never silently mocks.
- Seeds: `--seed 0` everywhere; `temperature=0`; deterministic mock; payload-
  keyed cache makes reruns bit-identical for identical prompts.
- Evidence viewer: `track_2a/viewer/` (static, bilingual, per-token heat maps
  with real-vs-mock labels and provenance banner).

## 8. AI-assistance disclosure

Strategy/design brief and implementation were produced by AI assistants
(a separate reviewer session plus Devin). Sharon Basovich supplied project
goals, participation/public-release approvals, and secure credential
provisioning. Team label "Waterloo Agent Lab" carries no other humans.

## 9. References

1. Hack Apertus Online 2026 Track 2A brief — hackapertus.notion.site/track-2a-uzh
2. ZurichNLP/SwissGov-RSD — github.com/ZurichNLP/SwissGov-RSD (pinned dev data commit)
3. HF dataset card — huggingface.co/datasets/ZurichNLP/swissgov-rsd (CC-BY-4.0)
4. CSCS Inference API — docs.cscs.ch/services/inference/api
5. Apertus — swiss-ai/Apertus-v1.5-8B
