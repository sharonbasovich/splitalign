# SplitAlign — Technical Report

Hack Apertus Online 2026 · Track 2A (UZH SwissGov-RSD) · Team Waterloo Agent Lab · Entrant Sharon Basovich

Repository: https://github.com/sharonbasovich/splitalign · Code license: Apache-2.0
Report license: CC-BY-4.0 · Prediction labels: CDLA-Permissive-2.0 over CC-BY-4.0 source texts

**Status: exploratory, baseline incomplete, historical run affected by a
configuration defect.** The Apertus SplitAlign numbers below are real outputs of run
`20261001T211451Z-8de1aab7` on 30 dev/val documents (10 per language), but that run
used prompt version v1, in which `predict_item` silently defaulted the judge prompt's
target language to German — every fr and it pair was judged with a prompt naming
German (§5.0). The figures are therefore NOT evidence for the intended multilingual
configuration and are kept unchanged, not relabelled or regenerated; no corrected
rerun has been performed (it requires a reviewed bounded plan). The re-implemented
whole-document baseline was declared in the run manifest but produced no output, so no
matched same-ID Apertus baseline exists. An offline same-ID context against published
encoder predictions is reported with paired uncertainty; it supports no win claim.

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
  malformed output; deterministic seed; the target language is a required
  argument, named in every judge prompt and recorded per judgment (`lang_b`).
  Calls are cached on backend|model|prompt_version|kind|split|item|payload keys
  (the payload includes the language pair; the judge kind is keyed under
  `splitalign-prompts-v2-lang`, so no v1 judgment can be served to the
  corrected code) and logged redacted (hashes, token counts, latency — no
  prompt text, no secrets). Logging is run-scoped since this repair:
  `results/runs/<id>/calls.jsonl`, every record tagged `run_id` (cache hits and
  errors included). The historical `results/calls.jsonl` is an UNSCOPED global
  log from earlier code with no `run_id`; it is preserved as-is and cannot be
  attributed per run.
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
- `calibrate` (score-config grid search) does NOT run under the shared budget,
  so it fails closed for `--backend apertus` before any client or network work;
  bounded real calibration is unsupported. The committed `calibration.json`
  came from the mock backend only.

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

## 5. Results (dev/val, 10 docs/language) — historical, defect-affected

### 5.0 Configuration defect in the reported run

Independent review of the repair PR found that `run.py` called `predict_item`
without a language while `pipeline.py` defaulted to `lang="de"`. Consequently in
run `20261001T211451Z-8de1aab7` (prompt version `splitalign-prompts-v1`) the 20
fr/it documents were judged with a prompt stating the target language was German
(the texts themselves were the correct fr/it texts; the similarity prompt is
language-agnostic). The de figures used the intended configuration; the fr/it
figures and the macro did not. Repair: language is now a required argument at
every entry point, named in each prompt and recorded per judgment; the judge
cache key is versioned `v2-lang`. The historical numbers below are reported as
produced and are explicitly not a claim for the intended multilingual method.

Run `20261001T211451Z-8de1aab7` — backend `apertus`, model `swiss-ai/Apertus-v1.5-8B`,
prompt `splitalign-prompts-v1` (defect-affected, see §5.0):

| Method | de | fr | it | Strict macro |
|---|---|---|---|---|
| SplitAlign | 0.254 | 0.267 | 0.224 | **0.248** |

Usage: 30/30 planned items completed; 97 new API requests, 72,624 new tokens,
158.9 s, on top of payload-identical cache hits from the earlier partial run
(cap 500/600k not hit in this run). Bootstrap CIs omitted (n_resamples=0).
Run artifacts: [`results/runs/20261001T211451Z-8de1aab7/`](https://github.com/sharonbasovich/splitalign/tree/main/track_2a/results/runs/20261001T211451Z-8de1aab7).

History (not current results): the first bounded run (`legacy-pre-runscope`,
pre-fix shared cap) completed 24/30 docs (DE 10 / FR 10 / IT 4) at 499
requests / 600,360 tokens before the token cap; earlier 3-doc diagnostics
(a −0.023 figure, an all-NaN baseline file) are quarantined in
`results/runs/legacy-pre-runscope/`.

**Baseline status: incomplete.** `manifest.json` of the run declares `baseline`
intended IDs, but no baseline predictions or summary were produced in that scope
(`viewer/evidence.js` marks it PARTIAL and lists the §5.0 defect). No matched
whole-document Apertus baseline number exists and none is estimated.

### 5.1 Offline same-ID context vs published encoder predictions (no API calls)

`scripts/encoder_context_dev.py` scores the authors' precomputed DEV
predictions (upstream commit `1807a42`, `data/evaluation/encoder_predictions/dev/`,
every ID validated against our dev manifest, files not redistributed — URLs and
sha256 in [`results/encoder_context/`](https://github.com/sharonbasovich/splitalign/tree/main/track_2a/results/encoder_context))
on exactly the same 30 IDs with the same metric code. Comparator policy was fixed
before looking: EuroBERT-210m is the preregistered reference; four named
mmBERT-base DiffAlign variants (`parallel_en-de`, `parallel_en-sw`,
`parallel_en-de-fr-it`, `parallel_all_1epoch`) are **selected comparison
context** — upstream publishes more mmBERT variants, so this set is neither
exhaustive nor proof against selection effects; it was chosen before any score was
seen and is reported unchanged. Paired = document-level paired bootstrap of the
strict-macro difference (SplitAlign − system), B = 2000, seed 0. SplitAlign's row
is the defect-affected v1 run (§5.0).

| System (same 30 IDs) | de | fr | it | Strict macro | Paired diff, 95% CI |
|---|---|---|---|---|---|
| SplitAlign (v1 run, §5.0) | 0.254 | 0.267 | 0.224 | 0.248 | — |
| EuroBERT-210m DiffAlign (reference) | 0.055 | 0.196 | 0.198 | 0.150 | +0.099 [−0.008, +0.184] |
| mmBERT parallel_en-de | 0.180 | 0.230 | 0.266 | 0.225 | +0.023 [−0.073, +0.102] |
| mmBERT parallel_en-sw | 0.170 | 0.228 | 0.271 | 0.223 | +0.025 [−0.070, +0.104] |
| mmBERT parallel_en-de-fr-it | 0.199 | 0.243 | 0.316 | 0.253 | −0.004 [−0.103, +0.077] |
| mmBERT parallel_all_1epoch | 0.206 | 0.243 | 0.299 | 0.249 | −0.001 [−0.095, +0.084] |

Reading: SplitAlign is numerically above the preregistered EuroBERT reference on
these 30 documents, but the paired 95% interval includes zero; against the
stronger mmBERT variants the difference is indistinguishable from zero. This is
**no evidence of improvement** over published encoders, and same-ID subset
figures are not comparable to whole-dev (168 docs/lang) numbers, and the
SplitAlign row carries the §5.0 defect. Prior-work figures from the dataset paper
are not reproduced here; see the upstream repository for the authors' reported
results.

## 6. Limitations

- **The only real Apertus run reported used the wrong judge-prompt language for
  fr/it (§5.0).** No corrected multilingual run exists; one needs a reviewed
  bounded plan (budget, IDs, cache policy) before any new inference.
- Historical call log `results/calls.jsonl` is unscoped (no `run_id`); per-run
  call provenance exists only for runs made after this repair.
- `calibrate` is mock-only (fails closed on a real backend; no shared budget).
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

- Code commit for THIS report: see the repository tag/commit on the submission
  form. The 30-doc run manifest predates source-commit recording: its code
  state lies in `7ad62c3..8480423` per git history and may have included
  then-uncommitted local fixes; the field is left null rather than invented.
  Every new run now records `source.commit`, `source.dirty`, package and
  prompt versions, gold-file sha256 per language, score config and CLI caps
  in `manifest.json`.
- `make run` (root) → Docker build + `pipeline --backend mock --limit 3` on a
  clean checkout; real backend via `SPLITALIGN_BACKEND=apertus` +
  `APERTUS_API_KEY`/`APERTUS_API_BASE`/`APERTUS_MODEL` env vars — contradictory
  or missing config fails closed, never silently mocks.
- Seeds: `--seed 0` everywhere; `temperature=0`; deterministic mock; payload-
  keyed cache makes reruns bit-identical for identical prompts.
- Evidence viewer: `track_2a/viewer/` (static, bilingual, per-token heat maps
  with real-vs-mock labels; failed or text-less items render an explicit error
  state). `export-viewer` requires an explicit `--run` and embeds run id,
  intended vs produced modes, completed/intended counts, coverage, cap/partial
  status, matched-ID info and known configuration defects; the newest run is
  never promoted implicitly. Inference provenance (`inference`: prompt version,
  model, package — from the run manifest and per-item records, null where
  unrecorded) is kept separate from `exporter` (the code that wrote the file),
  so re-exporting run `8de1aab7` labels it prompt `v1`, never the current
  version.
- Demo recordings in `track_2a/demo/` were made on the pre-repair viewer
  showing the §5.0-affected run; they are historical and are not a corrected
  demo (see `demo/README.md`).
- Offline encoder context: `PYTHONPATH=src python scripts/encoder_context_dev.py
  --run results/runs/<id>` (dev files only, no model access).

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
