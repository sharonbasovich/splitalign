# SplitAlign — Track 2A (UZH SwissGov-RSD)

Cross-lingual token-level semantic-difference recognition for Swiss
government documents. Submissions must use the Apertus model family —
SplitAlign uses hosted Apertus 8B for both sentence similarity and
pair-level difference judgments. See `technical_report.md` for details.

## Run it

From the repository root (or this directory):

```bash
make run
```

Builds a Docker image and runs the full pipeline on a bounded dev/val smoke
subset, writing predictions, metrics and the evidence viewer to `out/`.
It works on a clean checkout with **no credentials** — it then runs the
deterministic `mock` backend, which is clearly labelled in every artifact
and is *not* model output.

### Real Apertus inference

Set these environment variables, then `SPLITALIGN_BACKEND=apertus make run`:

| Variable | Secret? | Meaning |
|---|---|---|
| `APERTUS_API_BASE` | no | OpenAI-compatible base URL (`{base}/chat/completions`) |
| `APERTUS_API_KEY` | **yes** | bearer token (CSCS hosted inference) |
| `APERTUS_MODEL` | no | e.g. `swiss-ai/Apertus-v1.5-8B` |
| `APERTUS_TIMEOUT_S` | no | request timeout (default 120) |
| `APERTUS_RPS` | no | max requests/second (default 2) |

Missing config fails honestly with the variable names — never a silent mock.

## Layout

```
track_2a/
  Makefile           # docker build + run targets
  data/              # dev/train + dev/val gold only — see data/README.md
  src/splitalign/    # guard, segment, align, judge, score, evaluate,
                     # metricspec, run, ... (all original Apache-2.0 code)
  tests/             # pytest: firewall, segmentation, DP, scoring, parity
  viewer/            # static bilingual evidence viewer (evidence.js generated)
  docs/              # licenses + report assets
  results/           # committed predictions, evals, redacted call log
                     # (only results/cache/ is gitignored)
```

## Commands (inside track_2a, `PYTHONPATH=src`)

```bash
python -m splitalign.run fetch-data        # verify allowlisted dev data
python -m splitalign.run predict  --split val --lang de --limit 5 --backend mock
python -m splitalign.run baseline --split val --lang de --limit 5 --backend mock
python -m splitalign.run evaluate --pred results/predictions --split val --prefix splitalign_mock
python -m splitalign.run calibrate --lang all          # grid on dev/train
python -m splitalign.run export-viewer --split val     # -> viewer/evidence.js
```

## Data

`data/` ≈ 5 MB (limit 100 MB). Dev/train + dev/val gold files only, pinned
commit, sha256-manifested. Held-out splits are fenced off by a fail-closed
development guard (`splitalign/guard.py`): path allowlist, dev-ID manifest,
symlink-component rejection — the manifest is a local mutable file, so the
guard bounds this build's data access rather than proving anything about
other environments.

## License

Code Apache-2.0 (all original — no upstream source redistributed); report
CC-BY-4.0; prediction label files CDLA-Permissive-2.0 over underlying
texts that remain CC-BY-4.0 (dataset license per its Hugging Face card).
See `NOTICE`.
