# SplitAlign — Track 2A (UZH SwissGov-RSD)

Cross-lingual token-level semantic-difference recognition for Swiss
government documents. Submissions must use the Apertus model family —
SplitAlign uses hosted Apertus 8B for both sentence similarity and
pair-level difference judgments. See [the existing report](report/technical_report.md) for historical details.

## Packaging candidate and safe checks

This draft changes packaging only. The organizer's plain clean-checkout
`make run` contract and final prediction coverage remain pending clarification.
Existing reports, incident records and result artifacts are unchanged.

`make preflight` builds the container and checks local configuration without
loading a dataset or contacting a model. Without model configuration it reports
mock configuration only. `make -C track_2a test-contract` runs 18 synthetic
standard-library contract tests without Docker, datasets or model calls.

A plain `make run` fails before Docker execution. An intentional pipeline needs
an absolute `SPLITALIGN_DATA_DIR` mounted read-only and explicit `RUN_ARGS`
beginning with `pipeline`, including split, language, positive limit,
nonnegative offset, request cap and conditional token cap. No dataset or spending
default is selected automatically. The loader reads the selected language file
before applying limit and offset; those arguments alone do not establish which
records were accessed.

### Model configuration

The launcher accepts official `LLM_BASE_URL`, `LLM_API_KEY` and `LLM_NAME`,
alongside legacy `APERTUS_API_BASE`, `APERTUS_API_KEY` and `APERTUS_MODEL`.
When Apertus is selected, conflicting aliases and partial configuration fail
closed. Real-provider
configuration also requires a separately reviewed `APERTUS_BOUND_SPEC_JSON`.
Preflight does not authenticate a key or validate the bound's conservatism.
Credentials are forwarded by variable name, never embedded in source.

The earlier source-only candidate built and completed a selected synthetic mock
pipeline in Docker on 3 October 2026, with runtime networking disabled. See
[validation scope and provenance](docs/CONTAINER_VALIDATION.md) and
[packaging details](PACKAGING_CANDIDATE.md). This draft has later documentation
changes and preserves 15 original files' trailing whitespace; the Docker-tested
archive is identified separately. It has not been rerun as a final Git checkout.

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
  results/           # committed predictions, evals; runs/<id>/calls.jsonl = run-scoped
                     # redacted call log (calls.jsonl at this level = historical, unscoped)
                     # (only results/cache/ is gitignored)
```

## Commands (inside track_2a, `PYTHONPATH=src`)

```bash
python -m splitalign.run fetch-data        # verify allowlisted dev data
python -m splitalign.run predict  --split val --lang de --limit 5 --backend mock
python -m splitalign.run baseline --split val --lang de --limit 5 --backend mock
python -m splitalign.run evaluate --pred results/predictions --split val --prefix splitalign_mock
python -m splitalign.run calibrate --lang all          # grid on dev/train (mock backend only;
                                                       # fails closed for apertus — no shared budget)
python -m splitalign.run export-viewer --split val --run results/runs/<run-id>  # explicit scope -> viewer/evidence.js
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
