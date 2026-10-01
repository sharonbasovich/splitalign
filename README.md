# SplitAlign

**Hack Apertus Online 2026 — Track 2A (UZH SwissGov-RSD)**
Team: Waterloo Agent Lab — Sharon Basovich

Token-level recognition of semantic differences between related
cross-lingual Swiss government documents (English ↔ German/French/Italian),
built on hosted **Apertus** inference.

## Run

```bash
make run
```

Builds a Docker image and runs the pipeline on a dev/val smoke subset —
works credential-free via the clearly-labelled deterministic `mock` backend.
For real Apertus inference see `track_2a/README.md` (env vars
`APERTUS_API_BASE`, `APERTUS_API_KEY`, `APERTUS_MODEL`,
`SPLITALIGN_BACKEND=apertus`).

## Layout

- `track_2a/` — the competition deliverable (source, data, viewer, report)
- `Dockerfile`, `Makefile` — clean-checkout run entrypoint
- `LICENSE` (Apache-2.0), `NOTICE` (attributions + disclosures)

## Integrity notes

- Held-out splits of SwissGov-RSD are fenced off by `splitalign.guard` —
  a fail-closed development guard (path allowlist + dev-ID manifest +
  symlink checks); only `dev/train` + `dev/val` are present or reachable.
- Evaluation implements the documented official token-label metric with
  original Apache-2.0 code (`splitalign/metricspec.py`) — no upstream
  source is vendored (upstream ships no repository-level license).
  Parity vs the spec + scipy is tested in `tests/test_eval_parity.py`.
- All artifacts label `mock` vs `apertus` provenance explicitly.
