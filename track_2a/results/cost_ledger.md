# Cumulative cost ledger — all historical actual runs

Sanitized (no secrets). Token figures count only provider-reported `usage`;
attempts include retries/failures at the outbound boundary for run-scoped
records. `results/calls.jsonl` (pre-run-scope era) has `run_id: null` on every
record, so pre-scope processes can only be reported as one window — per-process
split for that era is a **missing record**, stated as such.

## Run-scoped era (per-run records exist)

| run | backend | new attempts | tokens (prompt+completion) | unknown-usage attempts | errors | cache hits | status |
|---|---|---|---|---|---|---|---|
| 20261002T082807Z-06698fcf | apertus | 0 | 0 | 0 | 0 | 0 | aborted before any call (missing env config); manifest only |
| 20261002T082810Z-76c9485d | apertus | 22 | 8,098 (7,127+971) | 0 | 0 | 10 | pilot: 6/6 IDs both methods; see `erratum_sidecar.json` for per-method split (splitalign 22 new + 6 cached; baseline 0 new + 4 cached) |
| 20261001T212533Z-1361c568 | mock | 0 real | 0 | — | — | — | mock backend only; no real API cost |

## Pre-run-scope era (lifetime `results/calls.jsonl`, window
## 2026-10-01T19:45Z – 21:25Z, all records `run_id: null`)

Covers all superseded processes incl. run `20261001T211451Z-8de1aab7`
(bug-affected: German judge prompts on FR/IT — never usable as corrected-config
proof) and the interrupted baseline continuation (0 persisted records).

| metric | value |
|---|---|
| noncached logical calls | 2,260 |
| cache hits | 1,528 |
| tokens | 2,114,006 prompt + 544,843 completion = **2,658,849** |
| attempts lacking usage | 3 (unknown token cost — never counted as zero) |
| recorded errors | 3 |
| entries with null backend/model | 557 (historical, pre-provenance-fix — left unmodified, not retroactively filled) |

**Missing records (stated):** the per-process split inside the pre-scope
window cannot be reconstructed — no per-run attribution exists for it.
Run `20261001T211451Z-8de1aab7` has no run-scoped `calls.jsonl`
(predates that feature); its usage is inside the window above and its own
summary reported shared-budget totals (`new_api_requests` 97 /
`new_api_tokens` 72,624) — see run-dir erratum coverage.

`legacy-pre-runscope/` holds artifacts only (no call records) — usage unknown,
not zero.

## Totals

- New outbound calls, all eras combined: **2,282** (2,260 pre-scope window + 22 pilot)
- Measured tokens: **2,666,947** (2,658,849 + 8,098)
- Unknown-cost attempts: **3**, plus any inside the pre-scope window already counted
- Cache hits: **1,538**
