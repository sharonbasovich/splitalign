# Cumulative cost ledger — all historical actual runs

Sanitized (no secrets). **Reproduced deterministically by
`results/audit_costs.py`** (reads the immutable logs, classifies each row into
exactly one bucket — no double counting). Source files:

- `results/calls.jsonl` — legacy pre-run-scope log, **3,788 raw / 3,725 distinct**
  records, all `run_id: null`.
- `results/runs/20261002T082810Z-76c9485d/calls.jsonl` — pilot run log, 32 rows,
  **zero exact-row overlap with the legacy log**.

**ERRATUM (v1 of this ledger):** the first version summed token figures across
all noncached records and thereby included **mock-backend ESTIMATED tokens
(1,002,496) as if they were real spend**. Corrected below: only
`backend=="apertus"` noncached successful records with provider `usage` count
as measured real spend; mock records are excluded and listed separately.

**Logical records ≠ outbound HTTP attempts.** Internal transport retries are
not logged per-attempt, so every noncached count below is a *floor* on actual
wire attempts, stated as such.

## Apertus — real provider usage (measured)

| source | successful logical records | measured tokens | window |
|---|---|---|---|
| legacy `calls.jsonl` | 1,243 (all distinct) | 1,295,021 prompt + 361,332 completion = **1,656,353** | 2026-10-01T19:52:37Z – 21:24:51Z |
| pilot run `76c9485d` | 22 | 7,127 + 971 = **8,098** | 2026-10-02T08:28:11Z – 08:28:38Z |
| **combined** | **1,265** | **1,664,451** | |

## Apertus — attempts with UNKNOWN token cost (never counted as zero)

| ts | kind | status |
|---|---|---|
| 2026-10-01T21:24:35Z | doc_baseline | call failed — cost unknown |

## Unknown-backend failures (cost and backend both unknown)

| ts | kind | status |
|---|---|---|
| 2026-10-01T20:11:17Z | doc_baseline | call failed |
| 2026-10-01T21:10:19Z | doc_baseline | call failed |

## Mock backend — ESTIMATED usage, excluded from real spend

1,014 noncached records, all `usage_estimated: true`
(818,985 + 183,511 = 1,002,496 *estimated* tokens — a deterministic local
estimate, never a provider charge). Window 19:45:36Z – 21:25:34Z.

## Cache hits (free replays — payload-identity matched)

| class | raw | distinct |
|---|---|---|
| apertus | 874 + 10 = 884 | 854 + 10 = 864 |
| mock | 99 | 72 |
| unknown backend | 555 | 539 |
| **combined** | **1,538** | **1,475** |

The 63 extra raw rows (63 exact-duplicate records: 20 apertus, 27 mock,
16 unknown) are **all cache hits** — legitimate same-second repeated reads,
NOT proven duplicate exports. Raw and distinct are both reported.

## Aborted / non-inference runs

| run | status |
|---|---|
| 20261002T082807Z-06698fcf | manifest only — aborted before any call (missing env config) |
| 20261001T212533Z-1361c568 | mock backend only; no real API cost |
| 20261001T211451Z-8de1aab7 | **bug-affected** (FR/IT judged with German prompts); no run-scoped call log — usage is inside the legacy window above; see `erratum_sidecar.json` |
| legacy-pre-runscope/ | artifacts only, no call records — usage unknown, not zero |

## Totals

- Successful Apertus logical records (measured): **1,265** / **1,664,451 tokens**
- Apertus + unknown-backend failed records with unknown cost: **3** (never
  counted as zero; wire attempts may exceed this — retries unlogged)
- Cache hits: **1,538 raw / 1,475 distinct**
- Mock estimated records (excluded): **1,014**
