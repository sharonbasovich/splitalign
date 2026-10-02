# Incident record — fabricated planning cache entries (2026-10-02)

## What happened
During a READ-ONLY Step-A planning pass (no real provider calls), a stub
client used to enumerate intended requests returned a fabricated
`{"scores":[]}` response for three `pair_similarity` repair calls that were
cache misses. `Judge._invoke` cached those stub texts under real Apertus
cache keys in `results/cache/` (backend `apertus`, model
`swiss-ai/Apertus-v1.5-8B`).

## Fabricated records (cache-key hashes = filenames)
- `611819174d0127fe002979f07949d7a3ff7a05a08762eb9c6e0374f32f6085ae.json`
- `2a694f87228d84adc8b8fded5fe2addbe23608fbce410822acb3548122be293d.json`
- `adfff7fe1a00e560ae581cee4c85c10f3c70e02e1574b577bc4c5f710dae60fc.json`

All three were deleted on discovery; `exists=false` verified for each.

## Honesty status (IMPORTANT)
The claims that (a) these keys collided with no real cached response, (b)
the fabricated records were never served to a real run, and (c) the removal
is complete, are **builder-reported, not independently verified**. The
pilot run `76c9485d` predates the pollution and its committed artifacts are
untouched; the corrected cost ledger does not depend on these keys.

## Corrective measures
- All further dry planning uses isolated temporary/in-memory caches —
  zero writes to any real cache, and no fabricated provider responses.
- Proposed Step A uses a NEW empty run-specific cache via `SPLITALIGN_OUT`
  (fresh directory, asserted empty before any call, recorded in the run
  manifest) rather than reusing `results/cache`.
