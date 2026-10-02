# Frozen inference protocol — v3-tag (splitalign-prompts-v3-tag)

Status: **frozen, NOT executed.** No real model/API calls are authorized by this
document. Both steps below remain gated on explicit owner approval; Step B in
particular is **not authorized** and its document IDs stay unopened.

## Hard rules (apply to every future real-inference proposal)

1. **Reserved conservative maxima, not estimates.** Every attempt and token
   figure here is a reserved maximum enforced by atomic predispatch
   reservation: before any bytes leave, one request slot plus a *verified
   conservative upper bound* on that request (UTF-8-byte bound on serialized
   message contents + flat template allowance + bounded `max_tokens`
   completion) is claimed against the shared `ApiBudget`. A bound is a bound —
   an estimate may never be presented as a maximum or a guaranteed hard total.
   If no trustworthy provider/tokenizer bound is available, real inference is
   BLOCKED and the hard-cap claim explicitly disclaimed.
2. **Unknown usage halts the run.** An attempt's token cost counts only
   provider-reported `usage` validated as nonnegative true integers. Missing,
   malformed, or absent usage; HTTP errors; timeouts; unparseable response
   bodies; or missing `choices`/`content` all mean **unknown cost**: the
   attempt is recorded, its reservation stays committed (never released as
   "free"), and the run halts fail-closed immediately — **no retry is ever
   attempted after an unknown-cost attempt**. Internal logical calls log at
   the call boundary; wire-level accounting happens only at reservation.
3. Cache hits are free and counted separately — only when the full
   backend/model/prompt_version/kind/split/item/payload key matches.
4. Held-out/quarantined/test splits remain fail-closed. Calibration, tuning,
   and prompt iteration on new dev outcomes are not part of either step.
5. **Coverage metric definition (locked):**
   `judge_valid_token_coverage` = non-punctuation tokens covered by *valid*
   judge_tag judgments ÷ non-punctuation tokens covered by matched ops,
   aggregated **token-weighted across all documents** (sum of numerators ÷
   sum of denominators — never a mean of per-document ratios). Punctuation
   follows the official exclusion convention. Baseline "emitted coverage"
   (count of IDs emitted in the response lists) is a distinct quantity from
   valid mapped token coverage and is never conflated with it.

## Step A — disclosed pilot (proposed, awaiting approval)

- Source: post-v3 code at the draft-PR head (v3-tag judge prompt).
- Documents: `admin_de_17`, `admin_fr_17`, `admin_it_17` — the already-disclosed
  pilot IDs (no new document exposure).
- Methods: splitalign (v3-tag) and baseline (`baseline-v1-repair-exhausted`,
  the repair-exhausted `fallback_label=5` whole-document policy — a reference
  fallback, **not** a strong comparator).
- Budget: reviewer-suggested **≤40 outbound attempts and ≤30,000 committed
  tokens** shared across both methods — reserved conservative maxima.
- Proposed CLI (not executed):
  `SPLITALIGN_BACKEND=apertus python -m splitalign.run pipeline --split val
   --lang all --limit 1 --offset 0 --bootstrap 0
   --max-requests 40 --max-tokens 30000`
- **Pass gates (primary):**
  - `judge_valid_token_coverage >= 0.90` — locked definition in rule 5.
  - ≥90% of matched pairs judged valid (≤1 repair).
  - The two real DE clause-pair differences missed under v2 each receive ≥1
    flagged token on the correct side.
  - Zero unknown-usage attempts.
- Reporting is **descriptive only** (n=1 per language). No general claims of
  model impossibility if Step A fails; a failure report refers only to the
  available results — never to nonexistent 15-document metrics.

## Step B — fresh documents (NOT authorized)

- IDs `69, 74, 75, 90, 92` per language (`admin_{de,fr,it}_{69,74,75,90,92}`),
  15 documents total — **remain unopened and untouched** until separately
  released by the owner.
- **Primary-method gate, applied to Step B itself:** the Step B run must
  satisfy `judge_valid_token_coverage >= 0.90` (locked definition, rule 5)
  on its own outputs — not merely on Step A's. A Step B proposal also
  requires Step A to have met the same gate.
- Comparator set is preserved unchanged from the preregistered comparison:
  **EuroBERT-210m DiffAlign as the reference** plus the four named mmBERT-base
  variants (`parallel_en-de`, `parallel_en-sw`, `parallel_en-de-fr-it`,
  `parallel_all_1epoch`), on the **same fixed IDs**, with **seeded paired
  confidence intervals** and the **fixed decision rule** already in
  `technical_report.md`. No comparator or decision-rule changes.
- Any Step B proposal must restate reserved maxima vs estimates per rule 1.

## Step-A-failure fallback

If Step A fails its gates: report only the actual Step-A artifacts (coverage,
invalid-pair counts, per-ID judgments), state which gates failed, and stop.
No inference about unrun documents; no claims beyond the observed evidence.
