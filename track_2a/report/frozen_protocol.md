# Frozen inference protocol — v3-tag (splitalign-prompts-v3-tag)

Status: **frozen, NOT executed.** No real model/API calls are authorized by this
document. Both steps below remain gated on explicit owner approval; Step B in
particular is **not authorized** and its document IDs stay unopened.

## Hard rules (apply to every future real-inference proposal)

1. **Reserved conservative maxima, not estimates.** Every attempt and token cap
   stated here is a reserved maximum. Any *estimated* footprint must be labeled
   an estimate and may never be presented as a guaranteed hard total.
2. **Unknown provider usage stops the run.** Token totals count only `usage`
   fields returned by the provider. Any outbound attempt whose usage is unknown
   (failed response, missing usage fields) counts as an attempted request AND
   halts the run with `attempts_no_usage` recorded — it is never silently
   treated as zero-cost and never folded into a "guaranteed" total.
3. Accounting is at the **outbound-attempt boundary**: every HTTP attempt —
   initial calls, retries, failures — counts against the attempt cap.
   Cache hits are free and counted separately.
4. Held-out/quarantined/test splits remain fail-closed. Calibration, tuning,
   and prompt iteration on new dev outcomes are not part of either step.

## Step A — disclosed pilot (proposed, awaiting approval)

- Source: post-v3 code at the draft-PR head (v3-tag judge prompt).
- Documents: `admin_de_17`, `admin_fr_17`, `admin_it_17` — the already-disclosed
  pilot IDs (no new document exposure).
- Methods: splitalign (v3-tag) and baseline (`baseline-v1-repair-exhausted`,
  the repair-exhausted `fallback_label=5` whole-document policy — a reference
  fallback, **not** a strong comparator).
- Budget: reviewer-suggested **≤40 outbound attempts and ≤30,000 tokens**
  shared across both methods — reserved conservative maxima.
- Proposed CLI (not executed):
  `SPLITALIGN_BACKEND=apertus python -m splitalign.run pipeline --split val
   --lang all --limit 1 --offset 0 --bootstrap 0
   --max-requests 40 --max-tokens 30000`
- **Pass gates (primary):**
  - `judge_valid_token_coverage >= 0.90` (fraction of matched-pair tokens
    scored under a *valid* judge_tag judgment; score-0 fallback tokens are
    counted and excluded — never called emitted/valid coverage).
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
- Primary-method gate (explicit, required before Step B is proposed):
  **judge_valid_token_coverage ≥ 0.90** on Step A — stated here as the
  primary gate, not only as a falsification note.
- Any Step B proposal must restate reserved maxima vs estimates per rule 1.

## Step-A-failure fallback

If Step A fails its gates: report only the actual Step-A artifacts (coverage,
invalid-pair counts, per-ID judgments), state which gates failed, and stop.
No inference about unrun documents; no claims beyond the observed evidence.
