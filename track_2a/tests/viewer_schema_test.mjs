// Node test for viewer/schema.js — exercises BOTH v2 (diff/span) and v3
// (tag ids) judgment shapes plus failure/empty edge cases.
import { createRequire } from "module";
const require = createRequire(import.meta.url);
const S = require("../viewer/schema.js");

let failures = 0;
function eq(got, want, name) {
  const g = JSON.stringify(got), w = JSON.stringify(want);
  if (g === w) { console.log(`ok   ${name}`); }
  else { failures++; console.log(`FAIL ${name}: got ${g} want ${w}`); }
}

// --- v3-tag shapes ---
eq(S.judgmentIsV3({ a_ids: [0], b_ids: [1], ok: true }), true, "v3 detected");
eq(S.judgmentIsV3({ a_ids: [], b_ids: [], ok: true }), true, "v3 empty lists detected");
eq(S.judgmentIsV3({ difference: 3, differing_spans: {}, ok: true }), false, "v2 not v3");
eq(S.judgmentIsV3(null), false, "null not v3");

eq(S.judgeBadge({ a_ids: [0, 2], b_ids: [1], ok: true }),
   '<span class="diff-badge">tags a:2 b:1</span>', "v3 ok badge");
eq(S.judgeBadge({ a_ids: [], b_ids: [], ok: true }),
   '<span class="diff-badge">tags a:0 b:0</span>', "v3 empty ok badge");
eq(S.judgeBadge({ a_ids: undefined, b_ids: [], ok: false }),
   '<span class="diff-badge">judge failed → neutral fallback</span>',
   "a_ids absent => not v3; failed judgment badge");
eq(S.judgeBadge({ a_ids: [0], b_ids: [], ok: false }),
   '<span class="diff-badge">judge invalid → score-0 fallback</span>',
   "v3 invalid badge (score-0, not 'emitted')");

// --- v2 diff/span shapes ---
eq(S.judgeBadge({ difference: 3, differing_spans: { side_a: [0, 1], side_b: [2] }, ok: true }),
   '<span class="diff-badge">diff=3 · spans a:2 b:1</span>', "v2 ok badge (differing_spans)");
eq(S.judgeBadge({ difference: 0, spans_a: [0], spans_b: [], ok: true }),
   '<span class="diff-badge">diff=0 · spans a:1 b:0</span>', "v2 ok badge (legacy spans_*)");
eq(S.judgeBadge({ difference: 5, ok: false }),
   '<span class="diff-badge">judge failed → neutral fallback</span>', "v2 failed badge");

// --- fallthrough ---
eq(S.judgeBadge({ ok: false }),
   '<span class="diff-badge">judge failed → neutral fallback</span>', "bare failed judgment");
eq(S.judgeBadge({ ok: true }), "", "bare ok judgment => no badge");
eq(S.judgeBadge(null), "", "null judgment => no badge");

// --- item-level detection ---
eq(S.itemIsV3({ judgments: [{ difference: 1, ok: true }] }, {}), false,
   "v2-only item");
eq(S.itemIsV3({ judgments: [{ a_ids: [], b_ids: [], ok: true }] }, {}), true,
   "v3 judgment marks item");
eq(S.itemIsV3({ judgments: [] }, { flagged_a: 0 }), true,
   "v3 stats mark item");
eq(S.itemIsV3({ judgments: [] }, { span_matched: 3 }), false,
   "v2 stats mark item v2");
eq(S.itemIsV3({ judgments: [{ ok: false }] }, {}), false,
   "failed-only judgments => v2 display");

process.exit(failures ? 1 : 0);
