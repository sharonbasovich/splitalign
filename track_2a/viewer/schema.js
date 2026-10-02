/* Schema-aware helpers shared by the viewer and its node tests.
   v3-tag judgments carry a_ids/b_ids; v2 judgments carry a 0-5
   `difference` score plus differing_spans lists. Detection keys on the
   FIELDS PRESENT, never on a version string, so mixed/historical runs
   render correctly. */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.SplitAlignSchema = api;
})(typeof self !== "undefined" ? self : globalThis, function () {
  "use strict";

  function judgmentIsV3(j) { return !!(j && j.a_ids !== undefined); }
  function judgmentIsV2(j) { return !!(j && j.difference !== undefined); }

  function itemIsV3(it, st) {
    return (it.judgments || []).some(judgmentIsV3) ||
           !!(st && (st.flagged_a !== undefined ||
                     st.invalid_pairs !== undefined));
  }

  // Returns the badge HTML for one judgment, or "" when nothing applies.
  function judgeBadge(j) {
    if (judgmentIsV3(j)) {
      return j.ok
        ? `<span class="diff-badge">tags a:${(j.a_ids || []).length} b:${(j.b_ids || []).length}</span>`
        : `<span class="diff-badge">judge invalid → score-0 fallback</span>`;
    }
    if (judgmentIsV2(j)) {
      const sa = (j.differing_spans && j.differing_spans.side_a) || j.spans_a || [];
      const sb = (j.differing_spans && j.differing_spans.side_b) || j.spans_b || [];
      return j.ok
        ? `<span class="diff-badge">diff=${j.difference} · spans a:${sa.length} b:${sb.length}</span>`
        : `<span class="diff-badge">judge failed → neutral fallback</span>`;
    }
    if (j && !j.ok) {
      return `<span class="diff-badge">judge failed → neutral fallback</span>`;
    }
    return "";
  }

  return { judgmentIsV3, judgmentIsV2, itemIsV3, judgeBadge };
});
