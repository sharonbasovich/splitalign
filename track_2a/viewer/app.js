/* SplitAlign evidence viewer — static, reads window.EVIDENCE from evidence.js */
(function () {
  "use strict";
  const E = window.EVIDENCE;
  const $ = (id) => document.getElementById(id);

  function heat(v) {
    if (v === null || v === undefined || v === -1) return "transparent";
    const t = Math.max(0, Math.min(1, v));
    // green -> yellow -> red
    const r = Math.round(255 * Math.min(1, 2 * t));
    const g = Math.round(255 * Math.min(1, 2 * (1 - t)));
    return `rgba(${r},${g},80,${0.25 + 0.45 * t})`;
  }
  function isPunct(t) { return /^[\p{P}\p{S}]+$/u.test(t); }
  function esc(s) {
    return s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  }

  const modes = Object.keys(E.items_by_mode || {});
  let state = { mode: modes.includes("splitalign") ? "splitalign" : modes[0],
                item: 0, pair: 0, labels: "pred" };

  function items() { return (E.items_by_mode[state.mode] || []); }
  function item() { return items()[state.item]; }

  function init() {
    const banner = $("banner");
    banner.className = "banner " + (E.mock ? "mock" : "real");
    const inf = E.inference || {};
    banner.textContent = E.mock
      ? "MOCK BACKEND — deterministic heuristics, NOT Apertus output. Plumbing demo only."
      : `Apertus backend — model: ${E.model || "unknown (not recorded)"} · inference prompt ${inf.prompt_version || E.prompt_version || "unknown (not recorded)"}` +
        (E.exporter && E.exporter.prompt_version && E.exporter.prompt_version !== (inf.prompt_version || E.prompt_version)
          ? ` (exported by ${E.exporter.prompt_version} code — NOT re-prompted)` : "");
    renderRunStatus();

    const msel = $("modeSel");
    modes.forEach((m) => msel.add(new Option(m, m)));
    msel.value = state.mode;  // keep selector in sync with the initial mode
    msel.addEventListener("change", () => { state.mode = msel.value; state.item = 0; state.pair = 0; render(); });

    $("labelSel").addEventListener("change", (e) => { state.labels = e.target.value; render(); });
    $("itemSel").addEventListener("change", (e) => { state.item = +e.target.value; state.pair = 0; render(); });

    const lim = $("limList");
    (E.limitations || []).forEach((l) => { const li = document.createElement("li"); li.textContent = l; lim.appendChild(li); });
    $("prov").textContent = JSON.stringify(
      { generated_utc: E.generated_utc,
        inference: E.inference || { splitalign_version: E.splitalign_version, prompt_version: E.prompt_version, model: E.model, note: "no separate inference block (older evidence file)" },
        exporter: E.exporter || null,
        backend: E.backend, split: E.split, run: E.run || null }, null, 2);

    document.addEventListener("keydown", (e) => {
      if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
      const its = items();
      if (!its.length) return;
      if (e.key === "ArrowRight") { state.item = Math.min(its.length - 1, state.item + 1); state.pair = 0; }
      else if (e.key === "ArrowLeft") { state.item = Math.max(0, state.item - 1); state.pair = 0; }
      else if (e.key === "ArrowDown") { const it = item(); state.pair = Math.min(Math.max(0, ((it && it.ops) || []).length - 1), state.pair + 1); }
      else if (e.key === "ArrowUp") { state.pair = Math.max(0, state.pair - 1); }
      else if (e.key === "g") { state.labels = "gold"; $("labelSel").value = "gold"; }
      else if (e.key === "p") { state.labels = "pred"; $("labelSel").value = "pred"; }
      else return;
      e.preventDefault(); render();
    });
    render();
  }

  function fmtCov(mode) {
    const m = E.run.modes[mode];
    const langs = Object.keys(m.intended);
    const per = langs.map((l) => `${l} ${(m.completed || {})[l] || 0}/${m.intended[l]} of ${E.run.n_split[l]}`).join(", ");
    return `${mode}: ${m.n_completed === undefined ? 0 : m.n_completed}/${m.n_intended} intended docs (${per})` +
      (m.budget_capped ? ` — CAPPED: ${m.budget_capped}` : "") +
      (!m.produced ? " — NOT PRODUCED in this run" : "") +
      (m.summary_present ? "" : " — no run summary (did not finish)");
  }

  function renderRunStatus() {
    const box = $("runStatus");
    if (!box) return;
    if (!E.run) {
      box.className = "runstatus partial";
      box.textContent = "No run scope recorded in this evidence file — coverage and comparability unknown.";
      return;
    }
    const R = E.run;
    const lines = [`Run ${R.id || R.run_id} (${R.started_utc || "?"})` +
      (R.source && R.source.commit ? ` · source ${String(R.source.commit).slice(0, 12)}${R.source.dirty ? " (DIRTY)" : ""}` : " · source commit not recorded")];
    Object.keys(R.modes).forEach((m) => lines.push(fmtCov(m)));
    if (R.matched && Object.keys(R.matched.n_matched || {}).length) {
      const mm = R.matched.n_matched;
      lines.push("matched-ID comparison on " + Object.keys(mm).map((l) => `${l} ${mm[l]}`).join(", ") +
        " docs" + (R.matched.macro_matched ? ` · macro splitalign ${fmt(R.matched.macro_matched.splitalign)} vs baseline ${fmt(R.matched.macro_matched.baseline)}` : ""));
    } else {
      lines.push("No matched-ID comparison in this run" +
        (R.modes_intended_not_produced.length ? ` (${R.modes_intended_not_produced.join(", ")} intended but not produced)` : "") + ".");
    }
    if (R.call_log) lines.push(`Call log: ${R.call_log}.`);
    (R.known_defects || []).forEach((d) => lines.push(`KNOWN CONFIGURATION DEFECT — ${d}`));
    if (R.partial) lines.unshift("PARTIAL / INCOMPLETE RUN — numbers below cover only the listed documents and support no complete comparison.");
    box.className = "runstatus" + ((R.partial || (R.known_defects || []).length) ? " partial" : "");
    box.textContent = lines.join("\n");
  }

  function fmt(v) { return (v === null || v === undefined || Number.isNaN(v)) ? "undefined" : Number(v).toFixed(3); }

  function renderItemSelector() {
    const sel = $("itemSel");
    sel.innerHTML = "";
    items().forEach((it, i) => sel.add(new Option(`${it.id} (${it.lang})`, i)));
    sel.value = state.item;
  }

  function hasText(it) { return typeof it.text_a === "string" && typeof it.text_b === "string"; }

  function renderMeta() {
    const it = item();
    if (!it) { $("meta").textContent = `No items in evidence for mode "${state.mode}".`; $("evalSummary").textContent = ""; return; }
    if (it.failed || !hasText(it)) {
      $("evalSummary").textContent = "";
      $("meta").textContent = `${it.id} — ${it.lang || "?"} · ` +
        (it.failed ? `FAILED: ${it.error || "backend error"} — no prediction for this item`
                   : "no document text recorded for this item — nothing to render");
      return;
    }
    const ev = (E.evaluation || {})[state.mode];
    let es = "";
    if (ev && ev.per_language && ev.per_language[it.lang]) {
      const r = ev.per_language[it.lang];
      es = ` · ${it.lang} Spearman ${fmt(r.spearman)}` +
           (r.invalid_reason ? ` (${r.invalid_reason})` : "") +
           (r.n_samples !== undefined ? ` on ${r.n_samples} docs` : "") +
           (E.mock ? " (mock — not a model result)" : "");
      if (ev.macro_spearman !== undefined)
        es += ` · strict macro ${fmt(ev.macro_spearman)}` +
              (ev.macro_spearman === null && ev.macro_spearman_invalid_reason ? ` (${ev.macro_spearman_invalid_reason})` : "");
    }
    $("evalSummary").textContent = es;
    const st = it.stats || {};
    $("meta").textContent =
      `${it.id} — ${it.lang} · ops: ${(it.ops || []).length}` +
      ` · repairs: ${st.repairs || 0} · parse failures: ${st.parse_failures || 0}` +
      ` · spans matched/unmatched: ${st.span_matched || 0}/${st.span_unmatched || 0}` +
      ` · cached judgments: ${(it.judgments || []).filter(j => j && j.cached).length}`;
  }

  function renderPairs() {
    const it = item();
    const box = $("alignView");
    box.innerHTML = "";
    if (!it) return;
    if (it.failed) {
      box.innerHTML = `<div class="pair"><div class="cell" style="grid-column:1/4"><em>Item failed: ${esc(it.error || "backend error")} (no prediction)</em></div></div>`;
      return;
    }
    if (!it.ops) {
      box.innerHTML = `<div class="pair"><div class="cell" style="grid-column:1/4"><em>Whole-document baseline — no alignment operations.</em></div></div>`;
      return;
    }
    const segsA = it.segments_a || [], segsB = it.segments_b || [];
    if (!it.ops.length) {
      box.innerHTML = `<div class="pair"><div class="cell" style="grid-column:1/4"><em>No alignment operations recorded.</em></div></div>`;
      return;
    }
    it.ops.forEach((op, k) => {
      const div = document.createElement("div");
      div.className = "pair" + (k === state.pair ? " active" : "");
      const j = (it.judgments || [])[k];
      const aTxt = op.a_start >= 0 ? segsA.slice(op.a_start, op.a_end).map(s => s.text).join(" ") : "";
      const bTxt = op.b_start >= 0 ? segsB.slice(op.b_start, op.b_end).map(s => s.text).join(" ") : "";
      const badge = j && j.ok ? `<span class="diff-badge">diff ${j.difference}/5</span>` :
                    j && !j.ok ? `<span class="diff-badge">judge failed → fallback</span>` : "";
      div.innerHTML =
        `<span class="op-tag ${op.op === "1:1" ? "" : "asym"}">${op.op}</span>` +
        `<div class="cell">${esc(aTxt) || "<em>—</em>"}</div>` +
        `<div class="cell">${esc(bTxt) || "<em>—</em>"}${badge}</div>`;
      div.addEventListener("click", () => { state.pair = k; render(); });
      box.appendChild(div);
    });
    if (state.pair >= it.ops.length) state.pair = it.ops.length - 1;
  }

  function pairTokenSets(it) {
    // tokens covered by the currently selected op (for highlight)
    const op = (it.ops || [])[state.pair];
    const sA = new Set(), sB = new Set();
    if (!op) return { sA, sB };
    (it.segments_a || []).slice(Math.max(0, op.a_start), Math.max(0, op.a_end))
      .forEach(s => { for (let t = s.start; t < s.end; t++) sA.add(t); });
    (it.segments_b || []).slice(Math.max(0, op.b_start), Math.max(0, op.b_end))
      .forEach(s => { for (let t = s.start; t < s.end; t++) sB.add(t); });
    return { sA, sB };
  }

  function labelFor(labels, gold, i) {
    if (state.labels === "pred") return labels[i];
    if (state.labels === "gold") return gold ? gold[i] : null;
    if (labels[i] === -1 || !gold || gold[i] === -1) return -1;
    return Math.abs(labels[i] - gold[i]);
  }

  function renderDoc() {
    const it = item();
    const elA = $("tokA"), elB = $("tokB");
    elA.innerHTML = ""; elB.innerHTML = "";
    $("hB").textContent = it ? ({ de: "German", fr: "French", it: "Italian" }[it.lang] || it.lang || "Other") : "Other";
    if (!it) {
      elA.innerHTML = "<em class='errstate'>No items in this mode.</em>";
      elB.innerHTML = "<em class='errstate'>No items in this mode.</em>";
      return;
    }
    if (it.failed || !hasText(it)) {
      const msg = it.failed ? `Item failed: ${esc(it.error || "backend error")} (no prediction, no tokens to show)`
                            : "No document text recorded for this item.";
      elA.innerHTML = `<em class="errstate">${msg}</em>`;
      elB.innerHTML = `<em class="errstate">${msg}</em>`;
      return;
    }
    const { sA, sB } = pairTokenSets(it);
    const toksA = it.text_a.split(/\s+/), toksB = it.text_b.split(/\s+/);
    const la = it.labels_a || [], lb = it.labels_b || [];
    const ga = it.gold_labels_a || [], gb = it.gold_labels_b || [];
    const op = (it.ops || [])[state.pair] || {};

    function emit(el, toks, labels, golds, hlSet, asymFlag) {
      toks.forEach((t, i) => {
        const sp = document.createElement("span");
        sp.className = "tok";
        sp.tabIndex = -1;
        const v = labelFor(labels, golds, i);
        if (v === -1) sp.classList.add("punct");
        sp.style.background = heat(v);
        if (hlSet.has(i)) sp.style.outline = "2px solid #42a5f5";
        if (asymFlag === "omit" && op.a_start >= 0 && i >= (it.segments_a[op.a_start]||{start:1e9}).start && i < (it.segments_a[op.a_end-1]||{end:-1}).end && op.b_start < 0) sp.classList.add("omitted");
        if (asymFlag === "add" && op.b_start >= 0 && i >= (it.segments_b[op.b_start]||{start:1e9}).start && i < (it.segments_b[op.b_end-1]||{end:-1}).end && op.a_start < 0) sp.classList.add("added");
        sp.textContent = t;
        sp.title = `label ${v === null ? "?" : v}`;
        el.appendChild(sp);
        el.appendChild(document.createTextNode(" "));
      });
    }
    emit(elA, toksA, la, ga, sA, "omit");
    emit(elB, toksB, lb, gb, sB, "add");
  }

  function render() {
    renderItemSelector();
    renderMeta();
    renderPairs();
    renderDoc();
  }

  if (!E || !E.items_by_mode) {
    document.body.innerHTML = "<p style='padding:2em'>No evidence.js found — run <code>splitalign.run export-viewer</code> first.</p>";
    return;
  }
  init();
})();
