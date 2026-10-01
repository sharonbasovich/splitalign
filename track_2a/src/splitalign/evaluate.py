"""Official-metric evaluation, isolated from the upstream repo.

Implements the official token-label scoring semantics — id-matched
predictions, -1 gold filtering, length-mismatch pad/truncate, all -1
labels_b skip, nlpstats global Spearman/Kendall and bootstrap — entirely
with original code (see metricspec.py); no upstream source is
redistributed. Proven equal to an independent scipy computation by
tests/test_eval_parity.py.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from nlpstats.correlations import bootstrap, correlate

from . import guard
from .metricspec import load_gold_data

# The official LLM scoring drops the ids in upstream ``list_to_drop.txt``
# (applied only to LLM-format predictions). The subset of those ids that
# lives inside the dev firewall is enumerated here — dev ids only, never
# fetched from or joined to any held-out file. Verified present in
# data/gold/dev/{train,val}: suffixes 18,100,106,153,196 x {de,fr,it}.
DEV_DROP_IDS = frozenset(
    f"admin_{lang}_{n}"
    for lang in ("de", "fr", "it")
    for n in (18, 100, 106, 153, 196))


@dataclass
class LangResult:
    lang: str
    spearman: float
    spearman_lo: float
    spearman_hi: float
    kendall: float
    n_tokens: int
    n_samples: int
    len_mismatch_a: int = 0
    len_mismatch_b: int = 0
    n_gold_items: int = 0
    coverage: float = 0.0
    missing_ids: list = field(default_factory=list)


def _correlate(pred_labels: list[float], gold_labels: list[float],
               n_resamples: int) -> tuple[float, float, float, float]:
    fp = [p for p, g in zip(pred_labels, gold_labels) if g != -1]
    fg = [g for g in gold_labels if g != -1]
    pa = np.expand_dims(np.array(fp), 0)
    ga = np.expand_dims(np.array(fg), 0)
    spear = correlate(pa, ga, level="global", coefficient="spearman")
    kend = correlate(pa, ga, level="global", coefficient="kendall")
    if n_resamples > 0:
        b = bootstrap(pa, ga, level="global", coefficient="spearman",
                      resampling_method="inputs", n_resamples=n_resamples)
        return float(spear), float(b.lower), float(b.upper), float(kend)
    # bootstrap 0: no interval was computed — callers must omit the bounds
    return float(spear), None, None, float(kend)


def _invalid_reason(r: LangResult) -> str | None:
    """Why a correlation is undefined — recorded truthfully, never hidden."""
    if not (r.spearman != r.spearman):  # not NaN
        return None
    if r.n_samples == 0:
        return "no matched predictions"
    return "undefined correlation (constant or degenerate output — e.g. an " \
           "all-one-label prediction gives no variance to correlate)"


def evaluate_predictions(pred_records: list[dict], gold_path: Path,
                         lang: str, n_resamples: int = 1000,
                         drop_ids: frozenset | None = None) -> LangResult:
    """One language: encoder-format predictions vs gold, official semantics."""
    gold_samples = load_gold_data(guard.guard_data_dir(gold_path))
    by_id = {r["id"]: r for r in pred_records}
    # The official script zips predictions and gold samples in file order;
    # we match on id which is equivalent and safer against reordering.
    gold_items = [json.loads(line) for line in
                  guard.guard_data_dir(gold_path).read_text().splitlines() if line.strip()]
    guard.assert_ids_allowed([r["id"] for r in gold_items])
    missing = [g["id"] for g in gold_items if g["id"] not in by_id]
    if drop_ids:
        keep = [g["id"] not in drop_ids for g in gold_items]
        gold_items = [g for g, k in zip(gold_items, keep) if k]
        gold_samples = [g for g, k in zip(gold_samples, keep) if k]
    n_gold = len(gold_items)

    pred_labels: list[float] = []
    gold_labels: list[float] = []
    mm_a = mm_b = 0
    used = 0
    for gold_sample, gold_item in zip(gold_samples, gold_items):
        rec = by_id.get(gold_item["id"])
        if rec is None:
            continue
        used += 1
        pa = list(rec["labels_a"])
        ga = list(gold_sample.labels_a)
        if len(pa) < len(ga):
            pa = pa + [0.0] * (len(ga) - len(pa))
            mm_a += 1
        elif len(pa) > len(ga):
            pa = pa[:len(ga)]
            mm_a += 1
        pred_labels.extend(pa)
        gold_labels.extend(ga)

        gb = list(gold_sample.labels_b)
        if not all(v == -1 for v in gb):
            pb = list(rec["labels_b"])
            if len(pb) < len(gb):
                pb = pb + [0.0] * (len(gb) - len(pb))
                mm_b += 1
            elif len(pb) > len(gb):
                pb = pb[:len(gb)]
                mm_b += 1
            pred_labels.extend(pb)
            gold_labels.extend(gb)

    spear, lo, hi, kend = _correlate(pred_labels, gold_labels, n_resamples)
    res = LangResult(lang=lang, spearman=spear, spearman_lo=lo,
                     spearman_hi=hi, kendall=kend,
                     n_tokens=len(gold_labels), n_samples=used,
                     len_mismatch_a=mm_a, len_mismatch_b=mm_b)
    res.n_gold_items = n_gold
    res.coverage = used / n_gold if n_gold else 0.0
    res.missing_ids = missing
    return res


def _finite_or_reason(v, reason):
    """JSON-safe float: nonfinite -> null + explicit reason."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None, reason
    return float(v), None


def _dump(r: LangResult) -> dict:
    # omit CI bounds entirely when no bootstrap was computed (never display
    # lo == hi == point as if it were a confidence interval); serialize
    # nonfinite metrics as null with an explicit reason — standards-safe JSON
    reason = _invalid_reason(r)
    out = {k: v for k, v in vars(r).items() if v is not None}
    for k in ("spearman", "spearman_lo", "spearman_hi", "kendall"):
        if k in out:
            v, _ = _finite_or_reason(out[k], reason)
            out[k] = v  # nonfinite -> explicit null, never bare NaN/Infinity
    if reason:
        out["invalid_reason"] = reason
    return out


def evaluate_split(pred_dir: Path, split: str, langs=("de", "fr", "it"),
                   n_resamples: int = 1000,
                   prefix: str | None = None,
                   require_full_coverage: bool = False) -> dict:
    """Evaluate a prediction-set directory over all languages.

    ``split`` is firewall-checked: only dev/train and dev/val are legal.
    ``prefix`` selects a single prediction file ``{prefix}_admin_{lang}.jsonl``;
    without it exactly one file per language must match.
    """
    norm = guard.normalize_split(split)
    results = {}
    results_excl = {}
    for lang in langs:
        gold = guard.gold_path(norm, lang)
        if prefix:
            cand = [Path(pred_dir) / f"{prefix}_admin_{lang}.jsonl"]
            cand = [c for c in cand if c.exists()]
        else:
            cand = sorted(Path(pred_dir).glob(f"*admin_{lang}.jsonl"))
        if not cand:
            continue
        if len(cand) > 1:
            raise SystemExit(
                f"ambiguous predictions for {lang}: {[c.name for c in cand]}; "
                "pass --prefix")
        recs = [json.loads(l) for l in cand[0].read_text().splitlines() if l.strip()]
        results[lang] = evaluate_predictions(recs, gold, lang, n_resamples)
        results_excl[lang] = evaluate_predictions(recs, gold, lang, n_resamples,
                                                  drop_ids=DEV_DROP_IDS)
    if require_full_coverage:
        gaps = {k: r.missing_ids for k, r in results.items() if r.missing_ids}
        if gaps:
            raise SystemExit(
                "incomplete prediction coverage: "
                + "; ".join(f"{k} missing {v[:5]}{'...' if len(v) > 5 else ''}"
                            for k, v in gaps.items()))
    def _macros(res_map):
        """Strict primary macro + clearly-labeled finite-language mean.

        macro_spearman is null/invalid when ANY required language's
        correlation is undefined (or the language is missing entirely) —
        constant/degenerate output is an informative failure, never a
        silently excluded success. The descriptive mean over defined
        languages is reported separately with its count and must not be
        used as the headline comparison metric.
        """
        required = set(res_map.keys())
        defined = {k: v.spearman for k, v in res_map.items()
                   if v.spearman == v.spearman}
        strict = (sum(defined.values()) / len(defined)
                  if len(defined) == len(required) else None)
        descr = sum(defined.values()) / len(defined) if defined else None
        return strict, descr, len(defined)
    macro, descr, n_def = _macros(results)
    macro_excl, descr_excl, n_def_excl = _macros(results_excl)
    n_missing = len(langs) - len(results)
    if n_missing:
        # a required language produced no predictions at all -> both invalid
        macro = None
        macro_excl = None
    return {"split": norm,
            "per_language": {k: _dump(v) for k, v in results.items()},
            "macro_spearman": macro,
            "macro_spearman_invalid_reason": (
                None if macro is not None else
                "undefined for >=1 required language (constant/degenerate "
                "output or missing predictions — see per_language)"),
            "descriptive_mean_finite_langs": descr,
            "descriptive_mean_langs": n_def,
            # official LLM scoring also drops list_to_drop.txt ids; report the
            # same metric excluding the dev subset of those ids for parity.
            "per_language_excl_dev_drop": {k: _dump(v) for k, v in results_excl.items()},
            "macro_spearman_excl_dev_drop": macro_excl,
            "macro_excl_invalid_reason": (
                None if macro_excl is not None else
                "undefined for >=1 required language — see per_language_excl_dev_drop"),
            "descriptive_mean_excl_dev_drop": descr_excl,
            "descriptive_mean_excl_dev_drop_langs": n_def_excl,
            "dev_drop_ids": sorted(DEV_DROP_IDS)}
