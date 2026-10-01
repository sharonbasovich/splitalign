"""Official-metric evaluation, isolated from the upstream repo.

Uses the vendored ``evaluation`` package (byte-identical to
ZurichNLP/SwissGov-RSD @ 1807a42) for loading gold data and token-label
parsing, plus ``nlpstats`` for Spearman/Kendall exactly as
``scripts/evaluate_predictions_admin.py`` does. We re-implement only the
glue (per-language aggregation, length-mismatch counting, -1 filtering)
with identical semantics — proven by tests/test_eval_parity.py.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

VENDOR_DIR = Path(__file__).resolve().parents[1] / "vendor" / "swissgov_rsd"
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

from evaluation.utils import load_gold_data  # noqa: E402  (vendored upstream)
from nlpstats.correlations import bootstrap, correlate  # noqa: E402

from . import guard  # noqa: E402

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


def _correlate(pred_labels: list[float], gold_labels: list[float],
               n_resamples: int) -> tuple[float, float, float, float]:
    fp = [p for p, g in zip(pred_labels, gold_labels) if g != -1]
    fg = [g for g in gold_labels if g != -1]
    pa = np.expand_dims(np.array(fp), 0)
    ga = np.expand_dims(np.array(fg), 0)
    spear = correlate(pa, ga, level="global", coefficient="spearman")
    kend = correlate(pa, ga, level="global", coefficient="kendall")
    lo = hi = spear
    if n_resamples > 0:
        b = bootstrap(pa, ga, level="global", coefficient="spearman",
                      resampling_method="inputs", n_resamples=n_resamples)
        lo, hi = float(b.lower), float(b.upper)
    return float(spear), lo, hi, float(kend)


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
    if drop_ids:
        keep = [g["id"] not in drop_ids for g in gold_items]
        gold_items = [g for g, k in zip(gold_items, keep) if k]
        gold_samples = [g for g, k in zip(gold_samples, keep) if k]

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
    return LangResult(lang=lang, spearman=spear, spearman_lo=lo,
                      spearman_hi=hi, kendall=kend,
                      n_tokens=len(gold_labels), n_samples=used,
                      len_mismatch_a=mm_a, len_mismatch_b=mm_b)


def evaluate_split(pred_dir: Path, split: str, langs=("de", "fr", "it"),
                   n_resamples: int = 1000,
                   prefix: str | None = None) -> dict:
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
    macro = float(np.mean([r.spearman for r in results.values()])) if results else float("nan")
    macro_excl = (float(np.mean([r.spearman for r in results_excl.values()]))
                  if results_excl else float("nan"))
    return {"split": norm,
            "per_language": {k: vars(v) for k, v in results.items()},
            "macro_spearman": macro,
            # official LLM scoring also drops list_to_drop.txt ids; report the
            # same metric excluding the dev subset of those ids for parity.
            "per_language_excl_dev_drop": {k: vars(v) for k, v in results_excl.items()},
            "macro_spearman_excl_dev_drop": macro_excl,
            "dev_drop_ids": sorted(DEV_DROP_IDS)}
