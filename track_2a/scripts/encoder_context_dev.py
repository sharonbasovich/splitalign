"""Offline same-ID context against the published DEV encoder predictions.

No API calls, no model access. Fetches ONLY the exact public dev-split
prediction files of the pinned upstream commit (data/evaluation/
encoder_predictions/dev/), validates every prediction ID against the local
dev manifest BEFORE any evaluation, scores each published system on exactly
the IDs a SplitAlign run completed (same-ID, same gold, same metric code),
and reports aggregates plus a document-level paired bootstrap of the
strict-macro difference. Upstream prediction files are never written into
the repository — only aggregates, URLs and sha256 hashes are recorded.

Comparator policy (fixed before looking at numbers): the preregistered
reference is EuroBERT-210m; the mmBERT DiffAlign variants are reported as
context, all of them, never a cherry-picked subset.

Usage (PYTHONPATH=src):
  python scripts/encoder_context_dev.py --run results/runs/<id> \
      [--cache-dir /path/with/previously/fetched/files] [--resamples 2000]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from splitalign import guard  # noqa: E402
from splitalign.evaluate import _correlate  # noqa: E402
from splitalign.fetch_data import PINNED_COMMIT  # noqa: E402
from splitalign.metricspec import load_gold_data  # noqa: E402

TRACK = Path(__file__).resolve().parents[1]
UPSTREAM_DIR = "data/evaluation/encoder_predictions/dev"
LANGS = ("de", "fr", "it")
REFERENCE = "EuroBERT_EuroBERT-210m"
CONTEXT = [
    ".._ModernSimCSE_results_mmBERT-base-parallel_en-de",
    ".._ModernSimCSE_results_mmBERT-base-parallel_en-sw",
    ".._ModernSimCSE_results_mmBERT-base-parallel_en-de-fr-it",
    ".._ModernSimCSE_results_mmBERT-base-parallel_all_1epoch",
]


def _fname(variant: str, lang: str) -> str:
    return f"DiffAlign(model={variant}, layer=-1_admin_{lang}.jsonl.jsonl"


def _url(variant: str, lang: str) -> str:
    return (f"https://raw.githubusercontent.com/ZurichNLP/SwissGov-RSD/"
            f"{PINNED_COMMIT}/{UPSTREAM_DIR}/"
            + urllib.parse.quote(_fname(variant, lang), safe=""))


def _fetch(variant: str, lang: str, cache_dir: Path) -> tuple[Path, dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / _fname(variant, lang)
    url = _url(variant, lang)
    if not p.exists():
        with urllib.request.urlopen(url, timeout=60) as r:  # dev-only path
            p.write_bytes(r.read())
    data = p.read_bytes()
    return p, {"url": url, "sha256": hashlib.sha256(data).hexdigest(),
               "bytes": len(data)}


def _records(path: Path, manifest: set[str]) -> dict[str, dict]:
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    ids = [r["id"] for r in recs]
    guard.assert_ids_allowed(ids, manifest)   # every ID must be a dev ID
    assert len(set(ids)) == len(ids), f"duplicate ids in {path.name}"
    return {r["id"]: r for r in recs}


def _gold(split: str, lang: str) -> dict[str, object]:
    p = guard.gold_path(split, lang)
    items = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return {it["id"]: gs for gs, it in zip(load_gold_data(p), items)}


def _doc_arrays(rec: dict, gs) -> tuple[np.ndarray, np.ndarray]:
    """Official per-document convention: pad/truncate, skip all -1 side."""
    pl, gl = [], []
    ga = list(gs.labels_a)
    pa = (list(rec["labels_a"]) + [0.0] * len(ga))[:len(ga)]
    pl += pa; gl += ga
    gb = list(gs.labels_b)
    if not all(v == -1 for v in gb):
        pb = (list(rec["labels_b"]) + [0.0] * len(gb))[:len(gb)]
        pl += pb; gl += gb
    keep = np.array(gl) != -1
    return np.array(pl, float)[keep], np.array(gl, float)[keep]


def _pooled(docs: dict[str, tuple], ids, n_res: int) -> dict:
    pl = np.concatenate([docs[i][0] for i in ids]) if ids else np.array([])
    gl = np.concatenate([docs[i][1] for i in ids]) if ids else np.array([])
    sp, lo, hi, kd = _correlate(list(pl), list(gl), n_res)
    f = lambda v: None if v is None or v != v else round(float(v), 4)  # noqa
    return {"spearman": f(sp), "spearman_lo": f(lo), "spearman_hi": f(hi),
            "kendall": f(kd), "n_docs": len(ids), "n_tokens": int(len(gl))}


def _strict_macro(per_lang: dict) -> float | None:
    vals = [per_lang[l]["spearman"] for l in LANGS]
    return None if any(v is None for v in vals) else round(sum(vals) / 3, 4)


def _macro_on(docsets, sample) -> float:
    vals = []
    for l in LANGS:
        p = np.concatenate([docsets[l][i][0] for i in sample[l]])
        g = np.concatenate([docsets[l][i][1] for i in sample[l]])
        vals.append(spearmanr(p, g).correlation)
    return float(np.mean(vals))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run scope dir")
    ap.add_argument("--backend", default="apertus")
    ap.add_argument("--mode", default="splitalign")
    ap.add_argument("--cache-dir", default=str(TRACK / "results" / "cache"
                                               / "encoder_dev"))
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run_dir = Path(a.run)
    man = json.loads((run_dir / "manifest.json").read_text())
    split = guard.normalize_split(man["split"])
    summ = json.loads((run_dir / f"run_summary_{a.mode}_{a.backend}_"
                       f"{split.replace('/', '_')}.json").read_text())
    manifest = guard.load_manifest()
    ids = {l: sorted(summ["completed_ids"].get(l, [])) for l in LANGS}
    gold = {l: _gold(split, l) for l in LANGS}
    guard.assert_ids_allowed([i for v in ids.values() for i in v], manifest)

    def docset(recs_by_lang):
        return {l: {i: _doc_arrays(recs_by_lang[l][i], gold[l][i])
                    for i in ids[l]} for l in LANGS}

    run_recs = {}
    for l in LANGS:
        p = run_dir / f"{a.mode}_{a.backend}_admin_{l}.jsonl"
        run_recs[l] = _records(p, manifest) if p.exists() else {}
        missing = set(ids[l]) - set(run_recs[l])
        assert not missing, f"{l}: completed ids without records {missing}"
    systems = {f"{a.mode} run {run_dir.name}": docset(run_recs)}
    sources = {}
    for variant in [REFERENCE] + CONTEXT:
        recs = {}
        for l in LANGS:
            path, meta = _fetch(variant, l, Path(a.cache_dir))
            sources[_fname(variant, l)] = meta
            recs[l] = _records(path, manifest)
            missing = set(ids[l]) - set(recs[l])
            assert not missing, f"{variant}/{l} lacks {missing}"
        systems[variant] = docset(recs)

    rows = {}
    for name, ds in systems.items():
        per = {l: _pooled(ds[l], ids[l], a.resamples) for l in LANGS}
        rows[name] = {"per_language": per, "strict_macro": _strict_macro(per)}

    rng = np.random.default_rng(a.seed)
    samples = [{l: list(rng.choice(ids[l], size=len(ids[l]), replace=True))
                for l in LANGS} for _ in range(a.resamples)]
    ours = systems[next(iter(systems))]
    full = {l: ids[l] for l in LANGS}
    paired = {}
    for name, ds in list(systems.items())[1:]:
        if any(not ids[l] for l in LANGS):
            paired[name] = {"note": "undefined: a language has no completed ids"}
            continue
        d0 = _macro_on(ours, full) - _macro_on(ds, full)
        diffs = np.array([_macro_on(ours, s) - _macro_on(ds, s)
                          for s in samples])
        paired[name] = {
            "diff_strict_macro": round(d0, 4),
            "ci95": [round(float(np.percentile(diffs, 2.5)), 4),
                     round(float(np.percentile(diffs, 97.5)), 4)],
            "p_diff_le_0": round(float(np.mean(diffs <= 0)), 3),
            "includes_zero": bool(np.percentile(diffs, 2.5) <= 0
                                  <= np.percentile(diffs, 97.5)),
        }

    out = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_id": run_dir.name, "split": split, "mode": a.mode,
        "ids": ids, "n_docs": sum(len(v) for v in ids.values()),
        "upstream_commit": PINNED_COMMIT, "upstream_dir": UPSTREAM_DIR,
        "comparator_policy": ("preregistered reference EuroBERT-210m; all "
                              "listed mmBERT DiffAlign variants as context; "
                              "no comparator chosen after seeing results"),
        "metric": ("official token-level Spearman convention via "
                   "splitalign.evaluate._correlate on identical IDs/gold; "
                   "strict macro = mean over de/fr/it, null if any undefined"),
        "paired_bootstrap": {"method": "document-level paired resampling "
                             "within language, strict-macro difference "
                             f"(ours - comparator), B={a.resamples}, seed={a.seed}"},
        "systems": rows, "paired_vs_run": paired, "sources": sources,
        "caveats": [
            "same-ID subset of dev/val only; NOT comparable to whole-dev "
            "numbers (168 docs/lang) and NOT a held-out result",
            "published encoder outputs are the authors' precomputed dev "
            "predictions; they were not rerun here",
            "no matched whole-document Apertus baseline exists for these "
            "IDs; none is invented",
        ],
    }
    out_path = Path(a.out) if a.out else (
        TRACK / "results" / "encoder_context" /
        f"encoder_context_{run_dir.name}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1, allow_nan=False))
    for name, r in rows.items():
        print(f"{str(r['strict_macro']):>8}  "
              + "  ".join(f"{l} {r['per_language'][l]['spearman']}"
                          for l in LANGS) + f"  {name}")
    for name, pr in paired.items():
        print(f"paired ours-{name}: {pr}")
    print(f"-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
