"""SplitAlign CLI.

    python -m splitalign.run fetch-data
    python -m splitalign.run predict  --split val --lang de --limit 3 --backend mock
    python -m splitalign.run baseline --split val --lang de --limit 3 --backend mock
    python -m splitalign.run evaluate --pred results/predictions --split val
    python -m splitalign.run calibrate --langs de,fr,it
    python -m splitalign.run pipeline --split val --lang all --limit 3 --backend mock
    python -m splitalign.run export-viewer --split val --backend mock

Splits are firewall-enforced: only dev/train and dev/val exist here.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import sys
import time
from pathlib import Path

from . import PROMPT_VERSION, __version__, guard
from .align import AlignConfig
from .apertus import (ApiBudget, ApertusUnavailable, BudgetExceeded,
                      MissingCredentials)
from .evaluate import evaluate_predictions, evaluate_split
from .fetch_data import ALLOWLIST, fetch, load_gold_items, write_manifest
from .pipeline import (make_judge, predict_baseline_item, predict_item,
                       provenance)
from .score import ScoreConfig
from .viewer_export import export_evidence

TRACK_DIR = guard.TRACK_DIR
# SPLITALIGN_OUT redirects every produced artifact (results + viewer) — the
# Docker image sets it to /out, which the Makefile bind-mounts to ./out.
OUT_DIR = Path(os.environ.get("SPLITALIGN_OUT", str(TRACK_DIR)))
RESULTS_DIR = OUT_DIR / "results"
PRED_DIR = RESULTS_DIR / "predictions"
DETAIL_DIR = RESULTS_DIR / "details"
EVIDENCE_PATH = OUT_DIR / "viewer" / "evidence.js"
CONFIG_PATH = RESULTS_DIR / "calibration.json"


def resolve_backend(cli_backend: str | None) -> str:
    """Single source of truth for the backend.

    Precedence: explicit ``--backend`` > ``SPLITALIGN_BACKEND`` env > mock.
    A set env that contradicts an explicit CLI value is a configuration
    error — never silently relabel output provenance.
    """
    env = os.environ.get("SPLITALIGN_BACKEND") or None
    if env is not None and env not in ("mock", "apertus"):
        raise SystemExit(
            f"invalid SPLITALIGN_BACKEND={env!r}; expected 'mock' or 'apertus'")
    if cli_backend and env and cli_backend != env:
        raise SystemExit(
            f"contradictory backend config: --backend {cli_backend} vs "
            f"SPLITALIGN_BACKEND={env}; set only one source")
    return cli_backend or env or "mock"


def _items(split: str, lang: str, limit: int | None, offset: int = 0):
    items = load_gold_items(split, lang)
    if offset:
        items = items[offset:]
    if limit:
        items = items[:limit]
    return items


def _langs(lang: str) -> list[str]:
    return list(guard.ALLOWED_LANGS) if lang == "all" else [lang]


def _pred_path(backend: str, tag: str, lang: str) -> Path:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    return PRED_DIR / f"{tag}_{backend}_admin_{lang}.jsonl"


def cmd_fetch_data(args) -> int:
    log = fetch()
    print(f"fetched/verified {len(log['files'])} allowlisted files @ {log['commit'][:10]}")
    offenders = guard.scan_tree_for_heldout()
    if offenders:
        print("HELD-OUT MATERIAL PRESENT:", offenders)
        return 2
    print("held-out scan: clean")
    return 0


def _predict(args, mode: str) -> int:
    split = guard.normalize_split(args.split)
    if split == "dev":
        raise SystemExit("specify --split train or val")
    seed = args.seed
    backend = args.backend
    # shared noncached-API budget across modes (mock calls are free)
    budget = getattr(args, "_budget", None)
    if budget is None and backend == "apertus":
        budget = ApiBudget(max_requests=args.max_requests,
                           max_tokens=args.max_tokens)
        args._budget = budget
    all_details = []
    capped = None
    completed_ids: dict[str, list[str]] = {}
    planned_ids: dict[str, list[str]] = {}
    t0 = time.monotonic()
    try:
        for lang in _langs(args.lang):
            recs = []
            wanted = _items(split, lang, args.limit, args.offset)
            planned_ids[lang] = [it["id"] for it in wanted]
            completed_ids[lang] = []
            for item in wanted:
                judge, backend = make_judge(args.backend, RESULTS_DIR, split,
                                            item["id"], seed, budget=budget)
                try:
                    if mode == "baseline":
                        out = predict_baseline_item(item, judge)
                    else:
                        out = predict_item(item, judge,
                                           score_cfg=_load_score_cfg(args))
                except ApertusUnavailable as e:
                    # truthful per-item failure: no prediction, run continues
                    det = {"id": item["id"], "lang": lang, "failed": True,
                           "error": str(e)[:300],
                           "provenance": provenance(backend, judge,
                                                    {"mode": mode, "split": split})}
                    all_details.append(det)
                    print(f"  {item['id']}: FAILED ({mode}) {e}", file=sys.stderr)
                    continue
                recs.append(out["record"])
                completed_ids[lang].append(item["id"])
                det = dict(out["detail"])
                det["id"] = item["id"]
                det["lang"] = lang
                det["text_a"] = item["text_a"]
                det["text_b"] = item["text_b"]
                det["labels_a"] = out["record"]["labels_a"]
                det["labels_b"] = out["record"]["labels_b"]
                det["gold_labels_a"] = item["labels_a"]
                det["gold_labels_b"] = item["labels_b"]
                det["provenance"] = provenance(backend, judge,
                                               {"mode": mode, "split": split})
                all_details.append(det)
                print(f"  {item['id']}: done ({mode})", file=sys.stderr)
            path = _pred_path(backend, mode, lang)
            with path.open("w") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"wrote {len(recs)} predictions -> {path}")
    except BudgetExceeded as e:
        # hard cap hit mid-item: keep every completed record, report honestly
        capped = str(e)
        print(f"BUDGET CAP: {capped} — writing partial results", file=sys.stderr)
        if recs:  # `lang`/`recs` remain bound from the interrupted iteration
            path = _pred_path(backend, mode, lang)
            with path.open("w") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"wrote {len(recs)} PARTIAL predictions -> {path}",
                  file=sys.stderr)
    DETAIL_DIR.mkdir(parents=True, exist_ok=True)
    det_path = DETAIL_DIR / f"{mode}_{args.backend}_{split.replace('/', '_')}.json"
    det_path.write_text(json.dumps(all_details, ensure_ascii=False, indent=1))
    print(f"details -> {det_path}")
    # run summary: completed ids, coverage, usage, remainder — never hide a cap
    elapsed = time.monotonic() - t0
    summary = {
        "mode": mode, "backend": backend, "split": split,
        "completed_ids": completed_ids,
        "remaining_ids": {k: [i for i in v if i not in set(completed_ids[k])]
                          for k, v in planned_ids.items()},
        "n_planned": sum(len(v) for v in planned_ids.values()),
        "n_completed": sum(len(v) for v in completed_ids.values()),
        "budget_capped": capped,
        "new_api_requests": budget.requests if budget else 0,
        "new_api_tokens": budget.tokens if budget else 0,
        "logical_calls": budget.logical_calls if budget else 0,
        "attempts_without_usage": budget.attempts_no_usage if budget else 0,
        "usage_uncertainty_note": (
            f"{budget.attempts_no_usage} failed HTTP attempt(s) have unknown "
            "token cost and are NOT included in new_api_tokens"
            if budget and budget.attempts_no_usage else None),
        "overshoot_note": (
            "a single in-flight request may overshoot the token cap by at "
            "most prompt + bounded max_tokens completion" if budget else None),
        "elapsed_s": round(elapsed, 1),
        "caution": ("PARTIAL RUN — do not treat coverage as complete"
                    if capped else None),
    }
    spath = RESULTS_DIR / f"run_summary_{mode}_{args.backend}_{split.replace('/', '_')}.json"
    spath.write_text(json.dumps({k: v for k, v in summary.items() if v is not None},
                                indent=2))
    print(f"run summary -> {spath}")
    return 0


def cmd_evaluate(args) -> int:
    split = guard.normalize_split(args.split)
    if split == "dev":
        raise SystemExit("specify --split train or val")
    res = evaluate_split(Path(args.pred), split,
                         n_resamples=args.bootstrap,
                         prefix=args.prefix,
                         require_full_coverage=args.require_full)
    print(json.dumps(res, indent=2))
    return 0


def _load_score_cfg(args) -> ScoreConfig:
    if args.cfg and Path(args.cfg).exists():
        return ScoreConfig(**json.loads(Path(args.cfg).read_text()))
    if CONFIG_PATH.exists():
        return ScoreConfig(**json.loads(CONFIG_PATH.read_text()))
    return ScoreConfig()


def cmd_calibrate(args) -> int:
    """Grid-search score/align params on dev/train, evaluate on dev/val."""
    split_train, split_val = "dev/train", "dev/val"
    langs = _langs(args.lang)
    limit = args.limit

    # run predictions ONCE per config on train, pick best, then val.
    grids = {
        "span_boost": [0.0, 0.2, 0.4],
        "omit_label": [0.7, 0.9, 1.0],
        "add_label": [0.7, 0.9, 1.0],
    }
    keys = list(grids)
    results = []
    for combo in itertools.product(*(grids[k] for k in keys)):
        cfg = ScoreConfig(**dict(zip(keys, combo)))
        spear_sum, n = 0.0, 0
        for lang in langs:
            recs = []
            for item in _items(split_train, lang, limit):
                judge, backend = make_judge(args.backend, RESULTS_DIR,
                                            split_train, item["id"], args.seed)
                recs.append(predict_item(item, judge, score_cfg=cfg)["record"])
            gold = guard.gold_path(split_train, lang)
            r = evaluate_predictions(recs, gold, lang, n_resamples=0)
            spear_sum += r.spearman
            n += 1
        macro = spear_sum / n
        results.append((macro, dict(zip(keys, combo))))
        print(f"train macro {macro:.4f}  {dict(zip(keys, combo))}")
    best = max(results, key=lambda r: r[0])
    cfg = ScoreConfig(**best[1])
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(vars(cfg), indent=2))
    print(f"best train config {best[1]} -> {CONFIG_PATH}")
    return 0


def cmd_export_viewer(args) -> int:
    split = guard.normalize_split(args.split)
    tag = split.replace("/", "_")
    items_by_mode: dict[str, list] = {}
    evals: dict = {}
    modes = sorted(p.name.split("_" + args.backend + "_")[0]
                   for p in DETAIL_DIR.glob(f"*_{args.backend}_{tag}.json"))
    for mode in modes:
        det_path = DETAIL_DIR / f"{mode}_{args.backend}_{tag}.json"
        if det_path.exists():
            items_by_mode[mode] = json.loads(det_path.read_text())
        ep = RESULTS_DIR / f"eval_{mode}_{args.backend}_{tag}.json"
        if ep.exists():
            evals[mode] = json.loads(ep.read_text())
    if not items_by_mode:
        raise SystemExit("no details found; run predict first")
    # actual executed model, from recorded provenance (never assumed)
    model = next((it["provenance"].get("model")
                  for d in items_by_mode.values() for it in d
                  if it.get("provenance", {}).get("model")), None)
    out = export_evidence(items_by_mode, EVIDENCE_PATH, backend=args.backend,
                          model=model, split=split, evaluation=evals,
                          limitations=_limitations(args.backend))
    # keep the viewer self-contained wherever OUT_DIR points
    for name in ("index.html", "style.css", "app.js"):
        src = TRACK_DIR / "viewer" / name
        dst = EVIDENCE_PATH.parent / name
        if src.resolve() != dst.resolve():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    print(f"evidence -> {out}")
    return 0


def _limitations(backend: str) -> list[str]:
    lims = [
        "Sentence-level judgments are mapped to uniform token scores except on reported differing spans — fine-grained in-sentence variation is approximated.",
        "Cross-lingual alignment is monotone; non-monotone reordering is not modelled.",
        "Only dev/train and dev/val were used; the held-out firewall is enforced by this build's guard module.",
        "The 'baseline' method is our re-implemented whole-document token-annotation prompt (same response contract; not the upstream template).",
        "Similarity candidates are banded (|i/(n-1) - j/(m-1)| <= 0.2, +/-2 absolute): far-off-diagonal pairs score 0 without an API call.",
    ]
    if backend == "mock":
        lims.insert(0, "MOCK backend: all shown outputs are deterministic lexical heuristics, NOT Apertus inference. Numbers are plumbing validations only.")
    return lims


def cmd_pipeline(args) -> int:
    """predict (splitalign + baseline) -> evaluate -> export viewer evidence."""
    split = guard.normalize_split(args.split)
    if split == "dev":
        raise SystemExit("specify --split train or val")
    for mode in ("splitalign", "baseline"):
        ns = argparse.Namespace(**vars(args))
        _predict(ns, mode)
        res = evaluate_split(PRED_DIR, split,
                             langs=_langs(args.lang),
                             n_resamples=args.bootstrap,
                             prefix=f"{mode}_{args.backend}",
                             require_full_coverage=args.require_full)
        (RESULTS_DIR / f"eval_{mode}_{args.backend}_{split.replace('/', '_')}.json"
         ).write_text(json.dumps(res, indent=2))
        _m = res['macro_spearman']
        print(f"[{mode}] macro Spearman: "
              + (f"{_m:.4f}" if _m is not None else "null (undefined)"))
        ns2 = argparse.Namespace(**vars(args))
        ns2.mode = mode
        cmd_export_viewer(ns2)
    return 0


def cmd_selftest(args) -> int:
    """Smoke: sample doc through the mock pipeline, evaluate vs itself."""
    import tempfile
    items = load_gold_items("dev/val", "de")[:1]
    with tempfile.TemporaryDirectory() as td:
        judge, backend = make_judge("mock", Path(td), "dev/val", items[0]["id"], 0)
        out = predict_item(items[0], judge)
        gold = guard.gold_path("dev/val", "de")
        r = evaluate_predictions([out["record"]], gold, "de", n_resamples=0)
        print(f"selftest ok: 1 item, spearman {r.spearman:.3f}, "
              f"tokens {r.n_tokens}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="splitalign")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("fetch-data"); p.set_defaults(fn=cmd_fetch_data)
    p = sub.add_parser("selftest"); p.set_defaults(fn=cmd_selftest)

    for name in ("predict", "baseline", "pipeline", "calibrate"):
        p = sub.add_parser(name)
        p.add_argument("--split", default="val")
        p.add_argument("--lang", default="all")
        p.add_argument("--limit", type=int, default=None)
        p.add_argument("--offset", type=int, default=0)
        p.add_argument("--backend", default=None, choices=["mock", "apertus"],
                       help="inference backend; default: SPLITALIGN_BACKEND env "
                            "else mock. Contradicting the env var is an error.")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--bootstrap", type=int, default=1000)
        p.add_argument("--cfg", default=None)
        p.add_argument("--require-full", action="store_true",
                       help="fail loudly if predictions cover <100% of gold "
                            "items (default: warn via coverage fields)")
        p.add_argument("--max-requests", type=int, default=500,
                       help="cap on NEW (noncached) API requests, apertus only")
        p.add_argument("--max-tokens", type=int, default=600_000,
                       help="cap on NEW API tokens (prompt+completion), "
                            "apertus only")
        if name == "predict":
            p.set_defaults(fn=lambda a: _predict(a, "splitalign"))
        elif name == "baseline":
            p.set_defaults(fn=lambda a: _predict(a, "baseline"))
        elif name == "pipeline":
            p.set_defaults(fn=cmd_pipeline)
        else:
            p.set_defaults(fn=cmd_calibrate)

    p = sub.add_parser("evaluate")
    p.add_argument("--pred", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--require-full", action="store_true",
                   help="fail loudly if predictions cover <100% of gold items")
    p.add_argument("--prefix", default=None,
                   help="prediction file prefix e.g. splitalign_mock")
    p.set_defaults(fn=cmd_evaluate)

    p = sub.add_parser("export-viewer")
    p.add_argument("--split", default="val")
    p.add_argument("--backend", default=None, choices=["mock", "apertus"],
                   help="see predict --backend")
    p.add_argument("--mode", default="splitalign")
    p.set_defaults(fn=cmd_export_viewer)

    args = ap.parse_args(argv)
    if getattr(args, "backend", None) is not None or hasattr(args, "backend"):
        args.backend = resolve_backend(args.backend)
    try:
        return args.fn(args)
    except MissingCredentials as e:
        print(f"CONFIGURATION REQUIRED\n{e}", file=sys.stderr)
        return 3
    except guard.HeldOutViolation as e:
        print(f"HELD-OUT FIREWALL: {e}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
