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
import hashlib
import subprocess
import itertools
import json
import os
import shutil
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from . import PROMPT_VERSION, __version__, guard
from .align import AlignConfig
from .apertus import (ApiBudget, ApertusUnavailable, BudgetExceeded,
                      MissingCredentials)
from .evaluate import _dump, evaluate_predictions, evaluate_split
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


def _pred_path(pred_dir: Path, backend: str, tag: str, lang: str) -> Path:
    pred_dir.mkdir(parents=True, exist_ok=True)
    return pred_dir / f"{tag}_{backend}_admin_{lang}.jsonl"


def _new_run_dir() -> Path:
    """Fresh immutable run scope: all artifacts of one invocation live here
    so partial/capped outputs can never mix with earlier runs."""
    d = RESULTS_DIR / "runs" / (
        f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-"
        f"{uuid.uuid4().hex[:8]}")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _source_info() -> dict:
    """Exact source state of THIS process — recorded, never inferred later.

    Falls back to SPLITALIGN_SOURCE_COMMIT / SPLITALIGN_SOURCE_DIRTY (passed
    into the Docker container by the Makefile) and reports null when neither
    git nor the env vars are available.
    """
    info = {"commit": None, "dirty": None, "note": None}
    try:
        root = TRACK_DIR.parent
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10)
        if head.returncode == 0:
            info["commit"] = head.stdout.strip()
            st = subprocess.run(["git", "-C", str(root), "status",
                                 "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True, timeout=10)
            info["dirty"] = bool(st.stdout.strip()) if st.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        pass
    if info["commit"] is None and os.environ.get("SPLITALIGN_SOURCE_COMMIT"):
        info["commit"] = os.environ["SPLITALIGN_SOURCE_COMMIT"]
        dirty = os.environ.get("SPLITALIGN_SOURCE_DIRTY")
        info["dirty"] = {"0": False, "1": True}.get(dirty)
        info["note"] = ("commit/dirty from SPLITALIGN_SOURCE_* env (host checkout)"
                        if info["dirty"] is not None else
                        "commit from SPLITALIGN_SOURCE_COMMIT env; dirty state unknown")
    elif info["commit"] is None:
        info["note"] = "git metadata unavailable (not a git checkout)"
    return info


def _gold_sha256(split: str, langs) -> dict:
    return {l: hashlib.sha256(guard.gold_path(split, l).read_bytes()).hexdigest()
            for l in langs}


def _write_manifest(run_dir: Path, mode: str, planned_ids: dict,
                    backend: str, split: str, meta: dict | None = None) -> None:
    """Intended-ID manifest, updated as each mode declares its plan."""
    man_path = run_dir / "manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {
        "run_id": run_dir.name,
        "backend": backend,
        "split": split,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "modes": {},
        "note": ("intended IDs declared before inference; compare methods "
                 "only on identical produced IDs; a mode missing from "
                 "'modes' never ran in this scope"),
    }
    if meta:
        for k, v in meta.items():
            man.setdefault(k, v)
    man["modes"][mode] = {"intended_ids": planned_ids,
                          "n_intended": sum(len(v) for v in planned_ids.values())}
    man_path.write_text(json.dumps(man, indent=2))


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
    langs = _langs(args.lang)
    # state for EVERY planned language exists before inference starts, so a
    # cap hit in an early language still yields truthful completed/remaining
    completed_ids: dict[str, list[str]] = {l: [] for l in langs}
    started: set[str] = set()
    run_dir = getattr(args, "_run_dir", None)
    if run_dir is None:
        run_dir = _new_run_dir()
        args._run_dir = run_dir
    planned_ids = {l: [it["id"] for it in _items(split, l, args.limit,
                                                  args.offset)]
                   for l in langs}
    score_cfg = _load_score_cfg(args)
    _write_manifest(run_dir, mode, planned_ids, args.backend, split, meta={
        "source": _source_info(),
        "splitalign_version": __version__,
        "prompt_version": PROMPT_VERSION,
        "gold_sha256": _gold_sha256(split, langs),
        "score_config": asdict(score_cfg),
        "cli": {k: getattr(args, k, None) for k in
                ("limit", "offset", "seed", "max_requests", "max_tokens",
                 "cfg", "bootstrap", "require_full")},
    })
    t0 = time.monotonic()
    recs: list = []
    lang: str | None = None
    try:
        for lang in langs:
            recs = []
            wanted = _items(split, lang, args.limit, args.offset)
            started.add(lang)
            for item in wanted:
                judge, backend = make_judge(args.backend, RESULTS_DIR, split,
                                            item["id"], seed, budget=budget)
                try:
                    if mode == "baseline":
                        out = predict_baseline_item(item, judge)
                    else:
                        out = predict_item(item, judge, score_cfg=score_cfg)
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
            path = _pred_path(run_dir, backend, mode, lang)
            with path.open("w") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"wrote {len(recs)} predictions -> {path}")
    except BudgetExceeded as e:
        # hard cap hit mid-item: keep every completed record, report honestly
        capped = str(e)
        print(f"BUDGET CAP: {capped} — writing partial results", file=sys.stderr)
        if recs and lang is not None:  # bound from the interrupted iteration
            path = _pred_path(run_dir, backend, mode, lang)
            with path.open("w") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"wrote {len(recs)} PARTIAL predictions -> {path}",
                  file=sys.stderr)
    det_path = run_dir / f"details_{mode}_{args.backend}_{split.replace('/', '_')}.json"
    det_path.write_text(json.dumps(all_details, ensure_ascii=False, indent=1))
    print(f"details -> {det_path}")
    # run summary: completed ids, coverage, usage, remainder — never hide a cap
    elapsed = time.monotonic() - t0
    def _status(l: str) -> str:
        if len(completed_ids[l]) == len(planned_ids[l]):
            return "complete"
        if completed_ids[l]:
            return "partial"
        return "started_no_output" if l in started else "not_started"
    summary = {
        "mode": mode, "backend": backend, "split": split,
        "completed_ids": completed_ids,
        "remaining_ids": {k: [i for i in v if i not in set(completed_ids[k])]
                          for k, v in planned_ids.items()},
        "lang_status": {l: _status(l) for l in langs},
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
    spath = run_dir / f"run_summary_{mode}_{args.backend}_{split.replace('/', '_')}.json"
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


def _load_strict_json(path: Path) -> dict:
    """Reject NaN/Infinity so a non-strict artifact can never be published."""
    def _bad(c):
        raise SystemExit(f"{path}: non-strict JSON constant {c}; regenerate "
                         "this artifact with current code")
    return json.loads(path.read_text(), parse_constant=_bad)


def _resolve_run_dir(args) -> Path:
    """Explicit run scope only — never promote the newest directory."""
    run_dir = getattr(args, "_run_dir", None)
    if run_dir is None:
        run = getattr(args, "run", None)
        if not run:
            runs_root = RESULTS_DIR / "runs"
            avail = sorted(d.name for d in runs_root.iterdir()
                           if d.is_dir()) if runs_root.exists() else []
            raise SystemExit(
                "export-viewer requires --run <run id or dir>; the newest run "
                "is never promoted implicitly (it may be capped or partial). "
                f"Available: {avail or 'none'}")
        run_dir = Path(run)
        if not run_dir.exists() and (RESULTS_DIR / "runs" / run).exists():
            run_dir = RESULTS_DIR / "runs" / run
    run_dir = Path(run_dir)
    if not (run_dir / "manifest.json").exists():
        raise SystemExit(f"{run_dir} is not a run scope (no manifest.json)")
    return run_dir


def _run_info(run_dir: Path, split: str, backend: str, tag: str,
              produced_modes) -> dict:
    """Identity + coverage of a run scope, read from its own artifacts only."""
    man = _load_strict_json(run_dir / "manifest.json")
    langs = sorted({l for m in man["modes"].values()
                    for l in m["intended_ids"]})
    n_split = {l: len(load_gold_items(split, l)) for l in langs}
    modes = {}
    for mode, m in man["modes"].items():
        sp = run_dir / f"run_summary_{mode}_{backend}_{tag}.json"
        summ = _load_strict_json(sp) if sp.exists() else None
        entry = {
            "intended": {l: len(v) for l, v in m["intended_ids"].items()},
            "n_intended": m["n_intended"],
            "produced": mode in produced_modes,
            "summary_present": summ is not None,
        }
        if summ:
            entry.update({
                "completed": {l: len(v) for l, v in summ["completed_ids"].items()},
                "n_completed": summ["n_completed"],
                "lang_status": summ.get("lang_status"),
                "budget_capped": summ.get("budget_capped"),
                "caution": summ.get("caution"),
                "new_api_requests": summ.get("new_api_requests"),
                "new_api_tokens": summ.get("new_api_tokens"),
            })
        else:
            entry["caution"] = ("intended in manifest but no run summary — "
                                "this mode did not finish in this scope")
        entry["coverage_of_split"] = {
            l: round(len(summ["completed_ids"].get(l, [])) / max(n_split[l], 1), 4)
            if summ else 0.0 for l in langs}
        modes[mode] = entry
    mp = run_dir / f"eval_matched_{backend}_{tag}.json"
    matched = None
    if mp.exists():
        mj = _load_strict_json(mp)
        matched = {"n_matched": {l: v["n_matched"] for l, v in
                                 mj["per_language"].items()},
                   "macro_matched": mj.get("macro_matched")}
    produced = sorted(produced_modes)
    not_produced = sorted(set(man["modes"]) - set(produced))
    partial = any((e.get("budget_capped") or not e["produced"]
                   or e.get("n_completed", 0) < e["n_intended"])
                  for e in modes.values())
    return {
        "run_id": man["run_id"],
        "started_utc": man.get("started_utc"),
        "source": man.get("source"),
        "prompt_version": man.get("prompt_version"),
        "n_split": n_split,
        "modes": modes,
        "modes_produced": produced,
        "modes_intended_not_produced": not_produced,
        "matched": matched,
        "partial": partial,
        "comparison_note": (
            "methods are comparable ONLY on matched IDs (see matched); "
            "an intended mode that produced nothing yields no comparison"
            if not_produced or not matched else
            "matched-ID comparison available for the produced modes"),
    }


def cmd_export_viewer(args) -> int:
    split = guard.normalize_split(args.split)
    tag = split.replace("/", "_")
    run_dir = _resolve_run_dir(args)
    det_paths = sorted(run_dir.glob(f"details_*_{args.backend}_{tag}.json"))
    if not det_paths:
        raise SystemExit(f"no details for backend '{args.backend}' / {split} "
                         f"in run scope {run_dir}")
    items_by_mode: dict[str, list] = {}
    evals: dict = {}
    for det_path in det_paths:
        mode = det_path.name.split(f"_{args.backend}_{tag}")[0] \
            .removeprefix("details_")
        items_by_mode[mode] = _load_strict_json(det_path)
        ep = run_dir / f"eval_{mode}_{args.backend}_{tag}.json"
        if ep.exists():
            evals[mode] = _load_strict_json(ep)
    run_info = _run_info(run_dir, split, args.backend, tag, items_by_mode)
    # actual executed model, from recorded provenance (never assumed)
    model = next((it["provenance"].get("model")
                  for d in items_by_mode.values() for it in d
                  if it.get("provenance", {}).get("model")), None)
    out = export_evidence(items_by_mode, EVIDENCE_PATH, backend=args.backend,
                          model=model, split=split, evaluation=evals,
                          limitations=_limitations(args.backend),
                          run=run_info)
    # keep the viewer self-contained wherever OUT_DIR points
    for name in ("index.html", "style.css", "app.js"):
        src = TRACK_DIR / "viewer" / name
        dst = EVIDENCE_PATH.parent / name
        if src.resolve() != dst.resolve():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    print(f"evidence -> {out} (run {run_info['run_id']}"
          f"{', PARTIAL' if run_info['partial'] else ''})")
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


def _matched_eval(run_dir: Path, split: str, langs, backend: str,
                  n_resamples: int) -> dict:
    """Compare methods ONLY on identical produced IDs.

    Per language, the matched set is the exact intersection of the IDs
    each method actually emitted — never a union, never the gold set.
    """
    per_lang = {}
    macros = {"splitalign": [], "baseline": []}
    for lang in langs:
        recs = {}
        for mode in ("splitalign", "baseline"):
            p = run_dir / f"{mode}_{backend}_admin_{lang}.jsonl"
            recs[mode] = {r["id"]: r for r in
                          (json.loads(l) for l in
                           p.read_text().splitlines() if l.strip())} \
                if p.exists() else {}
        common = sorted(set(recs["splitalign"]) & set(recs["baseline"]))
        gold = guard.gold_path(split, lang)
        gold_ids = [it["id"] for it in load_gold_items(split, lang)]
        entry = {"n_matched": len(common),
                 "n_split": len(gold_ids),
                 "coverage_of_split": round(len(common) / max(len(gold_ids), 1), 4),
                 "matched_ids": common}
        for mode in ("splitalign", "baseline"):
            r = evaluate_predictions(
                [recs[mode][i] for i in common], gold, lang, n_resamples)
            d = _dump_eval(r)
            entry[mode] = d
            if d["spearman"] is not None:
                macros[mode].append(d["spearman"])
        per_lang[lang] = entry

    def _macro(v):
        # strict: null unless every required lang produced a finite value
        return (sum(v) / len(v)) if len(v) == len(langs) else None
    return {
        "note": ("methods compared only on identical produced IDs; "
                 "macro is null unless ALL required languages are matched "
                 "and finite"),
        "per_language": per_lang,
        "macro_matched": {
            "splitalign": _macro(macros["splitalign"]),
            "baseline": _macro(macros["baseline"]),
        },
    }


def _dump_eval(r):
    return _dump(r)


def cmd_pipeline(args) -> int:
    """predict (splitalign + baseline) -> evaluate -> export viewer evidence."""
    split = guard.normalize_split(args.split)
    if split == "dev":
        raise SystemExit("specify --split train or val")
    # ONE shared budget + ONE immutable run scope across BOTH methods — a
    # second method can never reset the cap or mix with stale artifacts.
    if args.backend == "apertus":
        args._budget = ApiBudget(max_requests=args.max_requests,
                                 max_tokens=args.max_tokens)
    args._run_dir = _new_run_dir()
    for mode in ("splitalign", "baseline"):
        _predict(args, mode)
        res = evaluate_split(args._run_dir, split,
                             langs=_langs(args.lang),
                             n_resamples=args.bootstrap,
                             prefix=f"{mode}_{args.backend}",
                             require_full_coverage=args.require_full)
        (args._run_dir / f"eval_{mode}_{args.backend}_{split.replace('/', '_')}.json"
         ).write_text(json.dumps(res, indent=2))
        _m = res['macro_spearman']
        print(f"[{mode}] macro Spearman: "
              + (f"{_m:.4f}" if _m is not None else "null (undefined)"))
    matched = _matched_eval(args._run_dir, split, _langs(args.lang),
                            args.backend, args.bootstrap)
    mp = args._run_dir / f"eval_matched_{args.backend}_{split.replace('/', '_')}.json"
    mp.write_text(json.dumps(matched, indent=2))
    print(f"matched-ID comparison -> {mp}")
    # export once, after both modes + matched eval exist in this scope
    cmd_export_viewer(args)
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
    p.add_argument("--run", default=None,
                   help="run dir to export from (default: newest run scope)")
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
