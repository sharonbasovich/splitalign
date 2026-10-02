#!/usr/bin/env python3
"""Deterministic cost-ledger audit over the raw call logs.

Reads ONLY the immutable logs (results/calls.jsonl + results/runs/*/calls.jsonl)
and prints the classification used by cost_ledger.md. No file is modified;
nothing is summed twice — each input row is classified into exactly one bucket.

Buckets (a row is classified by backend THEN usage_estimated THEN cached THEN ok):
  - apertus noncached success   -> measured provider usage (real spend)
  - apertus noncached failure   -> real attempt, UNKNOWN token cost (never zero)
  - mock    noncached           -> estimated usage ONLY; excluded from real spend
  - unknown backend noncached   -> unknown cost, never counted as zero
  - cached (any backend)        -> free replay, counted raw and distinct
Exact-duplicate rows (byte-identical canonical JSON) are reported per class;
same-second cache hits are legitimate repeated reads and are NOT treated as
proven duplicate exports — both raw and distinct counts are reported.

Logical records are NOT outbound HTTP attempts: internal transport retries
are not individually logged, so attempt counts here are a floor.
"""
import argparse
import collections
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def canon(rec):
    return json.dumps(rec, sort_keys=True)


def backend_of(rec):
    b = rec.get("backend")
    return b if b in ("apertus", "mock") else "unknown"


def classify(rec):
    bk = backend_of(rec)
    if rec.get("cached"):
        return f"cache_{bk}"
    if bk == "mock" or rec.get("usage_estimated"):
        return "mock_noncached"
    if bk == "apertus":
        return "apertus_noncached_ok" if rec.get("ok") else "apertus_noncached_fail"
    return "unknown_noncached"


def audit_file(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    out = collections.OrderedDict()
    seen = set()
    dup_extra = collections.Counter()
    for r in rows:
        cls = classify(r)
        k = canon(r)
        if k in seen:
            dup_extra[cls] += 1
        seen.add(k)
        b = out.setdefault(cls, {
            "raw": 0, "distinct": 0, "ok": 0, "fail": 0,
            "prompt_tokens": 0, "completion_tokens": 0,
            "usage_estimated": 0, "first_ts": None, "last_ts": None})
        b["raw"] += 1
        b["ok"] += 1 if r.get("ok") else 0
        b["fail"] += 0 if r.get("ok") else 1
        b["usage_estimated"] += 1 if r.get("usage_estimated") else 0
        ts = r.get("ts")
        if ts:
            b["first_ts"] = ts if b["first_ts"] is None or ts < b["first_ts"] else b["first_ts"]
            b["last_ts"] = ts if b["last_ts"] is None or ts > b["last_ts"] else b["last_ts"]
    # second pass for distinct counts (canonical-set dedupe per class)
    per_cls = collections.defaultdict(set)
    for r in rows:
        per_cls[classify(r)].add(canon(r))
    for cls, keys in per_cls.items():
        out[cls]["distinct"] = len(keys)
    for cls, n in dup_extra.items():
        out[cls]["dup_extra_rows"] = n
    # token sums only over distinct rows to avoid double counting
    for cls, keys in per_cls.items():
        pt = ct = 0
        for k in keys:
            r = json.loads(k)
            pt += r.get("prompt_tokens") or 0
            ct += r.get("completion_tokens") or 0
        out[cls]["prompt_tokens_distinct"] = pt
        out[cls]["completion_tokens_distinct"] = ct
    return {"path": os.path.relpath(path, HERE), "n_raw": len(rows),
            "n_distinct": len(seen), "buckets": out}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=HERE, help="results dir (default: script dir)")
    a = ap.parse_args()
    files = [os.path.join(a.root, "calls.jsonl")]
    files += sorted(glob.glob(os.path.join(a.root, "runs", "*", "calls.jsonl")))
    report = {"files": [audit_file(f) for f in files if os.path.exists(f)]}
    # combined, no double counting: legacy window + pilot are disjoint logs
    # (verified: no canonical row appears in more than one file)
    union = set()
    overlap = 0
    for fr in report["files"]:
        p = os.path.join(a.root, fr["path"])
        keys = {canon(json.loads(l)) for l in open(p) if l.strip()}
        overlap += len(union & keys)
        union |= keys
    report["cross_file_exact_overlap"] = overlap
    comb = collections.defaultdict(lambda: collections.Counter())
    for fr in report["files"]:
        for cls, b in fr["buckets"].items():
            for kk in ("raw", "distinct", "ok", "fail", "prompt_tokens_distinct",
                       "completion_tokens_distinct", "usage_estimated"):
                comb[cls][kk] += b[kk]
    report["combined"] = {k: dict(v) for k, v in comb.items()}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
