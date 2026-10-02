"""SplitAlign prediction pipeline (per document pair).

segment -> similarity matrix -> monotone DP alignment -> per-pair judgments
-> token labels. Also includes the whole-document Apertus prompt baseline
for comparison.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from . import PROMPT_VERSION, PROMPT_VERSION_BY_KIND, METHOD_VERSIONS, __version__
from .align import AlignConfig, AlignOp, align, coverage
from .apertus import CallLogger, client_from_env_or_mock
from .cache import DiskCache
from .judge import Judge, Judgment, TagJudgment
from . import prompts
from .metricspec import (map_label_from_positive_to_negative,
                         parse_token_labels)
from .score import ScoreConfig, score_item
from .segment import segment_text, segment_text_str


def predict_item(item: dict, judge: Judge,
                 align_cfg: AlignConfig | None = None,
                 score_cfg: ScoreConfig | None = None, *,
                 lang: str) -> dict:
    """One gold item -> prediction record + alignment detail.

    ``lang`` (de/fr/it) is required and names the target language in every
    judge prompt and in each judgment's provenance.
    """
    text_a, text_b = item["text_a"], item["text_b"]
    tokens_a, segs_a = segment_text(text_a)
    tokens_b, segs_b = segment_text(text_b)

    a_texts = [segment_text_str(tokens_a, s, text_a) for s in segs_a]
    b_texts = [segment_text_str(tokens_b, s, text_b) for s in segs_b]

    sim = judge.similarity_matrix(a_texts, b_texts)
    ops = align(sim, align_cfg)

    judgments: list[TagJudgment | None] = []
    for op in ops:
        if op.a_start < 0 or op.b_start < 0:
            judgments.append(None)  # asymmetric op: no pair to judge
            continue
        a_tok = [t.text
                 for s in segs_a[op.a_start:op.a_end]
                 for t in tokens_a[s.start_token:s.end_token]]
        b_tok = [t.text
                 for s in segs_b[op.b_start:op.b_end]
                 for t in tokens_b[s.start_token:s.end_token]]
        judgments.append(judge.judge_tag(a_tok, b_tok,
                                         lang_a="en", lang_b=lang))
    assert all(j is None or j.lang_b == lang for j in judgments)

    # ops whose aligned segment cells touched an unknown similarity cell
    sim_unknown = getattr(judge, "sim_unknown", None) or []
    ops_touching_unknown = sum(
        1 for op in ops
        if op.a_start >= 0 and op.b_start >= 0
        and any(sim_unknown[i][j]
                for i in range(op.a_start, op.a_end)
                for j in range(op.b_start, op.b_end)))

    labels_a, labels_b, stats = score_item(
        tokens_a, tokens_b, segs_a, segs_b, text_a, text_b,
        ops, judgments, score_cfg)

    cov = coverage(ops, len(segs_a), len(segs_b))
    detail = {
        "segments_a": [{"index": s.index, "start": s.start_token,
                        "end": s.end_token, "text": t} for s, t in zip(segs_a, a_texts)],
        "segments_b": [{"index": s.index, "start": s.start_token,
                        "end": s.end_token, "text": t} for s, t in zip(segs_b, b_texts)],
        "ops": [vars(o) for o in ops],
        "judgments": [None if j is None else {
            "a_ids": j.a_ids, "b_ids": j.b_ids,
            "ok": j.ok, "repairs": j.repairs,
            "cached": j.cached, "lang_b": j.lang_b} for j in judgments],
        "judge_lang": lang,
        "judge_prompt_version": PROMPT_VERSION_BY_KIND["judge_tag"],
        "coverage": cov,
        "stats": {"parse_failures": stats.parse_failures,
                  "repairs": stats.repairs,
                  "span_matched": stats.span_matched,
                  "span_unmatched": stats.span_unmatched,
                  "ops": stats.ops,
                  "unparseable_similarity": judge.unparseable_similarity,
                  "sim_unknown_cells": getattr(judge, "sim_unknown_cells", 0),
                  "sim_imputed_cells": getattr(judge, "sim_imputed_cells", 0),
                  "ops_touching_unknown": ops_touching_unknown,
                  "flagged_a": stats.flagged_a,
                  "flagged_b": stats.flagged_b,
                  "dropped_punct_ids": stats.dropped_punct_ids,
                  "invalid_pairs": stats.invalid_pairs,
                  "invalid_fallback_tokens": stats.invalid_fallback_tokens,
                  "matched_tokens": stats.matched_tokens,
                  "valid_matched_tokens": stats.valid_matched_tokens,
                  "judge_valid_token_coverage": (
                      round(stats.valid_matched_tokens / stats.matched_tokens, 4)
                      if stats.matched_tokens else None)},
    }
    record = {
        "id": item["id"], "text_a": text_a, "text_b": text_b,
        "labels_a": labels_a, "labels_b": labels_b,
    }
    return {"record": record, "detail": detail}


_PAIR_RE = re.compile(
    r'\[\s*"((?:[^"\\]|\\.)*)"\s*,\s*"?(-?\d+(?:\.\d+)?)"?\s*\]')


def salvage_sentence_pairs(text: str, key: str) -> list:
    """Recover well-formed [token, label] pairs from a corrupted JSON blob.

    Used only when strict JSON extraction fails: locate the ``"key"`` section
    and regex-scan forward for complete pairs, stopping at the next section
    key or when unparseable structure appears. The recovered pairs still go
    through the official ``parse_token_labels`` matcher.
    """
    start = text.find(f'"{key}"')
    if start < 0:
        return []
    tail = text[start + len(key) + 1:]
    end = tail.find('"sentence')
    if end > 0:
        tail = tail[:end]
    out = []
    for m in _PAIR_RE.finditer(tail):
        try:
            tok = json.loads(f'"{m.group(1)}"')
        except json.JSONDecodeError:
            continue
        out.append([tok, float(m.group(2))])
    return out


def predict_baseline_item(item: dict, judge: Judge) -> dict:
    """Whole-document prompt baseline (official-style token annotation)."""
    tokens_a = item["text_a"].split()
    tokens_b = item["text_b"].split()
    req = prompts.baseline_request(tokens_a, tokens_b)
    msgs = prompts.messages(
        prompts.BASELINE_SYSTEM,
        prompts.BASELINE_USER.format(request=json.dumps(req, ensure_ascii=False)))
    repairs = 0
    cached = False
    while True:
        text, cached, _ = judge._invoke(
            kind="doc_baseline", messages=msgs,
            payload_for_key={"req": req, "r": repairs}, max_tokens=16384)
        # parse with the spec-faithful matcher (metricspec.py; Apache-2.0)
        try:
            from .judge import extract_first_json
            data = extract_first_json(text) or {}
        except Exception:
            data = {}
        s1, s2 = data.get("sentence1"), data.get("sentence2")
        if not isinstance(s1, list) or not isinstance(s2, list):
            # strict extraction failed (truncated/corrupt JSON): salvage
            # complete [token, label] pairs, then keep official parsing.
            s1 = salvage_sentence_pairs(text, "sentence1")
            s2 = salvage_sentence_pairs(text, "sentence2")
            data = {"sentence1": s1, "sentence2": s2}
        complete = (isinstance(s1, list) and len(s1) >= 0.5 * len(tokens_a)
                    and isinstance(s2, list) and len(s2) >= 0.5 * len(tokens_b))
        if complete or repairs >= judge.max_repairs:
            break
        repairs += 1
        msgs = msgs + [{"role": "assistant", "content": text},
                       {"role": "user", "content":
                        "That response was incomplete or invalid JSON. Respond "
                        "with ONLY the complete JSON object covering every "
                        "input token in order."}]
    labels_a = parse_token_labels(tokens_a, data.get("sentence1", []), fallback_label=5.)
    labels_b = parse_token_labels(tokens_b, data.get("sentence2", []), fallback_label=5.)
    labels_a = [map_label_from_positive_to_negative(v) for v in labels_a]
    labels_b = [map_label_from_positive_to_negative(v) for v in labels_b]
    record = {"id": item["id"], "text_a": item["text_a"], "text_b": item["text_b"],
              "labels_a": labels_a, "labels_b": labels_b}
    s1_emitted = len(data.get("sentence1") or [])
    s2_emitted = len(data.get("sentence2") or [])
    detail = {"mode": "doc_baseline", "cached": cached, "repairs": repairs,
              "method_version": METHOD_VERSIONS["baseline"],
              "policy": ("repair-exhausted whole-document annotation: tokens "
                         "with no emitted pair get fallback_label=5 — a "
                         "reference fallback, not a strong comparator"),
              "sentence1_emitted": s1_emitted,
              "sentence2_emitted": s2_emitted,
              "emitted_coverage_a": (
                  round(s1_emitted / len(tokens_a), 4) if tokens_a else None),
              "emitted_coverage_b": (
                  round(s2_emitted / len(tokens_b), 4) if tokens_b else None),
              "fallback_tokens_a": max(0, len(tokens_a) - s1_emitted),
              "fallback_tokens_b": max(0, len(tokens_b) - s2_emitted)}
    return {"record": record, "detail": detail}


def make_judge(backend: str, results_dir: Path, split: str, item_id: str,
               seed: int = 0, budget=None, *, run_id: str | None = None,
               log_path: Path | None = None) -> tuple[Judge, str]:
    """``run_id`` tags every log record (cached and error records included);
    ``log_path`` defaults to the shared ``results_dir/calls.jsonl`` and should
    be the run scope's own ``calls.jsonl`` for pipeline runs."""
    client, backend_name = client_from_env_or_mock(backend, seed=seed)
    if budget is not None and backend_name == "apertus":
        # one shared budget object caps every outbound attempt of this run
        client.budget = budget
    cache = DiskCache(Path(results_dir) / "cache")
    logger = CallLogger(Path(log_path) if log_path else
                        Path(results_dir) / "calls.jsonl", run_id=run_id)
    return Judge(client=client, backend=backend_name, cache=cache,
                 logger=logger, split=split, item_id=item_id, seed=seed,
                 budget=budget), backend_name


def provenance(backend: str, judge: Judge, extra: dict | None = None) -> dict:
    prov = {
        "splitalign_version": __version__,
        "prompt_version": PROMPT_VERSION,
        "backend": backend,
        "model": getattr(judge.client, "model", None),
        "note": ("MOCK backend: outputs are deterministic heuristics, NOT "
                 "Apertus inference and NOT model performance evidence."
                 if backend == "mock" else
                 "Apertus backend via configured hosted endpoint."),
    }
    if extra:
        prov.update(extra)
    return prov
