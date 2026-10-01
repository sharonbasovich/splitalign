"""SplitAlign prediction pipeline (per document pair).

segment -> similarity matrix -> monotone DP alignment -> per-pair judgments
-> token labels. Also includes the whole-document Apertus prompt baseline
for comparison.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from . import PROMPT_VERSION, __version__
from .align import AlignConfig, AlignOp, align, coverage
from .apertus import CallLogger, client_from_env_or_mock
from .cache import DiskCache
from .judge import Judge, Judgment
from . import prompts
from .score import ScoreConfig, score_item
from .segment import segment_text, segment_text_str

VENDOR_DIR = Path(__file__).resolve().parents[1] / "vendor" / "swissgov_rsd"
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

from evaluation.utils import map_label_from_positive_to_negative, parse_token_labels  # noqa: E402


def predict_item(item: dict, judge: Judge,
                 align_cfg: AlignConfig | None = None,
                 score_cfg: ScoreConfig | None = None,
                 lang: str = "de") -> dict:
    """One gold item -> prediction record + alignment detail."""
    text_a, text_b = item["text_a"], item["text_b"]
    tokens_a, segs_a = segment_text(text_a)
    tokens_b, segs_b = segment_text(text_b)

    a_texts = [segment_text_str(tokens_a, s, text_a) for s in segs_a]
    b_texts = [segment_text_str(tokens_b, s, text_b) for s in segs_b]

    sim = judge.similarity_matrix(a_texts, b_texts)
    ops = align(sim, align_cfg)

    judgments: list[Judgment | None] = []
    for op in ops:
        if op.a_start < 0 or op.b_start < 0:
            judgments.append(None)  # asymmetric op: no pair to judge
            continue
        pa = " ".join(a_texts[op.a_start:op.a_end])
        pb = " ".join(b_texts[op.b_start:op.b_end])
        judgments.append(judge.judge_pair(pa, pb, lang_a="en", lang_b=lang))

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
            "difference": j.difference, "spans_a": j.spans_a,
            "spans_b": j.spans_b, "ok": j.ok, "repairs": j.repairs,
            "cached": j.cached} for j in judgments],
        "coverage": cov,
        "stats": {"parse_failures": stats.parse_failures,
                  "repairs": stats.repairs,
                  "span_matched": stats.span_matched,
                  "span_unmatched": stats.span_unmatched,
                  "ops": stats.ops,
                  "unparseable_similarity": judge.unparseable_similarity},
    }
    record = {
        "id": item["id"], "text_a": text_a, "text_b": text_b,
        "labels_a": labels_a, "labels_b": labels_b,
    }
    return {"record": record, "detail": detail}


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
        # parse with the vendored official parser (same semantics as upstream)
        try:
            from .judge import extract_first_json
            data = extract_first_json(text) or {}
        except Exception:
            data = {}
        s1, s2 = data.get("sentence1"), data.get("sentence2")
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
    detail = {"mode": "doc_baseline", "cached": cached, "repairs": repairs,
              "sentence1_emitted": len(data.get("sentence1") or []),
              "sentence2_emitted": len(data.get("sentence2") or [])}
    return {"record": record, "detail": detail}


def make_judge(backend: str, results_dir: Path, split: str, item_id: str,
               seed: int = 0) -> tuple[Judge, str]:
    client, backend_name = client_from_env_or_mock(backend, seed=seed)
    cache = DiskCache(Path(results_dir) / "cache")
    logger = CallLogger(Path(results_dir) / "calls.jsonl")
    return Judge(client=client, backend=backend_name, cache=cache,
                 logger=logger, split=split, item_id=item_id, seed=seed), backend_name


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
