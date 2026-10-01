"""Map aligned-pair judgments back onto the official token arrays.

* ``1:1``/``1:2`` pairs: every covered token gets base = difference/5 (or the
  calibrated linear map), boosted where the model reported verbatim differing
  spans; asymmetric content inside the pair is therefore explicit.
* ``1:0`` (omitted) / ``0:1`` (added): covered tokens get the calibrated
  asymmetric label — additions/omissions are represented, not dropped.
* Punctuation tokens are always -1 (official convention).
* Failed judgments fall back to a similarity-derived score and are counted.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field

from .align import AlignOp, OP_ADD, OP_MATCH, OP_OMIT, OP_SPLIT
from .judge import Judgment
from .segment import Segment, Token, is_punctuation


@dataclass
class ScoreConfig:
    # map judgment d in 0..5 to label: label = a * (d/5) + b, clamped [0,1]
    diff_a: float = 1.0
    diff_b: float = 0.0
    span_boost: float = 0.25        # added to base for spanned tokens
    span_floor: float = 0.6         # spanned tokens are at least this
    omit_label: float = 0.9         # label for tokens in a 1:0 op
    add_label: float = 0.9          # label for tokens in a 0:1 op
    fallback_scale: float = 1.0     # (1-sim)*scale fallback on unparseable
    boilerplate_cap: float = 1.0    # max label for boilerplate-guard pairs


@dataclass
class ScoreStats:
    parse_failures: int = 0
    repairs: int = 0
    span_unmatched: int = 0        # spans the model emitted but we could not place
    span_matched: int = 0
    ops: dict = field(default_factory=dict)


def map_span_to_token_range(text: str, tokens: list[Token], span: str,
                            search_from: int = 0) -> tuple[int, int] | None:
    """Locate a verbatim span inside ``text`` -> token index range [lo, hi).

    Uses char offsets so repeated tokens map correctly. ``search_from`` is a
    char offset to start from (monotone searching preserves order for
    multiple spans in one segment).
    """
    pos = text.find(span, search_from)
    if pos < 0:
        pos = text.find(span)
        if pos < 0:
            return None
    end = pos + len(span)
    starts = [t.start for t in tokens]
    ends = [t.end for t in tokens]
    lo = bisect.bisect_right(starts, pos) - 1
    while lo < len(tokens) and tokens[lo].end <= pos:
        lo += 1
    hi = lo
    while hi < len(tokens) and tokens[hi].start < end:
        hi += 1
    if lo >= len(tokens) or hi <= lo:
        return None
    return (lo, hi)


def score_item(tokens_a: list[Token], tokens_b: list[Token],
               segs_a: list[Segment], segs_b: list[Segment],
               text_a: str, text_b: str,
               ops: list[AlignOp], judgments: list[Judgment | None],
               cfg: ScoreConfig | None = None) -> tuple[list[float], list[float], ScoreStats]:
    cfg = cfg or ScoreConfig()
    stats = ScoreStats()
    labels_a = [0.0] * len(tokens_a)
    labels_b = [0.0] * len(tokens_b)
    seen_a = [False] * len(tokens_a)
    seen_b = [False] * len(tokens_b)

    for op, jud in zip(ops, judgments):
        stats.ops[op.op] = stats.ops.get(op.op, 0) + 1
        a_tok_idx = _seg_token_range(segs_a, op.a_start, op.a_end)
        b_tok_idx = _seg_token_range(segs_b, op.b_start, op.b_end)

        if op.op == OP_OMIT:
            for t in a_tok_idx:
                labels_a[t] = cfg.omit_label
                seen_a[t] = True
            continue
        if op.op == OP_ADD:
            for t in b_tok_idx:
                labels_b[t] = cfg.add_label
                seen_b[t] = True
            continue

        # matched pair (1:1 or 1:2)
        if jud is None or not jud.ok:
            stats.parse_failures += 1
            base = min(1.0, (1.0 - op.sim) * cfg.fallback_scale)
            d_norm = base
            spans_a = spans_b = []
        else:
            stats.repairs += jud.repairs
            d_norm = max(0.0, min(1.0, cfg.diff_a * (jud.difference / 5.0) + cfg.diff_b))
            spans_a, spans_b = jud.spans_a, jud.spans_b

        _apply(labels_a, seen_a, tokens_a, text_a, a_tok_idx,
               d_norm, spans_a, cfg, stats, "a")
        _apply(labels_b, seen_b, tokens_b, text_b, b_tok_idx,
               d_norm, spans_b, cfg, stats, "b")

    # punctuation convention: -1 everywhere (mirrors official preprocessing)
    for i, t in enumerate(tokens_a):
        if is_punctuation(t.text):
            labels_a[i] = -1.0
            seen_a[i] = True
    for i, t in enumerate(tokens_b):
        if is_punctuation(t.text):
            labels_b[i] = -1.0
            seen_b[i] = True

    # safety net: any token never covered by an op (should not happen — the DP
    # covers every sentence) gets the neutral fallback, counted as failure.
    for i in range(len(tokens_a)):
        if not seen_a[i]:
            labels_a[i] = 0.0
    for i in range(len(tokens_b)):
        if not seen_b[i]:
            labels_b[i] = 0.0
    return labels_a, labels_b, stats


def _seg_token_range(segs: list[Segment], lo: int, hi: int) -> list[int]:
    out: list[int] = []
    for s in segs[lo:hi]:
        out.extend(range(s.start_token, s.end_token))
    return out


def _apply(labels, seen, tokens, text, tok_idx, base, spans, cfg, stats, side):
    seg_lo = min(tok_idx) if tok_idx else 0
    seg_hi = max(tok_idx) + 1 if tok_idx else 0
    boosted = set()
    for span in spans:
        rng = map_span_to_token_range(
            text, tokens, span,
            search_from=tokens[seg_lo].start if seg_lo < len(tokens) else 0)
        if rng is None:
            stats.span_unmatched += 1
            continue
        stats.span_matched += 1
        for t in range(max(rng[0], seg_lo), min(rng[1], seg_hi)):
            boosted.add(t)
    for t in tok_idx:
        val = base
        if t in boosted:
            val = max(cfg.span_floor, min(1.0, base + cfg.span_boost))
        labels[t] = min(cfg.boilerplate_cap, val)
        seen[t] = True
