"""Map aligned-pair judgments back onto the official token arrays.

* ``1:1``/``1:2`` pairs (v3-tag): tokens the judge flagged by INDEX get
  label 1.0, every other covered token gets 0.0 — localization is the
  judge's own token-level output, not a uniform per-pair score. Flagged
  punctuation indices are dropped and counted (official -1 convention
  applies to them regardless).
* ``1:0`` (omitted) / ``0:1`` (added): covered tokens get the calibrated
  asymmetric label — additions/omissions are represented, not dropped.
* Punctuation tokens are always -1 (official convention).
* Judgments invalid after the bounded repair fall back to a DECLARED
  score of 0 for every covered token and are counted (``invalid_pairs``,
  ``invalid_fallback_tokens``) — never reported as emitted/valid coverage.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field

from .align import AlignOp, OP_ADD, OP_MATCH, OP_OMIT, OP_SPLIT
from .judge import Judgment, TagJudgment
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
    parse_failures: int = 0        # legacy field: v3 counts invalid_pairs
    repairs: int = 0
    span_unmatched: int = 0        # spans the model emitted but we could not place
    span_matched: int = 0
    ops: dict = field(default_factory=dict)
    # v3-tag accounting
    flagged_a: int = 0             # valid flagged token labels (score 1.0)
    flagged_b: int = 0
    dropped_punct_ids: int = 0     # flagged indices dropped as punctuation
    invalid_pairs: int = 0         # pairs invalid after bounded repair
    invalid_fallback_tokens: int = 0   # tokens under the score-0 fallback
    matched_tokens: int = 0        # tokens covered by matched ops
    valid_matched_tokens: int = 0  # matched tokens under a VALID judgment


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
    matched_idx: set = set()   # side-keyed: a indices raw, b indices + len(tokens_a)
    valid_idx: set = set()
    invalid_idx: set = set()

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

        # matched pair (1:1, 1:2, 2:1) — v3-tag indexed judgment
        matched_idx.update(a_tok_idx)
        matched_idx.update(len(tokens_a) + i for i in b_tok_idx)
        if not isinstance(jud, TagJudgment) or not jud.ok:
            # declared score-0 fallback — counted, never emitted coverage
            stats.parse_failures += 1
            stats.invalid_pairs += 1
            invalid_idx.update(a_tok_idx)
            invalid_idx.update(len(tokens_a) + i for i in b_tok_idx)
            for t in a_tok_idx:
                labels_a[t] = 0.0
                seen_a[t] = True
            for t in b_tok_idx:
                labels_b[t] = 0.0
                seen_b[t] = True
            continue

        stats.repairs += jud.repairs
        valid_idx.update(a_tok_idx)
        valid_idx.update(len(tokens_a) + i for i in b_tok_idx)
        _apply_tags(labels_a, seen_a, tokens_a, a_tok_idx, jud.a_ids, stats,
                    "flagged_a")
        _apply_tags(labels_b, seen_b, tokens_b, b_tok_idx, jud.b_ids, stats,
                    "flagged_b")

    stats.matched_tokens = len(matched_idx)
    stats.valid_matched_tokens = len(valid_idx)
    stats.invalid_fallback_tokens = len(invalid_idx - valid_idx)

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


def _apply_tags(labels, seen, tokens, tok_idx, ids, stats, field_name):
    """v3-tag: flagged local indices -> 1.0, all other covered tokens -> 0.0.

    Flagged punctuation indices are dropped and counted — the official -1
    convention owns them regardless of what the model returned.
    """
    flagged = set()
    for i in ids:
        tok = tokens[tok_idx[i]]
        if is_punctuation(tok.text):
            stats.dropped_punct_ids += 1
            continue
        flagged.add(i)
    for pos, t in enumerate(tok_idx):
        if pos in flagged:
            labels[t] = 1.0
            setattr(stats, field_name, getattr(stats, field_name) + 1)
        else:
            labels[t] = 0.0
        seen[t] = True


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
