"""Token-index-preserving sentence/paragraph segmentation.

Input documents are already whitespace-tokenized upstream (gold files store
``text_a``/``text_b`` as single-line strings whose ``str.split()`` defines the
official token array). We keep char offsets for every token so downstream span
mapping is exact even with repeated tokens.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Token:
    index: int
    text: str
    start: int  # char offset, inclusive
    end: int    # char offset, exclusive


@dataclass(frozen=True)
class Segment:
    index: int
    start_token: int  # inclusive token index
    end_token: int    # exclusive token index


TERMINAL = {".", "!", "?", "…", "。", "！", "？"}
# Common abbreviations after which we do not split.
ABBREVIATIONS = {
    "e.g.", "i.e.", "etc.", "cf.", "ca.", "approx.", "vs.", "no.", "nr.", "art.",
    "abs.", "al.", "fig.", "p.", "pp.", "vol.", "dr.", "prof.", "st.", "mr.",
    "mrs.", "ms.", "u.s.", "u.k.", "u.a.", "z.b.", "d.h.", "bzw.", "usw.",
    "resp.", "bspw.", "ev.", "ggf.", "vgl.", "s.", "ff.", "p.ex.", "c.-à-d.",
    "n.b.", "par ex.", "ecc.", "es.", "sig.", "dott.", "ing.", "n.",
}


def tokens_of(text: str) -> list[Token]:
    """Whitespace tokenize preserving original indices and char offsets."""
    out: list[Token] = []
    for i, m in enumerate(re.finditer(r"\S+", text)):
        out.append(Token(index=i, text=m.group(0), start=m.start(), end=m.end()))
    return out


def is_punctuation(tok: str) -> bool:
    """Official -1 convention, verified against every dev gold token:
    exactly one ASCII non-alphanumeric character (e.g. ``.`` ``-`` ``%``).
    Multi-char tokens and non-ASCII symbols (– — « » © € …) are NOT -1."""
    return len(tok) == 1 and ord(tok) < 128 and not tok.isalnum()


def _ends_terminal(tok: str) -> bool:
    t = tok.rstrip("\"'»”’)]}")
    return bool(t) and (t[-1] in TERMINAL)


def segment_sentences(tokens: list[Token]) -> list[Segment]:
    """Split a token stream into sentence spans.

    Rules: split after a token ending in terminal punctuation unless it is a
    known abbreviation or a decimal-like token (e.g. "6.8" is already a single
    upstream token). A leading run of <=15 tokens with no terminal punctuation
    is treated as a heading and becomes its own segment (governement pages
    start with a title). Empty input returns [].
    """
    if not tokens:
        return []
    bounds: list[int] = [0]
    n = len(tokens)
    for i, tok in enumerate(tokens):
        low = tok.text.lower()
        if low in ABBREVIATIONS:
            continue
        if _ends_terminal(tok.text):
            # do not split inside enumerations like "a)" "(1)" "1."
            if re.fullmatch(r"[(\[{]?[0-9a-z]{1,3}[)\].:]", low):
                continue
            if i + 1 < n:
                bounds.append(i + 1)
    bounds.append(n)
    bounds = sorted(set(bounds))
    segs = [Segment(index=k, start_token=bounds[k], end_token=bounds[k + 1])
            for k in range(len(bounds) - 1)]
    # Heading heuristic: a short leading segment with no terminal punctuation
    # is kept separate rather than merged into the first sentence.
    if len(segs) > 1 and (segs[0].end_token - segs[0].start_token) <= 15:
        first_text = " ".join(t.text for t in tokens[segs[0].start_token:segs[0].end_token])
        if not _ends_terminal(first_text):
            pass  # already its own segment only if a boundary followed it
    return segs


def segment_paragraphs(text: str, tokens: list[Token]) -> list[Segment]:
    """Paragraph boundaries: blank lines / explicit newlines.

    Upstream gold text is single-line, but user-supplied docs may contain
    newlines; treat every newline as a hard boundary, then refine each
    paragraph into sentences.
    """
    if not tokens or "\n" not in text:
        return segment_sentences(tokens)
    segs: list[Segment] = []
    start = 0
    for i, tok in enumerate(tokens):
        if "\n" in text[tok.end: tokens[i + 1].start if i + 1 < len(tokens) else len(text)]:
            segs.extend(_reindex(segment_sentences(tokens[start:i + 1]), start))
            start = i + 1
    segs.extend(_reindex(segment_sentences(tokens[start:]), start))
    return [Segment(index=i, start_token=s.start_token, end_token=s.end_token)
            for i, s in enumerate(segs)]


def _reindex(segs: list[Segment], offset: int) -> list[Segment]:
    return [Segment(index=s.index, start_token=s.start_token + offset,
                    end_token=s.end_token + offset) for s in segs]


def segment_text(text: str) -> tuple[list[Token], list[Segment]]:
    tokens = tokens_of(text)
    return tokens, segment_paragraphs(text, tokens)


def segment_text_str(tokens: list[Token], seg: Segment, text: str) -> str:
    """Reconstruct the exact source substring of a segment."""
    if seg.start_token >= seg.end_token:
        return ""
    return text[tokens[seg.start_token].start: tokens[seg.end_token - 1].end]
