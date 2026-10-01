"""Metric-specification helpers — original Apache-2.0 implementation.

These functions implement the documented SwissGov-RSD token-label
semantics (Wastl, Vamvas & Sennrich, ACL 2026; task description at
https://hackapertus.notion.site/track-2a-uzh). They are written from the
published behavior specification, not copied from upstream source; the
upstream evaluation scripts were inspected only for semantics and are not
redistributed here (upstream ships no license covering them).

Specification implemented:

* Gold JSONL items carry ``text_a``/``text_b`` as single-line
  whitespace-tokenized strings plus per-token ``labels_a``/``labels_b``
  in {0.0..1.0} with -1 marking punctuation (filtered before scoring).
* Prompted-LLM responses have the form
  ``{"sentence1": [[token, similarity], ...], "sentence2": [...]}`` with
  similarity on a 0-5 scale. The reference harness walks the emitted
  (token, label) pairs in order: an emitted token matching the *current*
  gold position consumes that position and assigns its label; a token
  matching the *next* position skips the current one and assigns at the
  next; anything else is ignored. Positions never matched keep
  ``fallback_label``; surplus emissions past the token count are ignored;
  a non-numeric label leaves the fallback in place and — matching the
  reference behavior — does not advance the cursor on a current-position
  match but does keep the one-step advance on a next-position match.
* Similarity ``s`` maps to a difference label by ``1 - s/5`` for
  ``s >= 0``; ``-1`` stays ``-1``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DifferenceSample:
    """One document pair with its per-token gold labels."""
    tokens_a: tuple
    tokens_b: tuple
    labels_a: tuple
    labels_b: tuple
    annotator_tag: int | None = None

    @property
    def sentence1(self):
        return " ".join(self.tokens_a)

    @property
    def sentence2(self):
        return " ".join(self.tokens_b)


def tokenize(text: str) -> tuple[str, ...]:
    """Whitespace tokenization — gold texts are already Moses-tokenized."""
    return tuple(text.split())


def load_gold_data(gold_path: Path) -> list[DifferenceSample]:
    """Load a gold JSONL file into difference samples."""
    samples = []
    for line in Path(gold_path).read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        samples.append(DifferenceSample(
            tokens_a=tokenize(item["text_a"]),
            tokens_b=tokenize(item["text_b"]),
            labels_a=tuple(item["labels_a"]),
            labels_b=tuple(item["labels_b"]),
            annotator_tag=item.get("annotator_tag"),
        ))
    return samples


def parse_token_labels(tokens: list[str], token_predictions,
                       fallback_label: float = 0.0) -> list[float]:
    """Assign emitted (token, similarity) pairs onto gold token positions.

    Returns one label per gold token; unmatched positions keep
    ``fallback_label``. See module docstring for the matching semantics.
    """
    labels = [fallback_label] * len(tokens)
    try:
        predictions = list(token_predictions)
    except TypeError:
        return labels
    i = 0
    for entry in predictions:
        if i >= len(tokens):
            break
        if isinstance(entry, str):
            emitted, value = entry, None
        else:
            try:
                entry = list(entry)
            except TypeError:
                continue
            if not entry:
                continue
            emitted = entry[0]
            value = entry[1] if len(entry) > 1 else None
        if emitted == tokens[i]:
            pass
        elif i + 1 < len(tokens) and emitted == tokens[i + 1]:
            i += 1
        else:
            continue
        if value is not None:
            try:
                labels[i] = float(value)
            except (TypeError, ValueError):
                continue  # cursor not advanced on current-position match
        i += 1
    return labels


def map_label_from_positive_to_negative(label: float) -> float:
    """Similarity 0..5 -> difference 1.0..0.0; -1 stays -1."""
    return 1.0 - (label / 5.0) if label >= 0 else -1.0
