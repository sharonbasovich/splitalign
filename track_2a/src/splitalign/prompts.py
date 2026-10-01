"""Prompt templates (version-pinned via splitalign.PROMPT_VERSION).

All prompts carry a `"kind"` tag inside the JSON request block so the mock
backend can dispatch deterministically; real backends just see natural
instructions plus the same JSON payload.
"""
from __future__ import annotations

import json

SIM_SYSTEM = (
    "You are a bilingual corpus aligner for Swiss federal administration "
    "texts (English, German, French, Italian). Rate how well each source "
    "sentence matches each candidate target sentence in meaning. Answer "
    "only with JSON."
)

SIM_USER = """Rate the semantic similarity of each numbered sentence pair on a 0-5 scale:
5 = same meaning (translation-equivalent), 4 = minor divergence,
3 = same topic, partial overlap, 2 = related but clearly different content,
1 = mostly unrelated, 0 = unrelated / boilerplate.

Request:
```json
{request}
```

Return ONLY a JSON object of the form
{{"scores": [{{"i": <a-index>, "j": <b-index>, "score": <0-5>}}, ...]}}
with one entry per pair in the request."""

JUDGE_SYSTEM = (
    "You are a precise cross-lingual semantic difference annotator for Swiss "
    "government documents (English vs. {lang_name}). You judge how much the "
    "content of two aligned passages differs. Answer only with JSON."
)

JUDGE_USER = """Judge the semantic difference between passage A ({lang_a}) and passage B ({lang_b}).

Scale (0-5): 0 = fully equivalent meaning; 1 = trivial differences;
2 = minor differences in detail or emphasis; 3 = partially different
content; 4 = mostly different content; 5 = unrelated / no correspondence.

Also report the exact verbatim spans (copied character-for-character from
the passage) on each side whose meaning differs from the other side
(added, omitted or changed content). Use short spans, not whole sentences.
If there is no difference, return empty lists.

Request:
```json
{request}
```

Return ONLY a JSON object:
{{"semantic_difference": <0-5>,
  "differing_spans": {{"side_a": ["<verbatim span>", ...],
                      "side_b": ["<verbatim span>", ...]}}}}"""

BASELINE_SYSTEM = (
    "You annotate bilingual document pairs token-by-token. For every token, "
    "rate semantic similarity to the tokens of the other document: "
    "5 = complete equivalence, 3-4 = very/closely similar, 1-2 = slightly or "
    "somewhat similar, 0 = no related token, -1 = punctuation. "
    "Preserve the provided tokenization exactly. Answer only with JSON."
)

BASELINE_USER = """Annotate this document pair.

Request:
```json
{request}
```

Return ONLY a JSON object:
{{"sentence1": [["<token>", <label>], ...],
  "sentence2": [["<token>", <label>], ...]}}
containing exactly one [token, label] entry per input token, in order."""


def similarity_request(pairs: list[dict]) -> dict:
    """pairs: [{i, j, a, b}] -> request payload (kind-tagged)."""
    return {"kind": "pair_similarity", "pairs": pairs}


def judge_request(a_text: str, b_text: str, lang_a: str = "en",
                  lang_b: str = "de") -> dict:
    return {"kind": "judge_pair", "lang_a": lang_a, "lang_b": lang_b,
            "a": a_text, "b": b_text}


def baseline_request(tokens_a: list[str], tokens_b: list[str]) -> dict:
    return {"kind": "doc_baseline",
            "sentence1": json.dumps(tokens_a, ensure_ascii=False),
            "sentence2": json.dumps(tokens_b, ensure_ascii=False)}


def messages(system: str, user: str) -> list[dict]:
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]
