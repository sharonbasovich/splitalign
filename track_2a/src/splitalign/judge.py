"""Apertus-backed similarity + judgment calls, with caching and robust
JSON extraction. All calls flow through ``invoke`` which handles cache
lookup, retries-on-malformed, call logging and truthful failure stats.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from . import PROMPT_VERSION, PROMPT_VERSION_BY_KIND
from .apertus import ChatResult, CallLogger, client_from_env_or_mock
from .cache import DiskCache, payload_hash
from . import prompts


@dataclass
class Judgment:
    difference: int            # 0..5, 5 = unrelated
    spans_a: list[str]
    spans_b: list[str]
    ok: bool = True            # False when response had to be repaired/failed
    repairs: int = 0
    cached: bool = False
    raw: str = ""
    lang_b: str = ""           # target language the judge prompt was written for


@dataclass
class Judge:
    client: object
    backend: str
    cache: DiskCache
    logger: CallLogger
    split: str
    item_id: str
    seed: int = 0
    max_repairs: int = 1
    unparseable_similarity: int = 0
    budget: object = None       # ApiBudget | None — caps NEW calls only

    # -- low level ----------------------------------------------------------
    def _invoke(self, *, kind: str, messages: list[dict],
                payload_for_key, max_tokens: int = 2048) -> tuple[str, bool, ChatResult | None]:
        key = self.cache.key(
            backend=self.backend,
            model=getattr(self.client, "model", "?"),
            prompt_version=PROMPT_VERSION_BY_KIND.get(kind, PROMPT_VERSION),
            kind=kind, split=self.split, item_id=self.item_id,
            payload=payload_for_key,
        )
        hit = self.cache.get(key)
        if hit is not None:
            self.logger.log(split=self.split, item_id=self.item_id, kind=kind,
                            messages=messages, result=None, ok=True,
                            error=None, cached=True,
                            backend=hit.get("backend") or self.backend,
                            model=hit.get("model") or getattr(self.client, "model", None))
            return hit["text"], True, None
        if self.budget is not None:
            self.budget.logical_calls += 1  # non-cached call; HTTP attempts
            # are capped inside ApertusClient.complete (retries included)
        try:
            res = self.client.complete(messages, max_tokens=max_tokens,
                                       temperature=0.0, seed=self.seed)
        except Exception:
            self.logger.log(split=self.split, item_id=self.item_id, kind=kind,
                            messages=messages, result=None, ok=False,
                            error="call failed", cached=False,
                            backend=self.backend,
                            model=getattr(self.client, "model", None))
            raise
        self.logger.log(split=self.split, item_id=self.item_id, kind=kind,
                        messages=messages, result=res, ok=True,
                        error=None, cached=False)
        self.cache.put(key, {"text": res.text, "model": res.model,
                             "backend": res.backend})
        return res.text, False, res

    # -- similarity matrix --------------------------------------------------
    # Banded candidate policy: cross-lingual administrative texts are near-
    # parallel, so a segment pair far off the diagonal never yields a match
    # that beats the calibrated omit/add costs. We therefore query only
    # pairs whose normalised positions are close —
    # | i/max(n-1,1) - j/max(m-1,1) | <= band_width (default 0.2) —
    # plus an absolute ±2 neighbourhood for short documents. All cells
    # outside the band are 0.0 without an API call.
    # MEASURED on dev/train+dev/val (504 docs, w=0.2):
    #   304,968 full pairs -> 108,260 candidates (35%); worst doc
    #   (admin_de_174, 12648 pairs) -> 4,518. One API call per ~20 pairs.
    def similarity_matrix(self, a_texts: list[str], b_texts: list[str],
                          batch: int = 20, band_width: float = 0.2) -> list[list[float]]:
        """n x m matrix of 0..1 similarities via batched pair scoring."""
        n, m = len(a_texts), len(b_texts)
        sim = [[0.0] * m for _ in range(n)]
        nn, mm = max(n - 1, 1), max(m - 1, 1)
        pairs = [{"i": i, "j": j, "a": a_texts[i], "b": b_texts[j]}
                 for i in range(n) for j in range(m)
                 if abs(i / nn - j / mm) <= band_width
                 or abs(i * m / n - j) <= 2]
        for k in range(0, len(pairs), batch):
            chunk = pairs[k:k + batch]
            req = prompts.similarity_request(chunk)
            msgs = prompts.messages(
                prompts.SIM_SYSTEM,
                prompts.SIM_USER.format(request=json.dumps(req, ensure_ascii=False)))
            text, _, _ = self._invoke(kind="pair_similarity", messages=msgs,
                                      payload_for_key=req, max_tokens=4096)
            scores = _parse_scores(text)
            for p in chunk:
                s = scores.get((p["i"], p["j"]))
                if s is None:
                    self.unparseable_similarity += 1
                    continue
                sim[p["i"]][p["j"]] = max(0.0, min(1.0, s / 5.0))
        return sim

    # -- pair judgment ------------------------------------------------------
    def judge_pair(self, a_text: str, b_text: str, *,
                   lang_a: str, lang_b: str) -> Judgment:
        """lang_b is required: the prompt names the target language."""
        req = prompts.judge_request(a_text, b_text, lang_a, lang_b)
        lang_names = {"de": "German", "fr": "French", "it": "Italian", "en": "English"}
        msgs = prompts.messages(
            prompts.JUDGE_SYSTEM.format(lang_name=lang_names.get(lang_b, lang_b)),
            prompts.JUDGE_USER.format(lang_a=lang_names.get(lang_a, lang_a),
                                      lang_b=lang_names.get(lang_b, lang_b),
                                      request=json.dumps(req, ensure_ascii=False)))
        repairs = 0
        while True:
            text, cached, _ = self._invoke(kind="judge_pair", messages=msgs,
                                           payload_for_key={"req": req, "r": repairs},
                                           max_tokens=1024)
            j = parse_judgment(text, a_text, b_text)
            if j is not None:
                j.cached = cached
                j.repairs = repairs
                j.raw = text[:2000]
                j.lang_b = lang_b
                return j
            if repairs >= self.max_repairs:
                return Judgment(difference=-1, spans_a=[], spans_b=[], ok=False,
                                repairs=repairs, cached=cached, raw=text[:2000],
                                lang_b=lang_b)
            repairs += 1
            msgs = msgs + [{"role": "assistant", "content": text},
                           {"role": "user", "content":
                            "That was not valid JSON. Respond with ONLY the JSON object."}]


# --------------------------------------------------------------------------
# Response parsing
# --------------------------------------------------------------------------

def extract_first_json(text: str) -> dict | None:
    """Extract the first balanced {...} object; tolerant of fences/prose."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"```$", "", t).strip()
    start = t.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for k in range(start, len(t)):
        c = t[k]
        if esc:
            esc = False
            continue
        if c == "\\" and in_str:
            esc = True
            continue
        if c == '"':
            in_str = not in_str
        elif not in_str:
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[start:k + 1])
                    except json.JSONDecodeError:
                        return None
    return None


def parse_judgment(text: str, a_text: str, b_text: str) -> Judgment | None:
    obj = extract_first_json(text)
    if obj is None:
        return None
    d = obj.get("semantic_difference")
    if not isinstance(d, (int, float)) or isinstance(d, bool):
        return None
    d = int(round(float(d)))
    if d < 0 or d > 5:
        return None
    spans = obj.get("differing_spans", {})
    if not isinstance(spans, dict):
        return None
    def clean(side: str, hay: str) -> list[str]:
        out = []
        for s in spans.get(side, []) or []:
            if isinstance(s, str) and s and s in hay:
                out.append(s)
        return out
    return Judgment(difference=d,
                    spans_a=clean("side_a", a_text),
                    spans_b=clean("side_b", b_text))


def _parse_scores(text: str) -> dict[tuple[int, int], float]:
    obj = extract_first_json(text)
    if obj is None or not isinstance(obj.get("scores"), list):
        return {}
    out: dict[tuple[int, int], float] = {}
    for e in obj["scores"]:
        if not isinstance(e, dict):
            continue
        i, j, s = e.get("i"), e.get("j"), e.get("score")
        if isinstance(i, int) and isinstance(j, int) and isinstance(s, (int, float)):
            out[(i, j)] = float(s)
    return out
