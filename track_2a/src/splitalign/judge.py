"""Apertus-backed similarity + judgment calls, with caching and robust
JSON extraction. All calls flow through ``invoke`` which handles cache
lookup, retries-on-malformed, call logging and truthful failure stats.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from . import PROMPT_VERSION, PROMPT_VERSION_BY_KIND, METHOD_VERSIONS
from .apertus import (ApertusUnavailable, BoundNotConfigured, BudgetExceeded,
                      ChatResult, CallLogger, client_from_env_or_mock)
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
class TagJudgment:
    """v3-tag judge result: exact token indices judged differing.

    ``a_ids``/``b_ids`` are 0-based positions inside the token arrays sent
    to the model. ``ok=False`` means the response stayed invalid after the
    bounded repair — the pair falls back to a DECLARED score of 0 and is
    counted, never treated as emitted/valid coverage.
    """
    a_ids: list[int]
    b_ids: list[int]
    ok: bool = True
    repairs: int = 0
    cached: bool = False
    raw: str = ""
    lang_b: str = ""


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
    method: str | None = None   # caller-set method tag for attempt logs

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
            # are reserved/capped inside ApertusClient.complete
        try:
            res = self.client.complete(messages, max_tokens=max_tokens,
                                       temperature=0.0, seed=self.seed)
        except (BudgetExceeded, BoundNotConfigured) as e:
            # predispatch block: zero bytes left — NOT an outbound attempt
            self.logger.log(split=self.split, item_id=self.item_id, kind=kind,
                            messages=messages, result=None, ok=False,
                            error=type(e).__name__, cached=False,
                            backend=self.backend,
                            model=getattr(self.client, "model", None),
                            method=self.method, dispatched=False,
                            reserved=getattr(self.client, "last_reservation",
                                             None))
            raise
        except Exception as e:
            # real outbound attempt that failed; usage attached to the
            # exception (e.g. bound violation) is known and preserved
            usage = e.usage if isinstance(e, ApertusUnavailable) else None
            self.logger.log(split=self.split, item_id=self.item_id, kind=kind,
                            messages=messages, result=None, ok=False,
                            error=type(e).__name__, cached=False,
                            backend=self.backend,
                            model=getattr(self.client, "model", None),
                            method=self.method, dispatched=True,
                            reserved=getattr(self.client, "last_reservation",
                                             None),
                            usage=usage)
            raise
        self.logger.log(split=self.split, item_id=self.item_id, kind=kind,
                        messages=messages, result=res, ok=True,
                        error=None, cached=False, method=self.method,
                        dispatched=True,
                        reserved=getattr(self.client, "last_reservation",
                                         None))
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
    #
    # Missing/unparseable cells are NOT silently treated as evidence of
    # omission: the batch gets one bounded repair, then remaining missing
    # cells are marked ``unknown`` and filled with a deterministic neutral
    # prior — mean of the observed cells in the same row, else same column,
    # else ``unknown_prior`` — labelled prior-imputed and counted.
    def similarity_matrix(self, a_texts: list[str], b_texts: list[str],
                          batch: int = 20, band_width: float = 0.2,
                          unknown_prior: float = 0.5) -> list[list[float]]:
        """n x m matrix of 0..1 similarities via batched pair scoring.

        Unknown/imputed accounting is on ``self.sim_unknown`` (mask),
        ``self.sim_unknown_cells`` / ``self.sim_imputed_cells`` (counts) and
        ``self.unparseable_similarity`` (kept for backwards reporting).
        """
        n, m = len(a_texts), len(b_texts)
        sim = [[0.0] * m for _ in range(n)]
        self.sim_unknown = [[False] * m for _ in range(n)]
        self.sim_unknown_cells = 0
        self.sim_imputed_cells = 0
        nn, mm = max(n - 1, 1), max(m - 1, 1)
        pairs = [{"i": i, "j": j, "a": a_texts[i], "b": b_texts[j]}
                 for i in range(n) for j in range(m)
                 if abs(i / nn - j / mm) <= band_width
                 or abs(i * m / n - j) <= 2]
        observed = {}   # (i, j) -> parsed score 0..1
        for k in range(0, len(pairs), batch):
            chunk = pairs[k:k + batch]
            req = prompts.similarity_request(chunk)
            msgs = prompts.messages(
                prompts.SIM_SYSTEM,
                prompts.SIM_USER.format(request=json.dumps(req, ensure_ascii=False)))
            text, _, _ = self._invoke(kind="pair_similarity", messages=msgs,
                                      payload_for_key=req, max_tokens=4096)
            scores = _parse_scores(text)
            missing = [p for p in chunk if (p["i"], p["j"]) not in scores]
            if missing:
                # one bounded repair: ask for just the missing cells
                r_msgs = msgs + [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content":
                     "Some pairs had no score entry. Return ONLY a JSON "
                     "object with a \"scores\" list containing exactly the "
                     "missing (i, j) entries."}]
                r_req = prompts.similarity_request(missing)
                r_text, _, _ = self._invoke(
                    kind="pair_similarity", messages=r_msgs,
                    payload_for_key={"req": r_req, "r": 1},
                    max_tokens=4096)
                scores.update(_parse_scores(r_text))
            for p in chunk:
                s = scores.get((p["i"], p["j"]))
                if s is None:
                    self.unparseable_similarity += 1
                    self.sim_unknown[p["i"]][p["j"]] = True
                    self.sim_unknown_cells += 1
                    continue
                observed[(p["i"], p["j"])] = max(0.0, min(1.0, s / 5.0))
                sim[p["i"]][p["j"]] = observed[(p["i"], p["j"])]
        # deterministic neutral prior for unknown cells: observed row mean,
        # else observed column mean, else the declared default — imputed,
        # never model evidence.
        row_obs = {}
        col_obs = {}
        for (i, j), v in observed.items():
            row_obs.setdefault(i, []).append(v)
            col_obs.setdefault(j, []).append(v)
        for i in range(n):
            for j in range(m):
                if not self.sim_unknown[i][j]:
                    continue
                if row_obs.get(i):
                    prior = sum(row_obs[i]) / len(row_obs[i])
                elif col_obs.get(j):
                    prior = sum(col_obs[j]) / len(col_obs[j])
                else:
                    prior = unknown_prior
                sim[i][j] = prior
                self.sim_imputed_cells += 1
        return sim

    # -- token-tag judgment (v3) -------------------------------------------
    def judge_tag(self, a_tokens: list[str], b_tokens: list[str], *,
                  lang_a: str, lang_b: str) -> TagJudgment:
        """Indexed-token judge: returns exact differing token indices.

        ``lang_b`` is required and names the target language in the prompt.
        Invalid after one bounded repair -> ``ok=False`` (declared score-0
        fallback upstream, counted — never emitted coverage).
        """
        req = prompts.judge_tag_request(a_tokens, b_tokens, lang_a, lang_b)
        lang_names = {"de": "German", "fr": "French", "it": "Italian",
                      "en": "English"}
        msgs = prompts.messages(
            prompts.JUDGE_TAG_SYSTEM.format(
                lang_name=lang_names.get(lang_b, lang_b)),
            prompts.JUDGE_TAG_USER.format(
                lang_a=lang_names.get(lang_a, lang_a),
                lang_b=lang_names.get(lang_b, lang_b),
                request=json.dumps(req, ensure_ascii=False)))
        repairs = 0
        while True:
            text, cached, _ = self._invoke(
                kind="judge_tag", messages=msgs,
                payload_for_key={"req": req, "r": repairs}, max_tokens=1024)
            parsed = parse_tag_response(text, len(a_tokens), len(b_tokens))
            if parsed is not None:
                return TagJudgment(a_ids=parsed[0], b_ids=parsed[1], ok=True,
                                   repairs=repairs, cached=cached,
                                   raw=text[:2000], lang_b=lang_b)
            if repairs >= self.max_repairs:
                return TagJudgment(a_ids=[], b_ids=[], ok=False,
                                   repairs=repairs, cached=cached,
                                   raw=text[:2000], lang_b=lang_b)
            repairs += 1
            msgs = msgs + [{"role": "assistant", "content": text},
                           {"role": "user", "content":
                            "That was not valid JSON matching the required "
                            "schema (a_ids and b_ids, integer indices in "
                            "bounds, no duplicates). Respond with ONLY the "
                            "JSON object."}]

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


def parse_tag_response(text: str, n_a: int, n_b: int
                       ) -> tuple[list[int], list[int]] | None:
    """Validate a v3-tag response STRICTLY -> (a_ids, b_ids) or None.

    Both keys are required and may be empty. Every element must be a true
    integer (``bool`` is rejected, floats/strings rejected), in bounds, and
    no duplicates — duplicate or malformed entries invalidate the whole
    response rather than being silently repaired.
    """
    obj = extract_first_json(text)
    if obj is None or not isinstance(obj, dict):
        return None
    if "a_ids" not in obj or "b_ids" not in obj:
        return None
    out = []
    for key, bound in (("a_ids", n_a), ("b_ids", n_b)):
        vals = obj[key]
        if not isinstance(vals, list):
            return None
        ids = []
        for v in vals:
            if isinstance(v, bool) or not isinstance(v, int):
                return None
            if v < 0 or v >= bound:
                return None
            ids.append(v)
        if len(set(ids)) != len(ids):
            return None          # duplicates: reject, don't dedupe
        out.append(ids)
    return out[0], out[1]


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
