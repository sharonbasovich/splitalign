"""Apertus inference backends.

Two interchangeable clients behind one interface:

* :class:`ApertusClient` — real calls to an OpenAI-compatible hosted endpoint
  (CSCS / Swiss AI). Configured exclusively via environment variables so no
  secret ever lives in source, prompts, logs or the repo:

      APERTUS_API_BASE   e.g. https://<cscs-host>/v1   (OpenAI-compatible)
      APERTUS_API_KEY    bearer token
      APERTUS_MODEL      e.g. swiss-ai/Apertus-v1.5-8B
      APERTUS_TIMEOUT_S  optional, default 120
      APERTUS_RPS        optional max requests/second, default 2

  Missing configuration raises :class:`MissingCredentials` naming every
  expected variable — honest failure, never a silent fallback.

* :class:`MockApertusClient` — deterministic lexical heuristics used to
  exercise the full pipeline offline. Every record it produces is tagged
  backend="mock" and model="MOCK-NOT-APERTUS"; it must never be cited as
  sponsor/model performance.

Every call is appended to a JSONL usage log with split tags, latency and
token accounting (mock usage is clearly labelled as estimated).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


class MissingCredentials(RuntimeError):
    def __init__(self):
        super().__init__(
            "Apertus backend is not configured. Set these environment "
            "variables and retry: APERTUS_API_BASE (OpenAI-compatible base "
            "URL), APERTUS_API_KEY (bearer token), APERTUS_MODEL (model id, "
            "e.g. swiss-ai/Apertus-v1.5-8B). Optional: APERTUS_TIMEOUT_S, "
            "APERTUS_RPS. No credentials are bundled with this repo."
        )


class ApertusUnavailable(RuntimeError):
    pass


@dataclass
class ChatResult:
    text: str
    model: str
    backend: str            # "apertus" | "mock"
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    usage_estimated: bool = False


class CallLogger:
    """Append-only JSONL log of every model call (split- and run-tagged).

    ``run_id`` is written on every record (cache hits and errors included);
    records without it predate run scoping and are unattributable to a run.
    """

    def __init__(self, path: Path | None, run_id: str | None = None):
        self.path = Path(path) if path else None
        self.run_id = run_id

    def log(self, *, split: str, item_id: str, kind: str, messages,
            result: ChatResult | None, ok: bool, error: str | None,
            cached: bool, backend: str | None = None,
            model: str | None = None) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        prompt_text = "\n".join(m.get("content", "") for m in messages)
        rec = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "run_id": self.run_id,
            "split": split, "item_id": item_id, "kind": kind,
            # actual backend/model on EVERY record, incl. cache-hit/error paths
            "backend": (result.backend if result else backend),
            "model": (result.model if result else model),
            "prompt_sha256": hashlib.sha256(prompt_text.encode()).hexdigest(),
            "prompt_chars": len(prompt_text),
            "prompt_tokens": result.prompt_tokens if result else None,
            "completion_tokens": result.completion_tokens if result else None,
            "latency_ms": result.latency_ms if result else None,
            "usage_estimated": result.usage_estimated if result else None,
            "cached": cached,
            "ok": ok,
            "error": error,
        }
        with self.path.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------
# Real backend
# --------------------------------------------------------------------------

class ApertusClient:
    backend = "apertus"

    def __init__(self, base: str, key: str, model: str,
                 timeout_s: float = 120.0, rps: float = 2.0,
                 max_retries: int = 0):
        self.base = base.rstrip("/")
        self.key = key
        self.model = model
        self.timeout_s = timeout_s
        self.min_interval = 1.0 / rps if rps > 0 else 0.0
        self.max_retries = 0  # kept for signature compat; retries are
        # disabled: any failed attempt has unknown token cost => halt
        self._last_call = 0.0
        # optional shared ApiBudget; set post-construction by the caller so
        # the SAME object governs every attempt across both methods
        self.budget = None

    @classmethod
    def from_env(cls) -> "ApertusClient":
        base = os.environ.get("APERTUS_API_BASE")
        key = os.environ.get("APERTUS_API_KEY")
        model = os.environ.get("APERTUS_MODEL")
        if not (base and key and model):
            raise MissingCredentials()
        return cls(
            base=base, key=key, model=model,
            timeout_s=float(os.environ.get("APERTUS_TIMEOUT_S", "120")),
            rps=float(os.environ.get("APERTUS_RPS", "2")),
        )

    def _throttle(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)

    @staticmethod
    def _prompt_token_bound(messages) -> int:
        """Verified upper bound on prompt tokens (byte-level BPE: every token
        decodes to >= 1 byte) plus a flat 256-token chat-template/special-token
        allowance. NOT an estimate — a conservative bound."""
        return sum(len(str(m.get("content", "")).encode("utf-8"))
                   for m in messages) + 256

    @staticmethod
    def _valid_usage(payload) -> tuple[int, int] | None:
        """Return (prompt_tokens, completion_tokens) only when usage carries
        both fields as nonnegative true integers — anything else means the
        attempt's token cost is UNKNOWN (never treated as zero)."""
        u = payload.get("usage")
        if not isinstance(u, dict):
            return None
        out = []
        for k in ("prompt_tokens", "completion_tokens"):
            v = u.get(k)
            if not (isinstance(v, int) and not isinstance(v, bool) and v >= 0):
                return None
            out.append(v)
        return out[0], out[1]

    def complete(self, messages, *, max_tokens: int = 2048,
                 temperature: float = 0.0, seed: int | None = None) -> ChatResult:
        body: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            body["seed"] = seed
        data = json.dumps(body).encode()
        url = f"{self.base}/chat/completions"
        # Atomic predispatch reservation: one request slot plus the verified
        # upper bound on this request's prompt + bounded completion. If either
        # cap would be exceeded the reservation raises BEFORE any bytes leave
        # — an exhausted budget permits zero dispatch, and the bound is a
        # bound, never an estimate presented as a maximum.
        reserved = 0
        if self.budget is not None:
            reserved = self.budget.reserve(
                self._prompt_token_bound(messages) + max_tokens)
        self._throttle()
        t0 = time.monotonic()
        req = urllib.request.Request(
            url, data=data, method="POST",
            headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {self.key}"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                payload = json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            self._last_call = time.monotonic()
            if self.budget is not None:
                self.budget.fail_unknown()
            raise ApertusUnavailable(
                f"Apertus HTTP {e.code} — attempt consumed, token cost "
                f"UNKNOWN; run halts (no retry after unknown cost)") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if self.budget is not None:
                self.budget.fail_unknown()
            raise ApertusUnavailable(
                f"Apertus unreachable ({e}) — attempt consumed, token cost "
                f"UNKNOWN; run halts (no retry after unknown cost)") from e
        except (json.JSONDecodeError, ValueError, UnicodeDecodeError) as e:
            if self.budget is not None:
                self.budget.fail_unknown()
            raise ApertusUnavailable(
                f"Apertus response body not valid JSON — attempt consumed, "
                f"token cost UNKNOWN; run halts") from e
        self._last_call = time.monotonic()
        try:
            text = payload["choices"][0]["message"]["content"]
            if not isinstance(text, str):
                raise TypeError("content is not a string")
        except (KeyError, IndexError, TypeError) as e:
            if self.budget is not None:
                self.budget.fail_unknown()
            raise ApertusUnavailable(
                f"Apertus response missing choices/content ({e}) — attempt "
                f"consumed, token cost UNKNOWN; run halts") from e
        usage = self._valid_usage(payload)
        if usage is None:
            if self.budget is not None:
                self.budget.fail_unknown()
            raise ApertusUnavailable(
                "Apertus response missing/malformed usage — attempt "
                "consumed, token cost UNKNOWN; run halts")
        pt, ct = usage
        if self.budget is not None:
            self.budget.reconcile(reserved, pt, ct)
        return ChatResult(
            text=text,
            model=payload.get("model", self.model),
            backend=self.backend,
            prompt_tokens=pt,
            completion_tokens=ct,
            latency_ms=(time.monotonic() - t0) * 1000.0,
        )


# --------------------------------------------------------------------------
# Deterministic mock backend (offline development only — never a result)
# --------------------------------------------------------------------------

_COGNATES = {
    # small static seed lexicon so the mock produces non-trivial cross-lingual
    # signal for pipeline testing; this is NOT translation and NOT Apertus.
    "the": {"der", "die", "das", "le", "la", "les", "il", "lo", "i", "gli"},
    "of": {"de", "von", "der", "des", "di", "del", "della", "dei"},
    "and": {"und", "et", "e"},
    "in": {"in", "im", "dans", "en", "nel", "nella", "au"},
    "energy": {"energie", "énergie", "energia"},
    "switzerland": {"schweiz", "suisse", "svizzera"},
    "policy": {"politik", "politique", "politica"},
    "supply": {"versorgung", "approvisionnement", "fornitura"},
    "law": {"gesetz", "loi", "legge"},
    "federal": {"bundes", "fédéral", "federale"},
    "office": {"amt", "office", "ufficio", "bureau"},
    "water": {"wasser", "eau", "acqua"},
    "health": {"gesundheit", "santé", "salute"},
    "environment": {"umwelt", "environnement", "ambiente"},
    "international": {"internationale", "international", "internazionale"},
    "percent": {"prozent", "pour", "cent", "per"},
}


def _norm_tokens(text: str) -> list[str]:
    return [t.lower().strip(".,;:!?\"'()[]{}«»“”") for t in text.split()
            if t.strip(".,;:!?\"'()[]{}«»“”")]


def lexical_similarity(a: str, b: str) -> float:
    """Dice-ish overlap on normalized tokens with a tiny static lexicon."""
    ta, tb = set(_norm_tokens(a)), set(_norm_tokens(b))
    if not ta or not tb:
        return 0.0
    overlap = 0.0
    for t in ta:
        if t in tb:
            overlap += 1.0
        elif re.fullmatch(r"\d+([.,]\d+)?%?", t):
            continue  # numbers handled below
        else:
            overlap += 0.9 if tb & _COGNATES.get(t, set()) else 0.0
    nums_a = {t for t in ta if re.fullmatch(r"\d+([.,]\d+)?%?", t)}
    nums_b = {t for t in tb if re.fullmatch(r"\d+([.,]\d+)?%?", t)}
    num_shared = len(nums_a & nums_b)
    overlap += num_shared + 0.5 * min(len(nums_a), len(nums_b)) - 0.5 * num_shared \
        if nums_a or nums_b else 0.0
    return max(0.0, min(1.0, 2 * overlap / (len(ta) + len(tb))))


class BudgetExceeded(RuntimeError):
    """Noncached API budget exhausted — the run stops cleanly."""


@dataclass
class ApiBudget:
    """Hard cap on NEW (noncached) API spend across a whole run, enforced by
    ATOMIC PREDISPATCH RESERVATION inside ``ApertusClient.complete``.

    Before any bytes leave, ``reserve`` atomically (under a lock) claims one
    request slot plus a VERIFIED conservative upper bound on that request's
    prompt tokens (UTF-8-byte bound on serialized message contents + flat
    template allowance) and bounded completion (``max_tokens``). Either cap
    exceeded ⇒ ``BudgetExceeded`` before dispatch: an exhausted budget sends
    zero requests.

    After the response, ``reconcile`` releases the unused part of the
    reservation only when provider ``usage`` is present AND valid (both
    fields nonnegative true integers). Otherwise ``fail_unknown`` RETAINS
    the reservation — the worst case stays committed — and the caller halts:
    no retry is ever attempted after an unknown-cost attempt. ``tokens`` is
    therefore committed spend (measured + still-held reservations);
    ``measured_tokens`` is the provider-reported subset.

    Cache hits never consume budget — reuse is allowed only when the cache
    key (backend|model|prompt_version|kind|split|item|payload) matches
    exactly, which DiskCache already enforces.
    """
    max_requests: int = 500
    max_tokens: int = 600_000
    requests: int = 0            # reserved == dispatched attempts
    tokens: int = 0              # committed: measured + held reservations
    measured_tokens: int = 0     # provider-reported usage only
    attempts_no_usage: int = 0   # dispatched attempts with unknown token cost
    logical_calls: int = 0       # non-cached judge calls

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def reserve(self, bound_tokens: int) -> int:
        """Atomically hold 1 attempt + ``bound_tokens``; raises BEFORE dispatch."""
        with self._lock:
            if self.requests >= self.max_requests:
                raise BudgetExceeded(
                    f"request cap reached ({self.requests}/{self.max_requests})")
            if self.tokens + bound_tokens > self.max_tokens:
                raise BudgetExceeded(
                    f"token cap would be exceeded by reserved bound "
                    f"({self.tokens}+{bound_tokens}/{self.max_tokens})")
            self.requests += 1
            self.tokens += bound_tokens
        return bound_tokens

    def reconcile(self, reserved: int, prompt_tokens: int,
                  completion_tokens: int) -> None:
        """Valid usage received: charge actual, release the unused reserve."""
        with self._lock:
            actual = prompt_tokens + completion_tokens
            self.tokens += actual - reserved
            self.measured_tokens += actual

    def fail_unknown(self) -> None:
        """Dispatched attempt whose cost is UNKNOWN: keep the reservation
        held (worst case stays committed) and count it explicitly."""
        with self._lock:
            self.attempts_no_usage += 1


class MockApertusClient:
    """Deterministic stand-in. NEVER report its outputs as model results."""

    backend = "mock"
    model = "MOCK-NOT-APERTUS"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def _noise(self, *parts: str) -> float:
        h = hashlib.sha256(("|".join(parts) + f"|{self.seed}").encode()).digest()
        return (int.from_bytes(h[:4], "big") / 0xFFFFFFFF - 0.5) * 0.3

    def complete(self, messages, *, max_tokens: int = 2048,
                 temperature: float = 0.0, seed: int | None = None) -> ChatResult:
        t0 = time.monotonic()
        content = messages[-1]["content"]
        text = self._dispatch(content)
        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        return ChatResult(
            text=text, model=self.model, backend=self.backend,
            prompt_tokens=prompt_chars // 4,
            completion_tokens=len(text) // 4,
            latency_ms=(time.monotonic() - t0) * 1000.0,
            usage_estimated=True,
        )

    def _dispatch(self, content: str) -> str:
        if '"kind": "pair_similarity"' in content:
            return self._mock_similarity(content)
        if '"kind": "judge_tag"' in content:
            return self._mock_judge_tag(content)
        if '"kind": "judge_pair"' in content:
            return self._mock_judge(content)
        if '"kind": "doc_baseline"' in content:
            return self._mock_baseline(content)
        return json.dumps({"error": "mock: unrecognized call kind"})

    def _mock_similarity(self, content: str) -> str:
        req = _extract_json_block(content)
        out = []
        for p in req.get("pairs", []):
            s = lexical_similarity(p["a"], p["b"]) + self._noise(p["a"], p["b"])
            out.append({"i": p["i"], "j": p["j"],
                        "score": round(5 * max(0.0, min(1.0, s)), 1)})
        return json.dumps({"scores": out})

    def _mock_judge(self, content: str) -> str:
        req = _extract_json_block(content)
        a, b = req.get("a", ""), req.get("b", "")
        sim = lexical_similarity(a, b)
        diff = int(round(5 * (1.0 - sim) + self._noise(a, b) * 3))
        diff = max(0, min(5, diff))
        spans_a, spans_b = [], []
        if diff >= 2:
            wa = [t for t in a.split() if t.lower() not in {x.lower() for x in b.split()}]
            wb = [t for t in b.split() if t.lower() not in {x.lower() for x in a.split()}]
            spans_a = [" ".join(wa[:3])] if wa else []
            spans_b = [" ".join(wb[:3])] if wb else []
        return json.dumps({"semantic_difference": diff,
                           "differing_spans": {"side_a": spans_a, "side_b": spans_b}})

    def _mock_judge_tag(self, content: str) -> str:
        """Deterministic v3-tag mock: flag indices whose normalized token
        does not appear on the other side (lexical proxy, NOT semantic)."""
        req = _extract_json_block(content)
        arr_a = req.get("a", [])
        arr_b = req.get("b", [])
        def norm(t):
            return str(t).lower().strip(".,;:!?\"'()[]{}«»“”")
        words_a = {norm(t) for _, t in arr_a if norm(t)}
        words_b = {norm(t) for _, t in arr_b if norm(t)}
        def keep(tok):
            n = norm(tok)
            return bool(n) and not re.fullmatch(r"\d+([.,]\d+)?%?", n) \
                and len(n) > 1
        a_ids = [i for i, t in arr_a if keep(t) and norm(t) not in words_b]
        b_ids = [i for i, t in arr_b if keep(t) and norm(t) not in words_a]
        return json.dumps({"a_ids": a_ids, "b_ids": b_ids})

    def _mock_baseline(self, content: str) -> str:
        req = _extract_json_block(content)
        s1 = json.loads(req["sentence1"])
        s2 = json.loads(req["sentence2"])
        # distribute a deterministic score concentrated on unmatched tokens
        bset = {t.lower().strip(".,;:!?") for t in s2}
        aset = {t.lower().strip(".,;:!?") for t in s1}
        lab1 = [[t, -1 if all(c in ".,;:!?\"'-–—()[]{}" for c in t)
                 else (5 if t.lower().strip(".,;:!?") in bset or re.fullmatch(r"\d+", t)
                       else int(2 + self._noise(t, "a") * 6) % 4)] for t in s1]
        lab2 = [[t, -1 if all(c in ".,;:!?\"'-–—()[]{}" for c in t)
                 else (5 if t.lower().strip(".,;:!?") in aset or re.fullmatch(r"\d+", t)
                       else int(2 + self._noise(t, "b") * 6) % 4)] for t in s2]
        return json.dumps({"sentence1": lab1, "sentence2": lab2})


def _extract_json_block(content: str) -> dict:
    """Pull the last top-level JSON object out of a prompt body."""
    start = content.find("```json")
    if start >= 0:
        end = content.find("```", start + 7)
        return json.loads(content[start + 7:end])
    # fall back to first balanced object
    i = content.find("{")
    depth = 0
    for k in range(i, len(content)):
        if content[k] == "{":
            depth += 1
        elif content[k] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(content[i:k + 1])
    raise ValueError("no JSON block in mock request")


def client_from_env_or_mock(backend: str, *, seed: int = 0):
    """Return (client, backend_name) for the RESOLVED backend only.

    SPLITALIGN_BACKEND is read once by ``run.resolve_backend``; nothing here
    consults the env again, so a recorded backend name always reflects the
    client that actually produced the output.
    """
    if backend == "mock":
        return MockApertusClient(seed=seed), "mock"
    return ApertusClient.from_env(), "apertus"
