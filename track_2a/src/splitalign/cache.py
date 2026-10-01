"""Deterministic file cache for model calls.

Key = sha256(backend | model | prompt_version | call_kind | split | item_id |
canonical payload). Cached records are immutable; a corrupt entry is treated
as a miss rather than an error.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def payload_hash(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


class DiskCache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def key(self, *, backend: str, model: str, prompt_version: str,
            kind: str, split: str, item_id: str, payload) -> str:
        return payload_hash({
            "backend": backend, "model": model, "pv": prompt_version,
            "kind": kind, "split": split, "item": item_id, "payload": payload,
        })

    def get(self, key: str):
        p = self.root / f"{key}.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def put(self, key: str, record: dict) -> None:
        tmp = self.root / f"{key}.tmp"
        tmp.write_text(json.dumps(record, ensure_ascii=False))
        tmp.replace(self.root / f"{key}.json")
