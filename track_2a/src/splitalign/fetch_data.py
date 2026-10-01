"""Allowlisted data fetcher.

Downloads ONLY the files listed below from the pinned upstream commit,
verifies sha256, and writes the dev-ID manifest. There is no code path that
can fetch held-out material: the allowlist is the set of exact URLs.
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

from . import guard

PINNED_COMMIT = "1807a42100e742ed03d337c54c4b9ea86995f565"
RAW_BASE = f"https://raw.githubusercontent.com/ZurichNLP/SwissGov-RSD/{PINNED_COMMIT}"

ALLOWLIST = {
    # split -> lang -> upstream path under data/evaluation/gold_labels/dev
    f"gold/dev/{split}/gold_admin_{lang}{short}.jsonl": {
        "url": f"{RAW_BASE}/data/evaluation/gold_labels/dev/{split}/gold_admin_{lang}{short}.jsonl",
        "split": f"dev/{split}",
    }
    for split in ("train", "val")
    for lang in ("de", "fr", "it")
    for short in ("", "_short")
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def fetch(dest_root: Path = guard.DATA_DIR, verify_hashes: dict | None = None) -> dict:
    dest_root = Path(dest_root)
    log = {"commit": PINNED_COMMIT, "files": {}}
    for rel, meta in ALLOWLIST.items():
        out = guard.assert_path_allowed(dest_root / rel, root=dest_root / "gold" / "dev")
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            with urllib.request.urlopen(meta["url"], timeout=60) as r:
                out.write_bytes(r.read())
        digest = sha256_file(out)
        if verify_hashes and rel in verify_hashes and verify_hashes[rel] != digest:
            raise guard.HeldOutViolation(f"hash mismatch on {rel}")
        log["files"][rel] = {"sha256": digest, "bytes": out.stat().st_size}
    write_manifest(dest_root)
    (dest_root / "manifest").mkdir(parents=True, exist_ok=True)
    (dest_root / "manifest" / "fetch_log.json").write_text(json.dumps(log, indent=2))
    return log


def write_manifest(dest_root: Path = guard.DATA_DIR) -> Path:
    ids: list[str] = []
    for sub in ("train", "val"):
        for lang in guard.ALLOWED_LANGS:
            p = dest_root / "gold" / "dev" / sub / f"gold_admin_{lang}.jsonl"
            p = guard.assert_path_allowed(p, root=dest_root / "gold" / "dev")
            for line in p.read_text().splitlines():
                if line.strip():
                    ids.append(json.loads(line)["id"])
    (dest_root / "manifest").mkdir(parents=True, exist_ok=True)
    mp = dest_root / "manifest" / "dev_ids.json"
    mp.write_text(json.dumps({"commit": PINNED_COMMIT, "ids": sorted(set(ids))},
                             indent=2))
    return mp


def load_gold_items(split: str, lang: str, short: bool = False) -> list[dict]:
    """The ONLY entrypoint for reading gold data — firewall enforced."""
    p = guard.gold_path(split, lang, short)
    items = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    guard.assert_ids_allowed([it["id"] for it in items])
    for it in items:
        n_a = len(it["text_a"].split())
        n_b = len(it["text_b"].split())
        assert n_a == len(it["labels_a"]) and n_b == len(it["labels_b"]), \
            f"label/token length mismatch in {it['id']}"
    return items
