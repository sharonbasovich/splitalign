"""Held-out-data firewall.

Only ``dev/train`` and ``dev/val`` gold data may ever be read, inspected or
evaluated by this codebase. Every data access goes through this module so the
policy is enforced in exactly one place:

* path allowlist — only committed files under ``data/gold/dev/{train,val}``
  matching ``gold_admin_{de,fr,it}[_short].jsonl`` may be opened;
* symlink safety — the resolved real path must stay inside the allowlisted
  directories;
* ID manifest — every record loaded must carry an ID listed in
  ``data/manifest/dev_ids.json`` (written at fetch time);
* no bypass — there is intentionally no ``--final-run``/env escape hatch.
  Final held-out evaluation is a separately controlled, frozen-method process
  that does not live in this build.

Violation of any rule raises :class:`HeldOutViolation` (fail closed).
"""
from __future__ import annotations

import json
from pathlib import Path

# Repository layout anchor: <repo>/track_2a
TRACK_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = TRACK_DIR / "data"
GOLD_DIR = DATA_DIR / "gold" / "dev"
MANIFEST_PATH = DATA_DIR / "manifest" / "dev_ids.json"

ALLOWED_SPLITS = ("dev/train", "dev/val")
ALLOWED_LANGS = ("de", "fr", "it")

# Path fragments that are always held out, anywhere they appear.
HELDOUT_TOKENS = {"test", "full", "full-corpus", "heldout", "held_out"}


class HeldOutViolation(RuntimeError):
    """Raised whenever code attempts to touch held-out material."""


def _parts_lower(path: Path) -> set[str]:
    return {p.lower() for p in path.parts}


def assert_path_allowed(path: Path, root: Path = GOLD_DIR) -> Path:
    """Return ``path`` resolved, or raise if it is outside the dev allowlist.

    Fails closed on symlink escapes: the fully-resolved real path must live
    under ``root`` (which itself resolves first).
    """
    real_root = root.resolve()
    try:
        real = path.resolve(strict=False)
    except OSError as exc:  # pragma: no cover - defensive
        raise HeldOutViolation(f"cannot resolve path {path}: {exc}") from exc

    parts = _parts_lower(real.relative_to(real.anchor)) if real.is_absolute() else _parts_lower(real)
    if parts & HELDOUT_TOKENS:
        raise HeldOutViolation(f"held-out path component in {path}")
    if real_root != real and real_root not in real.parents:
        raise HeldOutViolation(f"path escapes dev allowlist: {path}")
    return real


def normalize_split(split: str) -> str:
    """Map a user-supplied split name onto the dev allowlist or fail closed."""
    s = split.strip().lower().replace("_", "/")
    aliases = {"train": "dev/train", "val": "dev/val", "dev/train": "dev/train",
               "dev/val": "dev/val", "dev": "dev"}
    if s in ("test", "full", "dev/test", "dev/full") or "test" in s.split("/"):
        raise HeldOutViolation(f"split '{split}' is held out")
    if s not in aliases:
        raise HeldOutViolation(f"unknown split '{split}' (allowed: {ALLOWED_SPLITS})")
    return aliases[s]


def gold_path(split: str, lang: str, short: bool = False) -> Path:
    """Resolve an allowlisted gold file path (never opens held-out data)."""
    norm = normalize_split(split)
    if norm == "dev":
        raise HeldOutViolation("specify dev/train or dev/val explicitly")
    if lang not in ALLOWED_LANGS:
        raise HeldOutViolation(f"unknown language '{lang}'")
    sub = norm.split("/", 1)[1]
    name = f"gold_admin_{lang}{'_short' if short else ''}.jsonl"
    return assert_path_allowed(GOLD_DIR / sub / name)


def load_manifest(path: Path = MANIFEST_PATH) -> set[str]:
    if not path.exists():
        raise HeldOutViolation(
            f"dev-ID manifest missing at {path}; run `python -m splitalign.run fetch-data`")
    return set(json.loads(path.read_text())["ids"])


def assert_ids_allowed(ids, manifest: set[str] | None = None) -> None:
    manifest = manifest if manifest is not None else load_manifest()
    unknown = [i for i in ids if i not in manifest]
    if unknown:
        raise HeldOutViolation(f"item ids outside dev manifest: {unknown[:5]} ...")


def scan_tree_for_heldout(root: Path = TRACK_DIR) -> list[str]:
    """Return repo-relative paths that look like held-out material.

    Used by tests to prove the checkout contains no test/full corpus data.
    """
    offenders = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        parts = {x.lower() for x in rel.parts}
        if ".git" in parts:
            continue
        if "gold" in parts and "dev" not in parts:
            offenders.append(str(rel))
        elif parts & {"test", "full"} and "data" in parts:
            offenders.append(str(rel))
    return offenders


def guard_data_dir(path: Path) -> Path:
    """Validate any path inside data/ before opening (symlink + token check)."""
    real = assert_path_allowed(path, root=DATA_DIR)
    parts = _parts_lower(real)
    if "gold" in parts and "dev" not in parts:
        raise HeldOutViolation(f"non-dev gold path: {path}")
    return real
