"""Firewall tests — synthetic fixtures only, never real held-out data."""
import json
import os
from pathlib import Path

import pytest

from splitalign import guard


def test_allows_dev_splits():
    assert guard.normalize_split("train") == "dev/train"
    assert guard.normalize_split("val") == "dev/val"
    assert guard.normalize_split("dev/train") == "dev/train"


@pytest.mark.parametrize("s", ["test", "full", "dev/test", "dev/full", "gold/test", "TEST"])
def test_refuses_heldout_splits(s):
    with pytest.raises(guard.HeldOutViolation):
        guard.normalize_split(s)


def test_gold_path_resolves_inside_dev(tmp_path):
    p = guard.gold_path("val", "de")
    assert "dev" in p.parts and "val" in p.parts
    assert p.name == "gold_admin_de.jsonl"


def test_gold_path_rejects_bad_lang():
    with pytest.raises(guard.HeldOutViolation):
        guard.gold_path("val", "xx")


def test_symlink_escape_fails(tmp_path):
    root = tmp_path / "gold" / "dev"
    (root / "val").mkdir(parents=True)
    outside = tmp_path / "secret.jsonl"
    outside.write_text("{}")
    link = root / "val" / "evil.jsonl"
    os.symlink(outside, link)
    with pytest.raises(guard.HeldOutViolation):
        guard.assert_path_allowed(link, root=root)


def test_symlinked_dir_component_fails(tmp_path, monkeypatch):
    """Whole-directory symlink replacement inside the repo must fail closed."""
    # data/gold/dev/val -> outside dir: component-level check must reject
    # Keep the probe wholly synthetic; selected corpus mounts are read-only.
    track = tmp_path / "synthetic_track"
    root = track / "data" / "gold" / "dev"
    root.mkdir(parents=True)
    monkeypatch.setattr(guard, "TRACK_DIR", track)
    monkeypatch.setattr(guard, "GOLD_DIR", root)
    outside = tmp_path / "outside"
    outside.mkdir()
    link_dir = guard.GOLD_DIR / "val_link_probe"
    try:
        os.symlink(outside, link_dir)
        probe = link_dir / "gold_admin_de.jsonl"
        with pytest.raises(guard.HeldOutViolation):
            guard.assert_path_allowed(probe, root=guard.GOLD_DIR)
    finally:
        link_dir.unlink(missing_ok=True)


def test_repo_parent_dir_named_test_is_not_false_positive(tmp_path):
    """A checkout under e.g. /tmp/test/... must still allow its dev files."""
    p = guard.gold_path("val", "de")
    # the real repo may live under a parent containing 'test' — resolve ok
    assert p.name == "gold_admin_de.jsonl"


def test_heldout_token_in_path_fails(tmp_path):
    root = tmp_path / "gold" / "dev"
    root.mkdir(parents=True)
    p = tmp_path / "test" / "x.jsonl"
    with pytest.raises(guard.HeldOutViolation):
        guard.assert_path_allowed(p, root=root)


def test_manifest_membership():
    manifest = guard.load_manifest()
    assert manifest and all(i.startswith("admin_") for i in manifest)
    with pytest.raises(guard.HeldOutViolation):
        guard.assert_ids_allowed(["admin_de_0", "test_item_1"], manifest)


def test_scan_tree_clean():
    assert guard.scan_tree_for_heldout() == []


def test_assert_ids_ok():
    guard.assert_ids_allowed(["admin_de_0"], {"admin_de_0", "admin_fr_1"})

