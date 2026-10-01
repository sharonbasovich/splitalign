# Data — dev-only firewall

This directory intentionally contains ONLY the development splits of
SwissGov-RSD (CC-BY-4.0, ZurichNLP):

- `gold/dev/train/gold_admin_{de,fr,it}.jsonl` — 134 items/lang
- `gold/dev/val/gold_admin_{de,fr,it}.jsonl` — 34 items/lang
- `manifest/dev_ids.json` — every dev item id (504), written at fetch time
- `manifest/sha256.json` — sha256 of each fetched file (integrity)
- `manifest/fetch_log.json` — fetch provenance

Pinned upstream commit: `1807a42100e742ed03d337c54c4b9ea86995f565`
(repo: https://github.com/ZurichNLP/SwissGov-RSD)

## Held-out firewall

`splitalign.guard` fails closed on any repo-internal `test`/`full` path
component, any path escaping `data/gold/dev/` (symlinks resolved, and no
repo-internal path component may itself be a symlink), and any item id
absent from the dev manifest. The manifest is a mutable local file — a
development guard, not a proof of impossibility. `splitalign.fetch_data` only knows an
explicit allowlist of upstream URLs — there is no code path that downloads
held-out material, and no `--final-run` bypass. Final held-out evaluation is
a separately controlled process outside this build.

`make fetch-data` (or `python -m splitalign.run fetch-data`) re-verifies the
allowlisted files and scans the checkout for held-out material.
