"""Export pipeline results to the static evidence viewer."""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import __version__, PROMPT_VERSION


def inference_provenance(items_by_mode: dict[str, list[dict]],
                         manifest: dict | None = None) -> dict:
    """Prompt/model/package versions that actually produced the items.

    Taken from the run manifest and the per-item ``provenance`` records —
    never from the exporting process. Fields are null when unrecorded;
    ``conflict`` lists any field with more than one recorded value.
    """
    seen: dict[str, set] = {"prompt_version": set(), "model": set(),
                            "splitalign_version": set(), "backend": set()}
    for d in items_by_mode.values():
        for it in d:
            prov = it.get("provenance") or {}
            for k in seen:
                if prov.get(k) is not None:
                    seen[k].add(prov[k])
    man = manifest or {}
    out: dict = {}
    for k, vals in seen.items():
        if man.get(k) is not None:
            vals = vals | {man[k]}
        out[k] = sorted(vals)[0] if len(vals) == 1 else None
    out["conflict"] = {k: sorted(v) for k, v in seen.items() if len(v) > 1}
    out["recorded_from"] = ("manifest+items" if man else "items") \
        if any(seen.values()) or man else "none"
    return out


def export_evidence(items_by_mode: dict[str, list[dict]], out_path: Path,
                    *, backend: str, split: str,
                    inference: dict | None = None,
                    model: str | None = None,
                    evaluation: dict | None = None,
                    limitations: list[str] | None = None,
                    run: dict | None = None) -> Path:
    """Write evidence.js (window.EVIDENCE) consumed by viewer/index.html.

    ``inference`` (see ``inference_provenance``) is what produced the items;
    ``exporter`` is only the code that wrote this file. The two are kept
    separate so re-exporting a historical run never relabels it with the
    current prompt or package version. ``run`` carries the run scope's
    identity and coverage (run_id, intended vs produced modes,
    completed/intended counts, cap/partial status, matched IDs) so the viewer
    can never present a capped or unmatched run as a complete comparison.
    Output is strict JSON (no NaN/Infinity).
    """
    inference = dict(inference or inference_provenance(items_by_mode))
    if model is not None and inference.get("model") is None:
        inference["model"] = model
    ev = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "exporter": {"splitalign_version": __version__,
                     "prompt_version": PROMPT_VERSION,
                     "note": "version of the code that WROTE this file; not "
                             "the inference configuration"},
        "inference": inference,
        # headline fields = inference provenance (null when unrecorded)
        "splitalign_version": inference.get("splitalign_version"),
        "prompt_version": inference.get("prompt_version"),
        "backend": backend,
        "model": inference.get("model"),
        "mock": backend == "mock",
        "split": split,
        "run": run,
        "evaluation": evaluation,
        "limitations": limitations or [],
        "items_by_mode": items_by_mode,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "window.EVIDENCE = " + json.dumps(ev, ensure_ascii=False, allow_nan=False) + ";\n")
    return out_path
