"""Export pipeline results to the static evidence viewer."""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import __version__, PROMPT_VERSION


def export_evidence(items_by_mode: dict[str, list[dict]], out_path: Path,
                    *, backend: str, model: str | None, split: str,
                    evaluation: dict | None = None,
                    limitations: list[str] | None = None,
                    run: dict | None = None) -> Path:
    """Write evidence.js (window.EVIDENCE) consumed by viewer/index.html.

    ``run`` carries the run scope's identity and coverage (run_id, intended
    vs produced modes, completed/intended counts, cap/partial status,
    matched IDs) so the viewer can never present a capped or unmatched run
    as a complete comparison. Output is strict JSON (no NaN/Infinity).
    """
    ev = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "splitalign_version": __version__,
        "prompt_version": PROMPT_VERSION,
        "backend": backend,
        "model": model,
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
