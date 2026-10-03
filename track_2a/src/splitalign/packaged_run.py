"""Safe Docker launcher; configuration checks never import the data pipeline.

This is a packaging preflight and an explicit wrapper around the existing dev
CLI. It does not define the competition's still-unspecified evaluation I/O.
"""
from __future__ import annotations

import argparse
import os
import sys

from .apertus import ApertusClient, BoundNotConfigured, MissingCredentials


def _positive(value: str) -> int:
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return n


def _nonnegative(value: str) -> int:
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return n


def _backend(cli: str | None) -> str:
    env = os.environ.get("SPLITALIGN_BACKEND") or None
    if env and env not in ("apertus", "mock"):
        raise ValueError("SPLITALIGN_BACKEND must be apertus or mock")
    if cli and env and cli != env:
        raise ValueError("--backend conflicts with SPLITALIGN_BACKEND")
    # Official LLM_* configuration must not silently select a mock backend.
    configured = any(os.environ.get(k) for k in (
        "LLM_NAME", "LLM_BASE_URL", "LLM_API_KEY", "APERTUS_MODEL",
        "APERTUS_API_BASE", "APERTUS_API_KEY"))
    return cli or env or ("apertus" if configured else "mock")


def _check_config(backend: str) -> None:
    if backend == "mock":
        return
    client = ApertusClient.from_env()
    spec = client.bound_spec
    if spec is None:
        raise BoundNotConfigured("APERTUS_BOUND_SPEC_JSON is required")
    for name in ("per_message_tokens", "request_overhead_tokens"):
        n = getattr(spec, name)
        if type(n) is not int or n < 0:
            raise BoundNotConfigured(f"bound spec {name} must be a nonnegative integer")
    if not isinstance(spec.provenance, str) or not spec.provenance.strip():
        raise BoundNotConfigured("bound spec must include nonempty provenance")
    # Validate model coverage without preparing or sending a model request.
    client._prompt_token_bound([])


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="splitalign-packaged")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check-config", help="no data reads or model calls")
    check.add_argument("--backend", choices=("mock", "apertus"))
    for mode in ("predict", "baseline", "pipeline"):
        p = sub.add_parser(mode)
        p.add_argument("--split", required=True,
                       choices=("train", "val", "dev/train", "dev/val"))
        p.add_argument("--lang", required=True, choices=("de", "fr", "it", "all"))
        p.add_argument("--limit", required=True, type=_positive)
        p.add_argument("--offset", required=True, type=_nonnegative)
        p.add_argument("--max-requests", required=True, type=_positive)
        p.add_argument("--max-tokens", required=True, type=_positive)
        p.add_argument("--backend", choices=("mock", "apertus"))
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--bootstrap", type=_nonnegative, default=0)
    args = parser.parse_args(argv or ["check-config"])
    try:
        backend = _backend(args.backend)
        _check_config(backend)
        if args.command == "check-config":
            print(f"CONFIGURATION OK: backend={backend}; no dataset opened; "
                  "no model request attempted. Provider/bound validity is unverified.")
            return 0
        if os.environ.get("SPLITALIGN_DATA_SELECTED") != "1":
            raise ValueError("set SPLITALIGN_DATA_DIR to an explicitly selected "
                             "dev dataset directory when using make run")
    except (MissingCredentials, BoundNotConfigured, ValueError) as exc:
        print(f"CONFIGURATION REQUIRED: {exc}", file=sys.stderr)
        return 3

    # Import only after configuration and explicit-selection checks succeed.
    # Existing guard, budgets, manifest and output schemas remain unchanged.
    from . import run
    forwarded = [args.command, "--backend", backend]
    for name in ("split", "lang", "limit", "offset", "max_requests",
                 "max_tokens", "seed", "bootstrap"):
        forwarded += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    return run.main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
