"""Synthetic configuration and launcher tests. No data or model calls.

Docker is replaced by an argv/env-capture executable. This checks the Make
recipe boundary, not Docker building, image execution, or official judging.
"""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import splitalign
from splitalign.apertus import ApertusClient, MissingCredentials
from splitalign import packaged_run


TRACK = Path(__file__).resolve().parents[1]
REPO = TRACK.parent
OFFICIAL = {
    "LLM_BASE_URL": "https://packaging-contract.invalid/v1",
    "LLM_API_KEY": "FAKE-TEST-ONLY-KEY",
    "LLM_NAME": "synthetic-contract-model",
    "APERTUS_BOUND_SPEC_JSON": json.dumps({
        "per_message_tokens": 8, "request_overhead_tokens": 16,
        "provenance": "Synthetic unit-test fixture; not a provider bound.",
        "provider": "synthetic", "model": "synthetic-contract-model"}),
}
BOUNDED = ["predict", "--split", "val", "--lang", "de", "--limit", "1",
           "--offset", "0", "--max-requests", "1", "--max-tokens", "100"]


class PackagingContractTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        for target in ("urllib.request.urlopen", "socket.create_connection",
                       "socket.socket.connect", "socket.socket.connect_ex"):
            blocker = patch(target, side_effect=AssertionError("NETWORK FORBIDDEN"))
            blocker.start()
            self.addCleanup(blocker.stop)
        self.no_complete = patch.object(ApertusClient, "complete",
                                       side_effect=AssertionError("DISPATCH FORBIDDEN"))
        self.no_complete.start()
        self.addCleanup(self.no_complete.stop)

    def invoke(self, argv, env=None):
        out, err = io.StringIO(), io.StringIO()
        with patch.dict(os.environ, env or {}), redirect_stdout(out), redirect_stderr(err):
            rc = packaged_run.main(argv)
        return rc, out.getvalue(), err.getvalue()

    def test_official_variables_construct_real_client_without_dispatch(self):
        with patch.dict(os.environ, OFFICIAL):
            client = ApertusClient.from_env()
        self.assertEqual(client.base, OFFICIAL["LLM_BASE_URL"])
        self.assertEqual(client.key, OFFICIAL["LLM_API_KEY"])
        self.assertEqual(client.model, OFFICIAL["LLM_NAME"])
        self.assertEqual(client.bound_spec.request_overhead_tokens, 16)

    def test_legacy_variables_remain_accepted(self):
        legacy = {"APERTUS_API_BASE": OFFICIAL["LLM_BASE_URL"],
                  "APERTUS_API_KEY": OFFICIAL["LLM_API_KEY"],
                  "APERTUS_MODEL": OFFICIAL["LLM_NAME"]}
        with patch.dict(os.environ, legacy):
            self.assertEqual(ApertusClient.from_env().model, OFFICIAL["LLM_NAME"])

    def test_equal_aliases_are_accepted(self):
        with patch.dict(os.environ, {**OFFICIAL, "APERTUS_MODEL": OFFICIAL["LLM_NAME"]}):
            self.assertEqual(ApertusClient.from_env().model, OFFICIAL["LLM_NAME"])

    def test_conflicting_aliases_fail_without_disclosing_values(self):
        for official, legacy in (("LLM_API_KEY", "APERTUS_API_KEY"),
                                 ("LLM_BASE_URL", "APERTUS_API_BASE"),
                                 ("LLM_NAME", "APERTUS_MODEL")):
            with self.subTest(official=official), patch.dict(os.environ, {**OFFICIAL, legacy: "OTHER-PRIVATE-VALUE"}):
                with self.assertRaises(MissingCredentials) as raised:
                    ApertusClient.from_env()
                msg = str(raised.exception)
                self.assertIn(official, msg)
                self.assertNotIn(OFFICIAL[official], msg)
                self.assertNotIn("OTHER-PRIVATE-VALUE", msg)

    def test_default_without_configuration_is_mock_preflight_only(self):
        rc, out, _ = self.invoke([])
        self.assertEqual(rc, 0)
        self.assertIn("backend=mock", out)
        self.assertIn("no dataset opened", out)

    def test_official_configuration_selects_apertus_preflight(self):
        rc, out, _ = self.invoke([], OFFICIAL)
        self.assertEqual(rc, 0)
        self.assertIn("backend=apertus", out)
        self.assertNotIn(OFFICIAL["LLM_API_KEY"], out)

    def test_partial_official_configuration_never_silently_mocks(self):
        rc, out, err = self.invoke([], {"LLM_NAME": OFFICIAL["LLM_NAME"]})
        self.assertEqual(rc, 3)
        self.assertNotIn("backend=mock", out)
        self.assertIn("LLM_API_KEY", err)

    def test_missing_bound_blocks_apertus_before_data_or_dispatch(self):
        env = {k: v for k, v in OFFICIAL.items() if k != "APERTUS_BOUND_SPEC_JSON"}
        rc, _, err = self.invoke(BOUNDED, env)
        self.assertEqual(rc, 3)
        self.assertIn("APERTUS_BOUND_SPEC_JSON", err)

    def test_invalid_bound_is_rejected(self):
        for bound in ("invalid-json", "{}", json.dumps({
            "per_message_tokens": -1, "request_overhead_tokens": 16,
            "provenance": "synthetic"}), json.dumps({
            "per_message_tokens": 8, "request_overhead_tokens": 16,
            "provenance": "synthetic", "model": "different-model"})):
            with self.subTest(bound=bound):
                rc, _, _ = self.invoke([], {**OFFICIAL, "APERTUS_BOUND_SPEC_JSON": bound})
                self.assertEqual(rc, 3)

    def test_explicit_backend_conflict_is_rejected(self):
        rc, _, _ = self.invoke(["check-config", "--backend", "mock"],
                               {**OFFICIAL, "SPLITALIGN_BACKEND": "apertus"})
        self.assertEqual(rc, 3)

    def test_unselected_dataset_blocks_before_importing_pipeline(self):
        with patch.dict(sys.modules, {"splitalign.run": None}):
            rc, _, err = self.invoke(BOUNDED, OFFICIAL)
        self.assertEqual(rc, 3)
        self.assertIn("SPLITALIGN_DATA_DIR", err)

    def test_missing_explicit_selection_or_budget_is_rejected(self):
        for flag in ("--split", "--lang", "--limit", "--offset", "--max-requests", "--max-tokens"):
            args = BOUNDED.copy()
            idx = args.index(flag)
            del args[idx:idx + 2]
            with self.subTest(flag=flag), self.assertRaises(SystemExit) as raised:
                self.invoke(args, OFFICIAL)
            self.assertEqual(raised.exception.code, 2)

    def test_invalid_language_and_nonpositive_bounds_are_rejected(self):
        for flag, value in (("--lang", "unknown"), ("--split", "test"),
                            ("--limit", "0"), ("--limit", "-1"),
                            ("--offset", "-1"), ("--max-requests", "0"),
                            ("--max-tokens", "0")):
            args = BOUNDED.copy()
            args[args.index(flag) + 1] = value
            with self.subTest(flag=flag, value=value), self.assertRaises(SystemExit):
                self.invoke(args, OFFICIAL)

    def test_explicit_run_forwards_exact_selection_to_stub_only(self):
        recorded = []
        fake_run = types.ModuleType("splitalign.run")
        fake_run.main = lambda argv: recorded.append(argv) or 0
        with patch.dict(sys.modules, {"splitalign.run": fake_run}), \
                patch.object(splitalign, "run", fake_run, create=True):
            rc, _, _ = self.invoke(BOUNDED, {**OFFICIAL, "SPLITALIGN_DATA_SELECTED": "1"})
        self.assertEqual(rc, 0)
        forwarded = recorded[0]
        for flag in ("--split", "--lang", "--limit", "--offset", "--max-requests", "--max-tokens"):
            self.assertEqual(forwarded[forwarded.index(flag) + 1], BOUNDED[BOUNDED.index(flag) + 1])
        self.assertEqual(forwarded[forwarded.index("--backend") + 1], "apertus")

    def test_explicit_all_languages_preserves_one_pipeline_invocation(self):
        recorded = []
        fake_run = types.ModuleType("splitalign.run")
        fake_run.main = lambda argv: recorded.append(argv) or 0
        args = BOUNDED.copy()
        args[0] = "pipeline"
        args[args.index("--lang") + 1] = "all"
        with patch.dict(sys.modules, {"splitalign.run": fake_run}), \
                patch.object(splitalign, "run", fake_run, create=True):
            rc, _, _ = self.invoke(args, {**OFFICIAL, "SPLITALIGN_DATA_SELECTED": "1"})
        self.assertEqual(rc, 0)
        self.assertEqual(len(recorded), 1)
        self.assertEqual(recorded[0][0], "pipeline")
        self.assertEqual(recorded[0][recorded[0].index("--lang") + 1], "all")

    def test_docker_default_is_preflight_and_build_context_excludes_data(self):
        dockerfile = (REPO / "Dockerfile").read_text()
        self.assertIn('ENTRYPOINT ["python", "-m", "splitalign.packaged_run"]', dockerfile)
        self.assertIn('CMD ["check-config"]', dockerfile)
        ignore = (REPO / ".dockerignore").read_text().splitlines()
        rules = [line for line in ignore if line and not line.startswith("#")]
        self.assertEqual(rules[0], "**")
        self.assertFalse(any("data" in line or "results" in line or "evidence" in line for line in rules[1:]))

    @unittest.skipUnless(shutil.which("make"), "make unavailable: recipe contract not executed")
    def test_make_preflight_and_run_forward_configuration_to_docker_stub(self):
        # A temporary fake executable records argv plus selected FAKE env only.
        # No Docker daemon, build, data loader or endpoint is contacted.
        with tempfile.TemporaryDirectory(prefix="splitalign-packaging-") as tmp:
            tmp = Path(tmp)
            log = tmp / "docker-argv.jsonl"
            docker = tmp / "docker"
            docker.write_text(f"#!{sys.executable}\nimport json,os,sys\n"
                "with open(os.environ['CONTRACT_LOG'], 'a') as f:\n"
                " f.write(json.dumps({'argv':sys.argv[1:], 'env':{k:os.environ[k] for k in "
                f"{list(OFFICIAL)!r}" + " if k in os.environ}})+'\\n')\n")
            docker.chmod(0o700)
            env = {"PATH": str(tmp) + os.pathsep + "/usr/bin:/bin",
                   "CONTRACT_LOG": str(log), **OFFICIAL}
            selected = tmp / "selected synthetic dev directory"
            selected.mkdir()  # Empty fixture: never fabricated dataset content.
            result = subprocess.run([shutil.which("make", path=env["PATH"]), "-s", "-C", str(TRACK),
                "preflight", "SOURCE_COMMIT=", "SOURCE_DIRTY=", f"SPLITALIGN_DATA_DIR={selected}"],
                env=env, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            entries = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual([e["argv"][0] for e in entries], ["build", "run"])
            run = entries[1]
            for name in OFFICIAL:
                self.assertIn(name, run["argv"])
                self.assertNotIn(OFFICIAL[name], run["argv"])
            mount = run["argv"][run["argv"].index("--mount") + 1]
            self.assertEqual(mount, f"type=bind,src={selected},dst=/app/track_2a/data,readonly")
            self.assertIn("SPLITALIGN_DATA_SELECTED=1", run["argv"])
            self.assertEqual(run["argv"][-1], "check-config")
            forwarded_env = {k: v for k, v in run["env"].items() if k in run["argv"]}
            rc, out, _ = self.invoke([], forwarded_env)
            self.assertEqual(rc, 0)
            self.assertIn("backend=apertus", out)

            explicit = BOUNDED.copy()
            explicit[0] = "pipeline"
            explicit[explicit.index("--lang") + 1] = "all"
            result = subprocess.run([shutil.which("make", path=env["PATH"]), "-s", "-C", str(TRACK),
                "run", "SOURCE_COMMIT=", "SOURCE_DIRTY=", f"SPLITALIGN_DATA_DIR={selected}",
                "RUN_ARGS=" + " ".join(explicit)], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            entries = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual([e["argv"][0] for e in entries], ["build", "run", "build", "run"])
            self.assertEqual(entries[-1]["argv"][-len(explicit):], explicit)

            # A plain make run must stop before invoking even the fake Docker.
            for extras in ([], ["RUN_ARGS=" + " ".join(explicit)],
                           [f"SPLITALIGN_DATA_DIR={selected}", "RUN_ARGS=check-config"]):
                result = subprocess.run([shutil.which("make", path=env["PATH"]), "-s", "-C", str(TRACK),
                    "run", *extras], env=env, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(len(log.read_text().splitlines()), 4)


    @unittest.skipUnless(shutil.which("make"), "make unavailable")
    def test_make_full_test_requires_explicit_fixture_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            log = tmp / "argv.jsonl"
            docker = tmp / "docker"
            docker.write_text(f"#!{sys.executable}\nimport json,os,sys\n"
                "with open(os.environ['CONTRACT_LOG'],'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n")
            docker.chmod(0o700)
            env = {"PATH": str(tmp) + ":/usr/bin:/bin", "CONTRACT_LOG": str(log)}
            base = ["make", "-s", "-C", str(TRACK), "test", "SOURCE_COMMIT=", "SOURCE_DIRTY="]
            rejected = subprocess.run(base, env=env, capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("approved dev test fixtures", rejected.stderr)
            self.assertFalse(log.exists())
            selected = tmp / "explicit synthetic fixture directory"
            selected.mkdir()
            accepted = subprocess.run(base + [f"SPLITALIGN_DATA_DIR={selected}"],
                                      env=env, capture_output=True, text=True)
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            entries = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual([e[0] for e in entries], ["build", "run"])
            invocation = entries[-1]
            self.assertEqual(invocation[invocation.index("--network") + 1], "none")
            self.assertEqual(invocation[invocation.index("--mount") + 1],
                f"type=bind,src={selected},dst=/app/track_2a/data,readonly")
            self.assertEqual(invocation[-4:], ["-m", "pytest", "tests", "-q"])
            self.assertFalse(any("API_KEY" in a for a in invocation))


if __name__ == "__main__":
    unittest.main()
