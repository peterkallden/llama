"""Offline runner contracts using a disposable process, never a real model."""
import json
import os
import pathlib
import signal
import subprocess
import tempfile
import time
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]


class DoseRunnerContracts(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = pathlib.Path(directory.name)
        binary = self.root / "build/bin/llama-agent-flydelta-dataset-question-repair-model-smoke"
        binary.parent.mkdir(parents=True)
        binary.write_text('''#!/usr/bin/env python3
import json, os, pathlib, sys, time
mode = os.environ["LLAMA_AGENT_DOSE_EXPERIMENT"]
folder = pathlib.Path(os.environ["LLAMA_AGENT_LEARNING_LEDGER"]).parent
manifest = json.loads((folder.parent / "run-manifest.json").read_text())
assert manifest["trace_required"] and "--dose-comparison" in sys.argv
assert "--low-rank-rank-sweep" not in sys.argv
(folder / "entered.json").write_text(json.dumps({"arguments": sys.argv, "trace": os.environ.get("LLAMA_AGENT_RESIDENT_TRACE")}))
if os.environ.get("DOSE_RUNNER_WAIT") == "1":
    while True: time.sleep(0.05)
sys.exit(0)
''')
        binary.chmod(0o755)
        for name in ("model.gguf", "suite.jsonl"):
            (self.root / name).write_text("offline fixture\n")
        self.environment = {**os.environ, "LLAMA_AGENT_MODEL": str(self.root / "model.gguf"),
            "LLAMA_AGENT_DATASET_QUESTION_SUITE": str(self.root / "suite.jsonl"),
            "LLAMA_AGENT_BUILD_DIR": str(self.root / "build"),
            "LLAMA_AGENT_SMOKE_WORKDIR": str(self.root / "runs")}

    def run_script(self, name):
        return subprocess.run(["bash", str(REPO / "scripts" / name)], env=self.environment,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)

    def test_paired_profile_records_exact_commands_before_execution(self):
        # The fake process emits no arms: the summary must fail coverage even
        # though both processes exit successfully.
        result = self.run_script("test-agent-flydelta-dose-paired-model-smoke.sh")
        self.assertNotEqual(result.returncode, 0)
        run = next((self.root / "runs").glob("dose-run.*"))
        manifest = json.loads((run / "run-manifest.json").read_text())
        self.assertEqual(manifest["modes"], ["raw", "direction-dose"])
        self.assertIsNone(manifest["timeout_seconds"])
        for mode in manifest["modes"]:
            entered = json.loads((run / mode / "entered.json").read_text())
            self.assertEqual(entered["trace"], "1")
            self.assertIn("--dose-comparison", entered["arguments"])
            self.assertNotIn("--low-rank-rank-sweep", entered["arguments"])
        self.assertFalse((run / "normalized").exists())
        self.assertTrue((run / "dose-comparison.md").exists())

    @unittest.skipUnless(os.name == "posix", "POSIX signal contract")
    def test_interruption_does_not_launch_remaining_condition(self):
        self.environment["DOSE_RUNNER_WAIT"] = "1"
        process = subprocess.Popen(["bash", str(REPO / "scripts/test-agent-flydelta-dose-paired-model-smoke.sh")],
            env=self.environment, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        try:
            deadline = time.monotonic() + 10
            entered = []
            while time.monotonic() < deadline and process.poll() is None:
                entered = list((self.root / "runs").glob("dose-run.*/raw/entered.json"))
                if entered: break
                time.sleep(0.02)
            self.assertTrue(entered, "fake first condition never started")
            process.send_signal(signal.SIGTERM)
            self.assertEqual(process.wait(timeout=10), 130)
            run = entered[0].parent.parent
            self.assertTrue((run / "interrupted").exists())
            self.assertFalse((run / "direction-dose").exists())
            self.assertEqual((run / "raw/process.exit").read_text().strip(), "130")
        finally:
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            process.wait()


if __name__ == "__main__":
    unittest.main()
