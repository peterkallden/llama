import copy
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


spec = importlib.util.spec_from_file_location("package_source", Path(__file__).with_name("agent-package-source.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PackageSourceTests(unittest.TestCase):
    def setUp(self):
        self.repository = "example/llama"
        self.sha = "a" * 40
        self.source = dict(version=1, source_branch="feature/llama-agent", commit_sha=self.sha,
                           source_ci_run_id="10", sanitizer_run_id="11", package_workflow_run_id="12")
        self.runs = {}
        for run_id, workflow in [("10", "agent-ci.yml"), ("11", "agent-analysis.yml"),
                                 ("12", "agent-package-archive.yml")]:
            self.runs[run_id] = dict(path=f".github/workflows/{workflow}", status="completed",
                                    conclusion="success", repository=dict(full_name=self.repository),
                                    head_repository=dict(full_name=self.repository), event="push",
                                    head_branch="feature/llama-agent", head_sha=self.sha)
        # workflow_run records describe the default branch, not the packaged source.
        self.runs["12"].update(event="workflow_run", head_branch="master", head_sha="b" * 40)

    def validate(self):
        return module.validate_source(self.source, "12", self.repository, self.runs.__getitem__)

    def test_default_branch_package_resolves_verified_feature_source(self):
        self.assertEqual(self.validate()["commit_sha"], self.sha)

    def test_manual_runs_are_accepted_with_matching_verifications(self):
        for run in self.runs.values():
            run["event"] = "workflow_dispatch"
        self.assertEqual(self.validate()["commit_sha"], self.sha)

    def test_different_verification_commits_fail(self):
        for run_id in ("10", "11"):
            with self.subTest(run_id=run_id):
                original = self.runs[run_id]["head_sha"]
                self.runs[run_id]["head_sha"] = "b" * 40
                with self.assertRaises(ValueError):
                    self.validate()
                self.runs[run_id]["head_sha"] = original

    def test_untrusted_or_unsuccessful_run_fails(self):
        cases = [dict(conclusion="failure"), dict(status="in_progress"), dict(event="pull_request"),
                 dict(path=".github/workflows/other.yml"), dict(repository=dict(full_name="other/repo")),
                 dict(head_repository=dict(full_name="other/repo")), dict(head_branch="master")]
        for patch in cases:
            with self.subTest(patch=patch):
                original = copy.deepcopy(self.runs["10"])
                self.runs["10"].update(patch)
                with self.assertRaises(ValueError):
                    self.validate()
                self.runs["10"] = original

    def test_invalid_metadata_fails(self):
        cases = [dict(version=2), dict(source_branch="master"), dict(commit_sha="feature/llama-agent"),
                 dict(package_workflow_run_id="999"), dict(sanitizer_run_id=None), dict(source_ci_run_id="../10")]
        for patch in cases:
            with self.subTest(patch=patch):
                original = self.source.copy()
                self.source.update(patch)
                with self.assertRaises(ValueError):
                    self.validate()
                self.source = original

    def test_failed_or_wrong_package_workflow_fails(self):
        self.runs["12"]["conclusion"] = "failure"
        with self.assertRaises(ValueError):
            self.validate()
        self.runs["12"].update(conclusion="success", path=".github/workflows/agent-ci.yml")
        with self.assertRaises(ValueError):
            self.validate()

    def test_download_to_output_uses_metadata_not_default_branch(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            output = Path(directory) / "output"
            environment = dict(GITHUB_REPOSITORY=self.repository, PACKAGE_RUN_ID="12",
                               RUNNER_TEMP=directory, GITHUB_OUTPUT=str(output))

            def download(command, **kwargs):
                self.assertEqual(command[0:4], ["gh", "run", "download", "12"])
                self.assertIn("agent-development-source", command)
                destination = Path(command[command.index("--dir") + 1])
                (destination / "agent-package-source.json").write_text(json.dumps(self.source))

            def api(*args):
                return self.runs[args[-1].rsplit("/", 1)[-1]]

            with mock.patch.dict(os.environ, environment), mock.patch.object(module.subprocess, "run", download), \
                    mock.patch.object(module, "gh_json", api), contextlib.redirect_stdout(io.StringIO()):
                module.main()
            self.assertEqual(output.read_text(), f"source_branch=feature/llama-agent\ncommit_sha={self.sha}\n")


if __name__ == "__main__":
    unittest.main()
