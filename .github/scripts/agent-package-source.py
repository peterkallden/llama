#!/usr/bin/env python3
"""Resolve a development package's verified source, not its workflow_run HEAD."""

import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


def gh_json(*args):
    return json.loads(subprocess.check_output(["gh", *args], text=True))


def validate_source(source, package_run_id, repository, get_run):
    if source.get("version") != 1:
        raise ValueError("Unsupported package source metadata version")
    if str(source.get("package_workflow_run_id")) != str(package_run_id):
        raise ValueError("Package source metadata belongs to another run")
    sha = source.get("commit_sha", "")
    if source.get("source_branch") != "feature/llama-agent" or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Invalid development package source branch or commit")
    checks = [
        (package_run_id, "agent-package-archive.yml", False),
        (source.get("source_ci_run_id"), "agent-ci.yml", True),
        (source.get("sanitizer_run_id"), "agent-analysis.yml", True),
    ]
    for run_id, workflow, is_verification in checks:
        if not str(run_id).isdigit():
            raise ValueError("Missing or invalid verification run ID")
        run = get_run(run_id)
        if (run.get("path", "").split("@", 1)[0] != f".github/workflows/{workflow}"
                or run.get("status") != "completed" or run.get("conclusion") != "success"
                or run.get("repository", {}).get("full_name") != repository
                or run.get("head_repository", {}).get("full_name") != repository):
            raise ValueError(f"Untrusted or unsuccessful workflow run {run_id}")
        if is_verification:
            if (run.get("head_branch") != source["source_branch"] or run.get("head_sha") != sha
                    or run.get("event") not in ("push", "workflow_dispatch")):
                raise ValueError(f"Verification run {run_id} does not match the package source")
        elif run.get("event") not in ("workflow_run", "workflow_dispatch"):
            raise ValueError("Package run must originate from verification or manual dispatch")
    return source


def main():
    repository = os.environ["GITHUB_REPOSITORY"]
    run_id = os.environ["PACKAGE_RUN_ID"]
    if not run_id.isdigit():
        raise ValueError("A successful development archive run ID is required")
    with tempfile.TemporaryDirectory(prefix="agent-package-source-", dir=os.environ["RUNNER_TEMP"]) as directory:
        subprocess.run([
            "gh", "run", "download", run_id, "--repo", repository,
            "--name", "agent-development-source", "--dir", directory,
        ], check=True)
        source = json.loads((Path(directory) / "agent-package-source.json").read_text(encoding="utf-8"))
    source = validate_source(source, run_id, repository,
                             lambda value: gh_json("api", f"repos/{repository}/actions/runs/{value}"))
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        for key in ("source_branch", "commit_sha"):
            output.write(f"{key}={source[key]}\n")
    print(f"Verified package source: {source['source_branch']} at {source['commit_sha']} (archive run {run_id})")


if __name__ == "__main__":
    main()
