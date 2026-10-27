#!/usr/bin/env python3
"""Explain dose experiments from recorded execution, preserving incomplete data."""
import argparse
import hashlib
import json
import os
import pathlib
import shlex
import subprocess
import sys
import uuid

MODES = ("raw", "normalized", "direction-dose")


def digest_file(path):
    value = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return "sha256:" + value.hexdigest()


def write_manifest(root, profile, repo, binary, arguments):
    modes = ("raw", "direction-dose") if profile == "paired" else MODES
    def argument(name):
        return arguments[arguments.index(name) + 1]
    def git(*args):
        return subprocess.check_output(["git", "-C", repo, *args])
    manifest = {
        "schema": "flydelta-dose-run.v1", "run_id": str(uuid.uuid4()), "profile": profile,
        "modes": list(modes), "trace_required": True, "timeout_seconds": None,
        "requirements": ["full_generation_baseline", "applied_full_generation_candidate",
                         "host_evaluation", "matched_prompts", "complete_response_trace"],
        "model": argument("--model"),
        "model_fingerprint": os.environ["LLAMA_AGENT_DOSE_MODEL_FINGERPRINT"],
        "binary": binary, "binary_fingerprint": digest_file(binary),
        "suite": argument("--suite"), "suite_fingerprint": digest_file(argument("--suite")),
        "revision": git("rev-parse", "HEAD").decode().strip(),
        "diff_fingerprint": "sha256:" + hashlib.sha256(git("diff", "--binary", "HEAD")).hexdigest(),
        "commands": {mode: {"argv": [binary, *arguments], "environment": {
            "LLAMA_AGENT_DOSE_EXPERIMENT": mode, "LLAMA_AGENT_RESIDENT_TRACE": "1",
            "GGML_VK_VISIBLE_DEVICES": os.environ.get("GGML_VK_VISIBLE_DEVICES"),
            "LLAMA_AGENT_LEARNING_LEDGER": str(root / mode / "learning-ledger.jsonl")}} for mode in modes},
    }
    (root / "run-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def _exchange(record, source):
    return {"arm_id": record.get("arm_id", record.get("id")), "source": source,
            "messages": record.get("messages"), "content": record.get("content"),
            "overlay_requested": record.get("overlay_requested"),
            "overlay_applied": record.get("overlay_applied"),
            "executed": record.get("executed"),
            "host_evaluated": record.get("host_evaluated"), "verifier_known": record.get("verifier_known"),
            "verifier_passed": record.get("verifier_passed"),
            "stop_reason": record.get("stop_reason"), "decoded_tokens": record.get("decoded_tokens"),
            "error": record.get("failure_reason", record.get("error", ""))}


def read_condition(root, mode):
    log = root / mode / "low-rank-rank-sweep.log"
    condition = {"trace": str(log), "issues": [], "arms": [], "calibrations": [],
                 "search_runs": [], "response_pairs": []}
    issues = condition["issues"]
    try:
        condition["process_exit"] = int((root / mode / "process.exit").read_text().strip())
    except (OSError, ValueError):
        condition["process_exit"] = None
        issues.append("process exit not recorded")
    code = condition["process_exit"]
    condition["execution_status"] = ("interrupted" if code in (130, 143) else
                                     "completed" if code == 0 else "unrecorded" if code is None else "failed")
    if code not in (None, 0):
        issues.append(f"process failed: {code}")
    execution_issue_count = len(issues)
    try:
        lines = log.read_text().splitlines()
    except OSError as error:
        issues.append(f"trace unavailable: {error}")
        lines = []
    exchanges = {}
    for number, line in enumerate(lines, 1):
        try:
            if line.startswith("flydelta_low_rank_arm "):
                arm = json.loads(line.split(" ", 1)[1])
                if not isinstance(arm, dict):
                    raise ValueError("arm must be an object")
                for field in ("fixture_id", "evaluation_scenario_id", "basis_identity", "mutation", "outcome"):
                    if not isinstance(arm.get(field), str) or not arm[field]:
                        raise ValueError(f"arm {field} must be a nonempty string")
                for field in ("requested_rank", "layer"):
                    if type(arm.get(field)) is not int or arm[field] < 0:
                        raise ValueError(f"arm {field} must be a nonnegative integer")
                for field in ("executed", "host_evaluated", "verifier_known"):
                    if type(arm.get(field)) is not bool:
                        raise ValueError(f"arm {field} must be a boolean")
                if arm.get("dose_experiment") != mode:
                    raise ValueError("trace condition mismatch")
                condition["arms"].append(arm)
            elif line.startswith("flydelta_low_rank_rank_sweep "):
                run = dict(part.split("=", 1) for part in shlex.split(line)[1:] if "=" in part)
                run["elapsed_ms"] = int(run["elapsed_ms"])
                if run["elapsed_ms"] < 0:
                    raise ValueError("negative elapsed time")
                condition["search_runs"].append(run)
            elif line.startswith("flydelta_dose_calibration "):
                calibration = json.loads(line.split(" ", 1)[1])
                if calibration["excluded_evaluation_scenario"] in calibration["source_scenarios"]:
                    issues.append("calibration overlaps its evaluation scenario")
                condition["calibrations"].append(calibration)
            elif line.startswith(("flydelta_daemon_arm ", "flydelta_model_exchange ")):
                prefix, text = line.split(" ", 1)
                record = json.loads(text)
                if record.get("phase") not in ("full", "deep_full"):
                    continue  # Canonical repair echoes and capture-only scoring are different observations.
                exchange = _exchange(record, prefix)
                if not isinstance(exchange["arm_id"], str) or not exchange["arm_id"]:
                    raise ValueError("full response missing arm identity")
                if not isinstance(exchange["messages"], list) or not exchange["messages"]:
                    raise ValueError("full response missing prompt messages")
                if not isinstance(exchange["content"], str):
                    raise ValueError("full response missing model content")
                exchange["line"] = number
                bucket = exchanges.setdefault(exchange["arm_id"], [])
                if not any(all(old[k] == exchange[k] for k in ("messages", "content", "overlay_applied", "overlay_requested")) for old in bucket):
                    bucket.append(exchange)
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            issues.append(f"invalid trace record at line {number}: {error}")
    arms = condition["arms"]
    full = [arm for arm in arms if arm["mutation"] == "full_generation_top_arm"]
    if not arms or not condition["search_runs"]:
        issues.append("missing candidate arms or search timing")
    if not full:
        issues.append("no full-generation candidate arms")
    if mode != "raw" and not condition["calibrations"]:
        issues.append("missing residual calibration")
    for arm in full:
        candidate = exchanges.get(arm.get("raw_output_trace_id", arm.get("arm_id")), [])
        baseline = exchanges.get(arm.get("baseline_arm_id"), [])
        errors = []
        if len(candidate) != 1 or len(baseline) != 1:
            errors.append("missing or ambiguous baseline/candidate response identity")
        else:
            if candidate[0]["messages"] != baseline[0]["messages"]:
                errors.append("baseline and candidate prompts differ")
            if baseline[0]["overlay_requested"] is not False or baseline[0]["overlay_applied"] is not False:
                errors.append("baseline is not confirmed unsteered")
            if candidate[0]["overlay_requested"] is not True or candidate[0]["overlay_applied"] is not True:
                errors.append("candidate overlay is not confirmed applied")
            if not candidate[0]["executed"] or not baseline[0]["executed"] or candidate[0]["error"] or baseline[0]["error"]:
                errors.append("baseline or candidate execution failed")
        if not arm["executed"] or not arm["host_evaluated"]:
            errors.append("candidate not executed and host evaluated")
        pair = {"scenario": arm["evaluation_scenario_id"], "task": arm.get("task"),
                "arm_id": arm.get("arm_id"), "baseline_arm_id": arm.get("baseline_arm_id"),
                "baseline": baseline[0] if len(baseline) == 1 else None,
                "candidate": candidate[0] if len(candidate) == 1 else None,
                "outcome": arm["outcome"], "verifier_known": arm["verifier_known"],
                "verifier_reason": arm.get("verifier_reason"), "semantic_progress": arm.get("semantic_progress"),
                "objective_assessment": arm.get("objective_assessment"),
                "dose": {key: arm.get(key) for key in ("coefficients", "calibration_ref", "requested_coefficient_norm",
                    "executed_coefficient_norm", "requested_residual_fraction", "applied_residual_fraction",
                    "requested_intervention_norm", "applied_intervention_norm", "dose_reason")}, "issues": errors}
        condition["response_pairs"].append(pair)
        issues.extend(f"arm {arm.get('arm_id')}: {error}" for error in errors)
    condition.update({
        "status": "incomplete" if issues else "complete",
        "coverage_status": "incomplete" if issues[execution_issue_count:] else "complete",
        "diagnostic_and_full_arms": len(arms), "full_arms": len(full),
        "host_verified_full_arms": sum(bool(arm["executed"] and arm["host_evaluated"] and arm["verifier_known"]) for arm in full),
        "outcomes": {outcome: sum(arm["outcome"] == outcome for arm in full) for outcome in sorted({arm["outcome"] for arm in full})},
        "search_elapsed_ms": sum(run["elapsed_ms"] for run in condition["search_runs"]),
        "cost_boundary": "search wall time; calibration/capture and setup remain in the full trace",
        "behavior_status": "paired_outcome_available" if any(arm["outcome"].lower() in ("helped", "harmed", "neutral") for arm in full)
            else "partial_progress" if any(arm.get("semantic_progress", {}).get("comparable") and
                arm.get("semantic_progress", {}).get("outcome") == "improved" for arm in full) else "unresolved",
        "interpretation": "Inspect raw responses and progress separately from the paired host outcome; no production adoption is inferred.",
    })
    return condition


def build_report(root, modes=MODES):
    if not modes or len(set(modes)) != len(modes) or any(mode not in MODES for mode in modes):
        raise ValueError("invalid requested dose conditions")
    conditions = {mode: read_condition(root, mode) for mode in modes}
    manifest_issues = []
    manifest = None
    try:
        manifest = json.loads((root / "run-manifest.json").read_text())
        if manifest.get("schema") != "flydelta-dose-run.v1" or manifest.get("trace_required") is not True:
            raise ValueError("missing dose manifest schema or required trace")
        for mode in modes:
            command = manifest["commands"][mode]
            if "--dose-comparison" not in command["argv"] or any(flag in command["argv"] for flag in
                    ("--low-rank-rank-sweep", "--force-tfo-lite")):
                raise ValueError("manifest does not request the dedicated dose comparison")
            if command["environment"].get("LLAMA_AGENT_RESIDENT_TRACE") != "1":
                raise ValueError("manifest did not enable required trace")
        for field in ("model_fingerprint", "binary_fingerprint", "suite_fingerprint"):
            if not isinstance(manifest.get(field), str) or not manifest[field].startswith("sha256:"):
                raise ValueError(f"manifest missing {field}")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        manifest_issues.append(f"manifest unavailable or invalid: {error}")
    fixture_sets = [{(arm["evaluation_scenario_id"], arm["requested_rank"], arm["layer"], arm["basis_identity"])
                     for arm in conditions[mode]["arms"]} for mode in modes]
    issues = list(manifest_issues)
    comparable = not manifest_issues and bool(fixture_sets[0]) and all(fixtures == fixture_sets[0] for fixtures in fixture_sets)
    if not comparable:
        issues.append("fixture/rank/layer/basis sets differ or are absent between conditions")
    # Compare actual unsteered prompts across conditions as well as within each pair.
    prompts = {}
    for mode, condition in conditions.items():
        for pair in condition["response_pairs"]:
            if pair["baseline"] is not None:
                key = pair["scenario"]
                messages = pair["baseline"]["messages"]
                if key in prompts and prompts[key] != messages:
                    comparable = False
                    issues.append(f"baseline prompt differs across conditions for {key}")
                prompts[key] = messages
    if any(condition["status"] != "complete" for condition in conditions.values()):
        issues.append("one or more conditions are incomplete")
    return {"schema": "flydelta-dose-comparison.v1", "production_decision": "pending",
            "status": "incomplete" if issues else "complete", "issues": issues,
            "comparability": "matched" if comparable else "unmatched", "manifest": manifest, "conditions": conditions}


def render_markdown(report):
    lines = ["# FlyDelta dose comparison", "", f"Coverage: {report['status']}. Comparability: {report['comparability']}.",
             "Production decision: pending. Raw responses are preserved below.", ""]
    for mode, condition in report["conditions"].items():
        lines.extend([f"## {mode}", "", f"Process: {condition['execution_status']}; coverage: {condition['coverage_status']}", ""])
        if condition["response_pairs"]:
            lines.extend(["| Scenario | Arm | Host outcome | Before | With arm |", "|---|---|---|---|---|"])
            def cell(value):
                return str(value).replace("&", "&amp;").replace("<", "&lt;").replace("|", "&#124;").replace("\n", "<br>")
            for pair in condition["response_pairs"]:
                before = pair["baseline"]["content"] if pair["baseline"] else "Unavailable"
                after = pair["candidate"]["content"] if pair["candidate"] else "Unavailable"
                lines.append("| " + " | ".join(cell(item) for item in (pair["scenario"], pair["arm_id"], pair["outcome"], before, after)) + " |")
            lines.append("")
        for pair in condition["response_pairs"]:
            lines.extend([f"### {pair['scenario']} / {pair['arm_id']}", "", f"Task: {pair['task']}", "",
                          f"Host outcome: {pair['outcome']}; verifier known: {pair['verifier_known']}",
                          f"Reason: {pair['verifier_reason']}", "", "Dose:", "", json.dumps(pair["dose"], ensure_ascii=False), "",
                          "Progress:", "", json.dumps(pair["semantic_progress"], ensure_ascii=False), ""])
            for label, key in (("Before", "baseline"), ("With arm", "candidate")):
                item = pair[key]
                lines.extend([f"**{label}**", ""])
                if item is None:
                    lines.extend(["Response unavailable; see trace issues.", ""])
                else:
                    # Four-space indentation preserves arbitrary model content, including markdown fences.
                    lines.extend("    " + line for line in item["content"].splitlines())
                    lines.extend(["", f"Prompt and response: trace line {item['line']}.", ""])
        if condition["issues"]:
            lines.extend(["Issues:", "", *[f"- {issue}" for issue in condition["issues"]], ""])
    return "\n".join(lines) + "\n"


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--manifest":
        write_manifest(pathlib.Path(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6:])
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=pathlib.Path)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    args = parser.parse_args()
    report = build_report(args.root, args.modes)
    output = args.root / "dose-comparison.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(output)
    (args.root / "dose-comparison.md").write_text(render_markdown(report))
    print(output)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
