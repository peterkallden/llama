#!/usr/bin/env python3
"""Explain dose experiments from recorded execution, preserving incomplete data."""
import argparse
import datetime
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
        "schema": "flydelta-dose-run.v2", "run_id": str(uuid.uuid4()), "profile": profile,
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


def phase_evidence(lines):
    """Summarize observed execution, never infer execution from evidence depth."""
    names = ("baseline", "baseline_capture", "direction", "layer_discovery", "whirlpool", "bootstrap_zoom",
        "adaptive_alpha", "orthogonal", "shallow", "deep", "tfo_lite", "utility_gate",
        "donor_capture", "representation_augmentation", "concept_capture", "concept_synthesis",
        "graft", "anchor_resume", "promotion_activation")
    phases = {name: {"status": "not_reached", "reason": "no execution evidence in this trace",
        "evidence_lines": [], "observations": 0} for name in names}
    def observed(name, number, count=1, forced=False):
        if name not in phases:
            return
        item = phases[name]
        item.update(status="forced" if forced else "executed", reason="recorded execution or host decision")
        item["observations"] += count
        item["evidence_lines"].append(number)
    issues, decisions, exchanges = [], [], []
    for number, line in enumerate(lines, 1):
        try:
            if line.startswith("flydelta_trace ") and " json=" in line:
                trace = json.loads(line.split(" json=", 1)[1])
                arms = [arm for arm in trace.get("arms", []) if arm.get("execution_class") in ("full", "diagnostic")]
                phase = trace.get("phase")
                if phase in ("direction", "donor_capture", "concept_capture", "concept_synthesis"):
                    observed(phase, number)
                for whirlpool in trace.get("whirlpool", []):
                    if whirlpool.get("model_evaluations", 0) > 0:
                        observed("whirlpool", number, whirlpool["model_evaluations"])
                # The orchestration plan and its request flags may describe
                # the next slice. Only completed callbacks prove these phases.
                if trace.get("bootstrap_zoom_callback_completed"):
                    observed("bootstrap_zoom", number)
                if trace.get("augmentation_callback_completed"):
                    observed("representation_augmentation", number)
                if trace.get("utility_gate_evaluated"):
                    observed("utility_gate", number)
            elif line.startswith("flydelta_low_rank_arm "):
                arm = json.loads(line.split(" ", 1)[1])
                if arm.get("executed"):
                    observed("deep", number, forced=arm.get("forced_deep") is True)
            elif line.startswith(("flydelta_model_exchange ", "flydelta_daemon_arm ")):
                record = json.loads(line.split(" ", 1)[1])
                record["trace_line"] = number
                exchanges.append(record)
                if record.get("executed") and record.get("phase") == "baseline":
                    observed("baseline", number)
                if (record.get("executed") and record.get("capture_observed") is True and
                        record.get("phase") in ("baseline", "replay_forced_baseline")):
                    observed("baseline_capture", number)
            elif line.startswith(("flydelta_progress_adviser ", "flydelta_progress_matched ",
                                  "flydelta_model_progress_guidance ", "flydelta_progress_continuation ")):
                record = json.loads(line.split(" ", 1)[1])
                decisions.append({"line": number, "event": line.split(" ", 1)[0], "record": record})
            elif line.startswith("flydelta_low_rank_rank_unavailable "):
                phases["deep"].setdefault("unavailable_rank_records", []).append({"line": number, "detail": line})
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            issues.append(f"invalid phase evidence at line {number}: {error}")
    return {"phases": phases, "decisions": decisions, "exchanges": exchanges, "issues": issues,
        "interpretation": "Execution coverage for this trace only. Requests, evidence depth and retained references do not prove later execution."}


def read_condition(root, mode):
    log = root / mode / "low-rank-rank-sweep.log"
    condition = {"trace": str(log), "issues": [], "arms": [], "calibrations": [],
                 "search_runs": [], "response_pairs": [], "execution_contexts": [],
                 "rendered_prompts": [], "baseline_repeats": []}
    issues = condition["issues"]
    timestamps = {}
    try:
        for line in (root / mode / "status.log").read_text().splitlines():
            for key, value in (part.split("=", 1) for part in shlex.split(line) if "=" in part):
                if key in ("condition_started", "condition_finished"):
                    timestamps[key] = datetime.datetime.fromisoformat(value)
    except (OSError, ValueError):
        pass
    condition["total_elapsed_seconds"] = ((timestamps["condition_finished"] - timestamps["condition_started"]).total_seconds()
        if len(timestamps) == 2 else None)
    condition["total_cost_boundary"] = "process wall time including model load, baseline repeats, capture, calibration and search"
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
            if line.startswith("agent_generation_prompt "):
                prompt = json.loads(line.split(" ", 1)[1])
                rendered = prompt["rendered_prompt"]
                if not isinstance(rendered, str) or not isinstance(prompt["messages"], list):
                    raise ValueError("invalid rendered prompt record")
                if prompt["rendered_prompt_fingerprint"] != "sha256:" + hashlib.sha256(rendered.encode()).hexdigest():
                    raise ValueError("rendered prompt fingerprint mismatch")
                prompt["line"] = number
                condition["rendered_prompts"].append(prompt)
            elif line.startswith("flydelta_dose_context "):
                context = json.loads(line.split(" ", 1)[1])
                if not isinstance(context, dict) or not isinstance(context.get("scenario"), str):
                    raise ValueError("invalid effective dose context")
                condition["execution_contexts"].append(context)
            elif line.startswith("flydelta_model_exchange ") and '"baseline_repeat"' in line:
                condition["baseline_repeats"].append(json.loads(line.split(" ", 1)[1]))
            elif line.startswith("flydelta_low_rank_arm "):
                arm = json.loads(line.split(" ", 1)[1])
                if not isinstance(arm, dict):
                    raise ValueError("arm must be an object")
                for field in ("fixture_id", "evaluation_scenario_id", "basis_identity", "outcome"):
                    if not isinstance(arm.get(field), str) or not arm[field]:
                        raise ValueError(f"arm {field} must be a nonempty string")
                mutation = arm.get("mutation")
                if mutation == "" and mode == "raw" and arm.get("host_evaluated") is False:
                    # Older raw diagnostic records did not carry a mutation label.
                    arm["mutation"] = "raw_coefficient_diagnostic"
                elif not isinstance(mutation, str) or not mutation:
                    raise ValueError("arm mutation must be a nonempty string")
                for field in ("requested_rank", "layer"):
                    if type(arm.get(field)) is not int or arm[field] < 0:
                        raise ValueError(f"arm {field} must be a nonnegative integer")
                for field in ("executed", "host_evaluated", "verifier_known"):
                    if type(arm.get(field)) is not bool:
                        raise ValueError(f"arm {field} must be a boolean")
                if arm.get("dose_experiment") != mode:
                    raise ValueError("trace condition mismatch")
                if arm.get("semantic_progress") is not None and not isinstance(arm["semantic_progress"], dict):
                    raise ValueError("semantic progress must be an object or unavailable")
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
    condition["phase_coverage"] = phase_evidence(lines)
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
                "ambiguous_baselines": baseline if len(baseline) > 1 else [],
                "ambiguous_candidates": candidate if len(candidate) > 1 else [],
                "outcome": arm["outcome"], "verifier_known": arm["verifier_known"],
                "verifier_reason": arm.get("verifier_reason"), "semantic_progress": arm.get("semantic_progress"),
                "objective_assessment": arm.get("objective_assessment"),
                "observed_decision": arm.get("observed_decision"),
                "expected_decision": arm.get("expected_decision"),
                "text_changed": (candidate[0]["content"] != baseline[0]["content"]
                                 if len(candidate) == 1 and len(baseline) == 1 else None),
                "dose": {key: arm.get(key) for key in ("proposed_intervention_norm", "proposal_norm_limit_known", "proposal_norm_limited", "dose_action", "requested_coefficients", "coefficients", "dose_safety_limited", "calibration_ref", "requested_coefficient_norm",
                    "executed_coefficient_norm", "requested_residual_fraction", "applied_residual_fraction",
                    "requested_intervention_norm", "applied_intervention_norm", "dose_reason")}, "issues": errors}
        guidance_id = arm.get("guidance_arm_id")
        pair["selection_evidence"] = []
        if guidance_id:
            for decision in condition["phase_coverage"]["decisions"]:
                event, record = decision["event"], decision["record"]
                if event == "flydelta_progress_adviser":
                    for selected in record.get("selected", []):
                        if selected.get("arm_id") == guidance_id:
                            pair["selection_evidence"].append({"line": decision["line"],
                                "wave": record.get("wave"), "stage": record.get("stage"),
                                "reason": selected.get("reason"),
                                "history_rows": record.get("compatible_history_rows"),
                                "posterior_computed": record.get("posterior_computed"),
                                "arm_id": guidance_id})
                elif event == "flydelta_progress_continuation" and record.get("guidance_arm_id") == guidance_id:
                    pair["selection_evidence"].append({"line": decision["line"],
                        "wave": record.get("wave"), "executed": record.get("executed"),
                        "execution_arm_id": record.get("arm_id"), "arm_id": guidance_id})
        pair["rendered_prompts"] = [record for record in condition["rendered_prompts"]
            if pair["baseline"] and record["messages"] == pair["baseline"]["messages"]
            and record["n_predict"] > 0]
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
            else "partial_progress" if any((arm.get("semantic_progress") or {}).get("comparable") and
                (arm.get("semantic_progress") or {}).get("outcome") == "improved" for arm in full) else "unresolved",
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
        if manifest.get("schema") not in ("flydelta-dose-run.v1", "flydelta-dose-run.v2") or manifest.get("trace_required") is not True:
            raise ValueError("missing dose manifest schema or required trace")
        expected_command = None
        expected_environment = None
        for mode in modes:
            command = manifest["commands"][mode]
            environment = {key: value for key, value in command["environment"].items()
                           if key not in ("LLAMA_AGENT_DOSE_EXPERIMENT", "LLAMA_AGENT_LEARNING_LEDGER")}
            if command["environment"].get("LLAMA_AGENT_DOSE_EXPERIMENT", mode) != mode:
                raise ValueError(f"manifest mode mismatch for {mode}")
            if expected_command is not None and (command["argv"] != expected_command or environment != expected_environment):
                raise ValueError("manifest commands or common environment differ between conditions")
            expected_command = command["argv"]
            expected_environment = environment
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
    baseline_responses = {}
    for mode, condition in conditions.items():
        for pair in condition["response_pairs"]:
            if pair["baseline"] is not None:
                key = pair["scenario"]
                messages = pair["baseline"]["messages"]
                if key in prompts and prompts[key] != messages:
                    comparable = False
                    issues.append(f"baseline prompt differs across conditions for {key}")
                prompts[key] = messages
                response = pair["baseline"]["content"]
                if key in baseline_responses and baseline_responses[key] != response:
                    comparable = False
                    issues.append(f"baseline response differs across conditions for {key}")
                baseline_responses[key] = response
    # Legacy logs remain readable, but missing execution identity is not proof
    # of a matched experiment. Compare effective settings, not just CLI flags.
    required_context = ("model_fingerprint", "tokenizer_fingerprint", "template_fingerprint",
        "capture_layout", "intervention_site", "actual_n_ctx", "n_parallel", "n_gpu_layers",
        "cache_type_k", "cache_type_v", "temperature", "seed", "n_predict", "baseline_repeats",
        "candidate_budget", "full_generation_budget", "verifier_revision", "verification_mode")
    context_by_scenario = {}
    rendered_by_scenario = {}
    calibration_by_scenario = {}
    for mode, condition in conditions.items():
        contexts = condition["execution_contexts"]
        expected_scenarios = {arm["evaluation_scenario_id"] for arm in condition["arms"]}
        if {context["scenario"] for context in contexts} != expected_scenarios:
            comparable = False
            issues.append(f"{mode}: effective execution identity missing for one or more scenarios")
        if not contexts:
            comparable = False
            issues.append(f"{mode}: effective execution identity unavailable (legacy trace)")
        for context in contexts:
            if any(field not in context for field in required_context):
                comparable = False
                issues.append(f"{mode}: incomplete effective execution identity")
                continue
            key = context["scenario"]
            identity = {field: context[field] for field in required_context}
            if manifest and identity["model_fingerprint"] != manifest.get("model_fingerprint"):
                comparable = False
                issues.append(f"{mode}: executed model differs from manifest for {key}")
            if key in context_by_scenario and identity != context_by_scenario[key]:
                comparable = False
                issues.append(f"effective execution settings differ for {key}")
            context_by_scenario[key] = identity
            repeat_count = context["baseline_repeats"]
            repeats = [item for item in condition["baseline_repeats"]
                       if str(item.get("id", "")).startswith(key + "/unsteered-repeat/")]
            if (type(repeat_count) is not int or repeat_count < 1 or
                    len(repeats) != repeat_count - 1 or
                    any(item.get("executed") is not True or not isinstance(item.get("content"), str)
                        for item in repeats)):
                comparable = False
                issues.append(f"{mode}: unsteered repeats missing or failed for {key}")
        for pair in condition["response_pairs"]:
            rendered = {record["rendered_prompt_fingerprint"] for record in pair["rendered_prompts"]}
            key = pair["scenario"]
            if len(rendered) != 1:
                comparable = False
                issues.append(f"{mode}: rendered full-generation prompt missing or ambiguous for {key}")
            elif key in rendered_by_scenario and rendered != rendered_by_scenario[key]:
                comparable = False
                issues.append(f"rendered prompt differs for {key}")
            else:
                rendered_by_scenario[key] = rendered
        for calibration in condition["calibrations"]:
            key = calibration["excluded_evaluation_scenario"]
            fields = ("source_scenarios", "sample_norms", "median_norm", "capture_layout_revision", "capture_position", "token_index")
            identity = {field: calibration.get(field) for field in fields}
            if any(identity[field] is None for field in fields):
                comparable = False
                issues.append(f"{mode}: incomplete calibration identity for {key}")
            elif key in calibration_by_scenario and identity != calibration_by_scenario[key]:
                comparable = False
                issues.append(f"calibration measurements differ for {key}")
            calibration_by_scenario[key] = identity
    if any(condition["status"] != "complete" for condition in conditions.values()):
        comparable = False
        issues.append("one or more conditions are incomplete")
    return {"schema": "flydelta-dose-comparison.v1", "production_decision": "pending",
            "status": "incomplete" if issues else "complete", "issues": issues,
            "comparability": "matched" if comparable else "unmatched", "manifest": manifest, "conditions": conditions, "comparison_scope": "effective_execution_and_rendered_prompt"}


def render_markdown(report):
    lines = ["# FlyDelta dose comparison", "", f"Coverage: {report['status']}. Comparability: {report['comparability']}.",
             "Production decision: pending. Raw responses are preserved below.", ""]
    for mode, condition in report["conditions"].items():
        lines.extend([f"## {mode}", "", f"Process: {condition['execution_status']}; coverage: {condition['coverage_status']}", ""])
        lines.extend([f"Search wall time: {condition['search_elapsed_ms']} ms; total process wall time: {condition['total_elapsed_seconds']} s.", ""])
        if condition["baseline_repeats"]:
            lines.extend(["Unsteered repeat outputs (greedy decoding):", ""])
            for repeat in condition["baseline_repeats"]:
                lines.append("    " + json.dumps({key: repeat.get(key) for key in ("id", "executed", "content")}, ensure_ascii=False))
            lines.append("")
        lines.extend(["Phase coverage:", "", "| Phase | Status | Observations |", "|---|---|---|"])
        for phase, evidence in condition["phase_coverage"]["phases"].items():
            lines.append(f"| {phase} | {evidence['status']} | {evidence['observations']} |")
        lines.append("")
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
                          "Progress:", "", json.dumps(pair["semantic_progress"], ensure_ascii=False), "",
                          "Selection evidence:", "", json.dumps(pair["selection_evidence"], ensure_ascii=False), ""])
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
    parser.add_argument("--trace", action="store_true", help="Summarize a single phase trace without dose comparison")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    args = parser.parse_args()
    if args.trace:
        report = phase_evidence(args.root.read_text().splitlines())
        output = args.root.with_suffix(args.root.suffix + ".summary.json")
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(output)
        return 1 if report["issues"] else 0
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
