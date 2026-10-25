#!/usr/bin/env python3
"""Summarize dose traces, preserving failed and incomplete conditions as data."""
import json
import pathlib
import shlex
import sys

MODES = ("raw", "normalized", "direction-dose")


def read_condition(root, mode):
    log = root / mode / "low-rank-rank-sweep.log"
    condition = {"trace": str(log), "issues": [], "arms": [], "calibrations": [], "search_runs": []}
    issues = condition["issues"]
    try:
        condition["process_exit"] = int((root / mode / "process.exit").read_text().strip())
    except (OSError, ValueError):
        condition["process_exit"] = None
        issues.append("process exit not recorded")
    if condition["process_exit"] not in (None, 0):
        issues.append(f"process failed: {condition['process_exit']}")
    try:
        lines = log.read_text().splitlines()
    except OSError as error:
        issues.append(f"trace unavailable: {error}")
        lines = []
    for number, line in enumerate(lines, 1):
        try:
            if line.startswith("flydelta_low_rank_arm "):
                arm = json.loads(line.split(" ", 1)[1])
                if not isinstance(arm, dict):
                    raise ValueError("arm must be an object")
                for field in ("fixture_id", "evaluation_scenario_id", "requested_rank", "layer", "basis_identity", "mutation",
                              "outcome", "executed", "host_evaluated", "verifier_known"):
                    if field not in arm:
                        raise ValueError(f"arm missing {field}")
                for field in ("fixture_id", "evaluation_scenario_id", "basis_identity", "mutation", "outcome"):
                    if not isinstance(arm[field], str) or not arm[field]:
                        raise ValueError(f"arm {field} must be a nonempty string")
                for field in ("requested_rank", "layer"):
                    if type(arm[field]) is not int or arm[field] < 0:
                        raise ValueError(f"arm {field} must be a nonnegative integer")
                for field in ("executed", "host_evaluated", "verifier_known"):
                    if type(arm[field]) is not bool:
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
        except (ValueError, KeyError, TypeError) as error:
            issues.append(f"invalid trace record at line {number}: {error}")
    arms = condition["arms"]
    full = [arm for arm in arms if arm["mutation"] == "full_generation_top_arm"]
    if not arms or not condition["search_runs"]:
        issues.append("missing candidate arms or search timing")
    if mode != "raw" and not condition["calibrations"]:
        issues.append("missing residual calibration")
    condition.update({
        "status": "incomplete" if issues else "complete",
        "diagnostic_and_full_arms": len(arms), "full_arms": len(full),
        "host_verified_full_arms": sum(bool(arm["executed"] and arm["host_evaluated"]
                                            and arm["verifier_known"]) for arm in full),
        "outcomes": {outcome: sum(arm["outcome"] == outcome for arm in full)
                     for outcome in sorted({arm["outcome"] for arm in full})},
        "search_elapsed_ms": sum(run["elapsed_ms"] for run in condition["search_runs"]),
        "cost_boundary": "search wall time; calibration/capture and setup remain in the full trace",
        "interpretation": "no full arms or an unknown verdict provides no verified quality conclusion",
    })
    return condition


def build_report(root):
    conditions = {mode: read_condition(root, mode) for mode in MODES}
    fixture_sets = [{(arm["evaluation_scenario_id"], arm["requested_rank"], arm["layer"], arm["basis_identity"])
                     for arm in conditions[mode]["arms"]} for mode in MODES]
    issues = []
    if not all(fixtures == fixture_sets[0] for fixtures in fixture_sets):
        issues.append("fixture/rank/layer/basis sets differ between conditions")
    if any(condition["status"] != "complete" for condition in conditions.values()):
        issues.append("one or more conditions are incomplete")
    return {"schema": "flydelta-dose-comparison.v1", "production_decision": "pending",
            "status": "incomplete" if issues else "complete", "issues": issues,
            "conditions": conditions}


def main():
    root = pathlib.Path(sys.argv[1])
    report = build_report(root)
    output = root / "dose-comparison.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(output)
    print(output)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
