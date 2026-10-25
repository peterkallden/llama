#!/usr/bin/env python3
"""Summarize persisted dose traces without granting learning or promotion credit."""
import json
import pathlib
import shlex
import sys

root = pathlib.Path(sys.argv[1])
summary = {"schema": "flydelta-dose-comparison.v1", "production_decision": "pending", "conditions": {}}
fixture_sets = []
for mode in ("raw", "normalized", "direction-dose"):
    log = root / mode / "low-rank-rank-sweep.log"
    arms, calibrations, runs = [], [], []
    for line in log.read_text().splitlines():
        if line.startswith("flydelta_low_rank_arm "):
            arms.append(json.loads(line.split(" ", 1)[1]))
        elif line.startswith("flydelta_low_rank_rank_sweep "):
            runs.append(dict(part.split("=", 1) for part in shlex.split(line)[1:] if "=" in part))
        elif line.startswith("flydelta_dose_calibration "):
            calibrations.append(json.loads(line.split(" ", 1)[1]))
    if not arms:
        raise SystemExit(f"comparison incomplete: no model arms for {mode}")
    full = [arm for arm in arms if arm["mutation"] == "full_generation_top_arm"]
    if not full or not runs:
        raise SystemExit(f"comparison incomplete: no full verification or search timing for {mode}")
    fixture_sets.append({(arm["fixture_id"], arm["requested_rank"], arm["layer"], arm["basis_identity"]) for arm in arms})
    if mode != "raw" and not calibrations:
        raise SystemExit(f"comparison incomplete: no calibration for {mode}")
    for calibration in calibrations:
        if calibration["excluded_evaluation_scenario"] in calibration["source_scenarios"]:
            raise SystemExit("calibration overlaps its evaluation scenario")
    summary["conditions"][mode] = {
        "trace": str(log), "diagnostic_and_full_arms": len(arms), "full_arms": len(full),
        "outcomes": {outcome: sum(arm["outcome"] == outcome for arm in full)
                     for outcome in sorted({arm["outcome"] for arm in full})},
        "calibrations": calibrations, "arms": arms, "search_runs": runs,
        "search_elapsed_ms": sum(int(run["elapsed_ms"]) for run in runs),
        "cost_boundary": "search wall time; calibration/capture and setup remain in the full trace",
        "interpretation": "geometric diagnostics are not host-verified improvement",
    }
if not all(fixtures == fixture_sets[0] for fixtures in fixture_sets):
    raise SystemExit("comparison incomplete: fixture/rank sets differ")
output = root / "dose-comparison.json"
output.write_text(json.dumps(summary, indent=2) + "\n")
print(output)
