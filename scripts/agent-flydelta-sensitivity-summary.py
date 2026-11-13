#!/usr/bin/env python3
"""Report matched model evidence; missing execution is inconclusive, never success."""
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
report = {"schema": "flydelta-sensitivity-comparison.v1", "conditions": {}}
keys = []
for condition in ("control", "enabled"):
    arms, events = [], []
    for line in (root / f"{condition}.log").read_text().splitlines():
        for prefix, target in (("flydelta_low_rank_arm ", arms), ("flydelta_sensitivity_trace ", events)):
            if line.startswith(prefix):
                target.append(json.loads(line[len(prefix):]))
    diagnostic = [a for a in arms if a["execution_class"] == "diagnostic" and a["executed"]]
    full = [a for a in arms if a["execution_class"] == "full" and a["executed"]]
    helped = [a for a in full if a.get("counterfactual_credit_eligible")]
    proposals = [a for a in diagnostic if a.get("mutation") == "sensitivity_proposal"]
    report["conditions"][condition] = {
        "diagnostic_arms": len(diagnostic), "full_arms": len(full),
        "verified_helped": len(helped), "executed_sensitivity_proposals": len(proposals),
        "component_trace": events,
        "best_diagnostic_margin": max((a["margin_normalized"] for a in diagnostic if "margin_normalized" in a), default=None),
        "known_harmed": sum(a.get("outcome", "").lower() == "harmed" for a in arms),
    }
    keys.append({(a["group"], a["evaluation_scenario_id"], a["basis_identity"]) for a in diagnostic})
report["matched_fixtures_and_bases"] = bool(keys[0]) and keys[0] == keys[1]
report["status"] = "observed" if report["matched_fixtures_and_bases"] and report["conditions"]["enabled"]["executed_sensitivity_proposals"] else "inconclusive"
report["efficiency_claim"] = "Not established: fixed-budget staged runs do not measure early stopping to HELPED."
(root / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
sys.exit(0 if report["status"] == "observed" else 77)
