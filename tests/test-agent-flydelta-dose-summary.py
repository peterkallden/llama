"""Offline contracts for dose trace aggregation; never starts inference."""
import importlib.util
import json
import pathlib
import tempfile
import unittest

module_path = pathlib.Path(__file__).resolve().parents[1] / "scripts/agent-flydelta-dose-summary.py"
spec = importlib.util.spec_from_file_location("dose_summary", module_path)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


class DoseSummaryContracts(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = pathlib.Path(self.directory.name)
        manifest = {"schema": "flydelta-dose-run.v1", "trace_required": True,
                    "model_fingerprint": "sha256:model", "binary_fingerprint": "sha256:binary",
                    "suite_fingerprint": "sha256:suite", "commands": {mode: {"argv": ["smoke", "--dose-comparison"],
                    "environment": {"LLAMA_AGENT_RESIDENT_TRACE": "1"}} for mode in summary.MODES}}
        (self.root / "run-manifest.json").write_text(json.dumps(manifest))
        for mode in summary.MODES:
            folder = self.root / mode
            folder.mkdir()
            (folder / "process.exit").write_text("0\n")
            arm = {"fixture_id": "resource://" + mode, "evaluation_scenario_id": "scenario-1",
                   "requested_rank": 1, "layer": 2, "basis_identity": "basis-1",
                   "mutation": "full_generation_top_arm", "outcome": "UNKNOWN",
                   "executed": True, "host_evaluated": True, "verifier_known": False,
                   "dose_experiment": mode, "requested_coefficients": [0.5],
                   "coefficients": [0.25], "dose_safety_limited": True}
            arm.update({"arm_id": "arm:1", "raw_output_trace_id": "arm:1", "baseline_arm_id": "baseline:1",
                        "task": "Calculate regional totals"})
            baseline = {"phase": "full", "arm_id": "baseline:1", "executed": True,
                        "messages": [{"role": "user", "content": "Calculate regional totals"}],
                        "content": "before", "overlay_requested": False, "overlay_applied": False}
            candidate = {**baseline, "arm_id": "arm:1", "content": "after",
                         "overlay_requested": True, "overlay_applied": True}
            records = ["flydelta_daemon_arm " + json.dumps(baseline),
                       "flydelta_daemon_arm " + json.dumps(candidate),
                       "flydelta_low_rank_arm " + json.dumps(arm),
                       "flydelta_low_rank_rank_sweep group=test elapsed_ms=12"]
            if mode != "raw":
                records.append("flydelta_dose_calibration " + json.dumps({
                    "excluded_evaluation_scenario": "scenario-1", "source_scenarios": ["scenario-2"]}))
            (folder / "low-rank-rank-sweep.log").write_text("\n".join(records) + "\n")

    def test_rank_sweep_manifest_cannot_pass_as_dose_comparison(self):
        manifest = json.loads((self.root / "run-manifest.json").read_text())
        manifest["commands"]["raw"]["argv"] = ["smoke", "--low-rank-rank-sweep"]
        (self.root / "run-manifest.json").write_text(json.dumps(manifest))
        self.assertEqual(summary.build_report(self.root)["status"], "incomplete")

    def test_paired_profile_does_not_require_normalized_condition(self):
        report = summary.build_report(self.root, ("raw", "direction-dose"))
        self.assertEqual(report["status"], "complete")
        self.assertNotIn("normalized", report["conditions"])
        pair = report["conditions"]["raw"]["response_pairs"][0]
        self.assertEqual(pair["baseline"]["content"], "before")
        self.assertEqual(pair["candidate"]["content"], "after")
        self.assertEqual(pair["dose"]["requested_coefficients"], [0.5])
        self.assertEqual(pair["dose"]["coefficients"], [0.25])
        self.assertTrue(pair["dose"]["dose_safety_limited"])
        self.assertIn("| Scenario |", summary.render_markdown(report))

    def test_missing_baseline_is_incomplete_even_with_successful_process(self):
        path = self.root / "raw/low-rank-rank-sweep.log"
        path.write_text("\n".join(line for line in path.read_text().splitlines()
                                if not line.startswith("flydelta_daemon_arm ") or '"arm_id": "baseline:1"' not in line))
        report = summary.build_report(self.root)
        self.assertEqual(report["conditions"]["raw"]["execution_status"], "completed")
        self.assertEqual(report["status"], "incomplete")

    def test_repair_echo_cannot_substitute_for_arm_response(self):
        path = self.root / "raw/low-rank-rank-sweep.log"
        path.write_text(path.read_text().replace('"phase": "full", "arm_id": "arm:1"',
                                                '"phase": "repair", "arm_id": "arm:1"'))
        self.assertEqual(summary.build_report(self.root)["status"], "incomplete")

    def test_mismatched_prompts_are_not_comparable(self):
        path = self.root / "direction-dose/low-rank-rank-sweep.log"
        path.write_text(path.read_text().replace('"content": "Calculate regional totals"', '"content": "Another question"'))
        report = summary.build_report(self.root)
        self.assertEqual(report["comparability"], "unmatched")
        self.assertEqual(report["status"], "incomplete")

    def test_interrupted_condition_preserves_raw_responses(self):
        (self.root / "raw/process.exit").write_text("130\n")
        report = summary.build_report(self.root)
        self.assertEqual(report["conditions"]["raw"]["execution_status"], "interrupted")
        self.assertEqual(report["conditions"]["raw"]["coverage_status"], "complete")
        self.assertEqual(report["conditions"]["raw"]["response_pairs"][0]["candidate"]["content"], "after")

    def test_unknown_host_outcome_keeps_comparable_partial_progress(self):
        path = self.root / "raw/low-rank-rank-sweep.log"
        records = path.read_text().splitlines()
        for index, line in enumerate(records):
            if line.startswith("flydelta_low_rank_arm "):
                arm = json.loads(line.split(" ", 1)[1])
                arm["semantic_progress"] = {"outcome": "improved", "comparable": True, "regression_known": False}
                records[index] = "flydelta_low_rank_arm " + json.dumps(arm)
        path.write_text("\n".join(records))
        self.assertEqual(summary.build_report(self.root)["conditions"]["raw"]["behavior_status"], "partial_progress")

    def test_scope_local_resource_references_do_not_invalidate_same_scenario(self):
        report = summary.build_report(self.root)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["production_decision"], "pending")
        self.assertEqual(report["conditions"]["raw"]["host_verified_full_arms"], 0)

    def test_failure_keeps_successful_condition_data(self):
        (self.root / "normalized/process.exit").write_text("124\n")
        report = summary.build_report(self.root)
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["conditions"]["raw"]["status"], "complete")
        self.assertEqual(report["conditions"]["normalized"]["process_exit"], 124)

    def test_missing_trace_and_corruption_are_reported(self):
        (self.root / "normalized/low-rank-rank-sweep.log").unlink()
        with (self.root / "direction-dose/low-rank-rank-sweep.log").open("a") as output:
            output.write("flydelta_low_rank_arm {malformed}\n")
        report = summary.build_report(self.root)
        self.assertEqual(report["status"], "incomplete")
        self.assertTrue(report["conditions"]["normalized"]["issues"])
        self.assertTrue(report["conditions"]["direction-dose"]["issues"])

    def test_invalid_record_types_do_not_discard_other_conditions(self):
        path = self.root / "normalized/low-rank-rank-sweep.log"
        path.write_text(path.read_text().replace('"outcome": "UNKNOWN"', '"outcome": []'))
        report = summary.build_report(self.root)
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["conditions"]["raw"]["status"], "complete")

    def test_changed_basis_is_not_a_matched_comparison(self):
        path = self.root / "direction-dose/low-rank-rank-sweep.log"
        path.write_text(path.read_text().replace("basis-1", "basis-2"))
        self.assertEqual(summary.build_report(self.root)["status"], "incomplete")

    def test_changed_baseline_is_not_a_matched_comparison(self):
        path = self.root / "normalized/low-rank-rank-sweep.log"
        path.write_text(path.read_text().replace('"content": "before"', '"content": "different"'))
        self.assertEqual(summary.build_report(self.root)["comparability"], "unmatched")

    def test_legacy_raw_diagnostic_with_empty_mutation_is_preserved(self):
        path = self.root / "raw/low-rank-rank-sweep.log"
        records = path.read_text().splitlines()
        full = next(json.loads(line.split(" ", 1)[1]) for line in records
                    if line.startswith("flydelta_low_rank_arm "))
        diagnostic = {**full, "mutation": "", "host_evaluated": False,
                      "arm_id": "diagnostic:1"}
        path.write_text("\n".join(records + ["flydelta_low_rank_arm " + json.dumps(diagnostic)]) + "\n")
        condition = summary.build_report(self.root)["conditions"]["raw"]
        self.assertEqual(condition["status"], "complete")
        self.assertEqual(condition["full_arms"], 1)
        self.assertEqual(condition["arms"][-1]["mutation"], "raw_coefficient_diagnostic")

    def test_empty_mutation_on_host_evaluated_arm_is_invalid(self):
        path = self.root / "raw/low-rank-rank-sweep.log"
        path.write_text(path.read_text().replace('"mutation": "full_generation_top_arm"', '"mutation": ""'))
        self.assertEqual(summary.build_report(self.root)["conditions"]["raw"]["status"], "incomplete")

    def test_calibration_overlap_is_rejected(self):
        path = self.root / "normalized/low-rank-rank-sweep.log"
        path.write_text(path.read_text().replace('"source_scenarios": ["scenario-2"]',
                                                  '"source_scenarios": ["scenario-1"]'))
        self.assertEqual(summary.build_report(self.root)["status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
