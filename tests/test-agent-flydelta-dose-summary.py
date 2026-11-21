"""Offline contracts for dose trace aggregation; never starts inference."""
import hashlib
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
            context = {field: "test" for field in (
                "model_fingerprint", "tokenizer_fingerprint", "template_fingerprint", "capture_layout",
                "intervention_site", "actual_n_ctx", "n_parallel", "n_gpu_layers", "cache_type_k", "cache_type_v",
                "temperature", "seed", "n_predict", "baseline_repeats", "candidate_budget", "full_generation_budget",
                "verifier_revision", "verification_mode")}
            context["scenario"] = "scenario-1"
            context["model_fingerprint"] = "sha256:model"
            context["baseline_repeats"] = 1
            rendered = {"messages": baseline["messages"], "n_predict": 32, "rendered_prompt": "rendered",
                "rendered_prompt_fingerprint": "sha256:" + hashlib.sha256(b"rendered").hexdigest()}
            records = ["flydelta_dose_context " + json.dumps(context),
                       "agent_generation_prompt " + json.dumps(rendered),
                       "flydelta_daemon_arm " + json.dumps(baseline),
                       "flydelta_daemon_arm " + json.dumps(candidate),
                       "flydelta_low_rank_arm " + json.dumps(arm),
                       "flydelta_low_rank_rank_sweep group=test elapsed_ms=12"]
            if mode != "raw":
                records.append("flydelta_dose_calibration " + json.dumps({
                    "excluded_evaluation_scenario": "scenario-1", "source_scenarios": ["scenario-2"], "sample_norms": [2.0], "median_norm": 2.0,
                    "capture_layout_revision": "layout", "capture_position": "generation_boundary", "token_index": -1}))
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

    def test_context_topology_difference_is_unmatched(self):
        path = self.root / "normalized/low-rank-rank-sweep.log"
        path.write_text(path.read_text().replace('"n_parallel": "test"', '"n_parallel": 4'))
        self.assertEqual(summary.build_report(self.root)["comparability"], "unmatched")

    def test_missing_effective_context_preserves_responses_but_is_unmatched(self):
        path = self.root / "raw/low-rank-rank-sweep.log"
        path.write_text("\n".join(line for line in path.read_text().splitlines()
            if not line.startswith("flydelta_dose_context ")))
        report = summary.build_report(self.root)
        self.assertEqual(report["comparability"], "unmatched")
        self.assertEqual(report["conditions"]["raw"]["response_pairs"][0]["candidate"]["content"], "after")

    def test_calibration_difference_is_unmatched(self):
        path = self.root / "normalized/low-rank-rank-sweep.log"
        path.write_text(path.read_text().replace('"median_norm": 2.0', '"median_norm": 3.0'))
        self.assertEqual(summary.build_report(self.root)["comparability"], "unmatched")

    def test_evidence_depth_does_not_prove_bootstrap_execution(self):
        report = summary.phase_evidence(["flydelta_depth_summary depth=bootstrap bootstrap_overlay_evaluated=no"])
        self.assertEqual(report["phases"]["bootstrap_zoom"]["status"], "not_reached")

    def test_planned_bootstrap_and_arms_do_not_prove_refinement(self):
        trace = {"phase": "search_pipeline", "persisted_experiment_phase": "bootstrap",
                 "bootstrap_refinement": "bootstrap_zoom", "arms": [{"execution_class": "full"}]}
        report = summary.phase_evidence(["flydelta_trace group=test json=" + json.dumps(trace)])
        self.assertEqual(report["phases"]["bootstrap_zoom"]["status"], "not_reached")
        self.assertEqual(report["phases"]["deep"]["status"], "not_reached")

    def test_completed_bootstrap_callback_is_phase_evidence(self):
        trace = {"phase": "search_pipeline", "bootstrap_zoom_callback_completed": True, "arms": []}
        report = summary.phase_evidence(["flydelta_trace group=test json=" + json.dumps(trace)])
        self.assertEqual(report["phases"]["bootstrap_zoom"]["status"], "executed")

    def test_unavailable_rank_four_is_not_a_failure_of_deep(self):
        report = summary.phase_evidence(["flydelta_low_rank_rank_unavailable requested_rank=4 reason=insufficient_independent_evidence"])
        self.assertEqual(report["phases"]["deep"]["status"], "not_reached")
        self.assertEqual(len(report["phases"]["deep"]["unavailable_rank_records"]), 1)

    def test_recorded_utility_gate_is_observed(self):
        trace = {"phase": "search_pipeline", "utility_gate_evaluated": True}
        report = summary.phase_evidence(["flydelta_trace group=test json=" + json.dumps(trace)])
        self.assertEqual(report["phases"]["utility_gate"]["status"], "executed")

    def test_next_plan_is_not_evidence_for_deep(self):
        trace = {"phase": "search_pipeline", "persisted_experiment_phase": "deep_controls",
                 "persisted_tfo_lite_requested": True, "arms": [{"execution_class": "diagnostic"}]}
        report = summary.phase_evidence(["flydelta_trace group=test json=" + json.dumps(trace)])
        self.assertEqual(report["phases"]["deep"]["status"], "not_reached")
        self.assertEqual(report["phases"]["tfo_lite"]["status"], "not_reached")

    def test_generation_does_not_prove_a_capture(self):
        record = {"phase": "baseline", "executed": True}
        report = summary.phase_evidence(["flydelta_model_exchange " + json.dumps(record)])
        self.assertEqual(report["phases"]["baseline"]["status"], "executed")
        self.assertEqual(report["phases"]["baseline_capture"]["status"], "not_reached")
        record["capture_observed"] = True
        report = summary.phase_evidence(["flydelta_model_exchange " + json.dumps(record)])
        self.assertEqual(report["phases"]["baseline_capture"]["status"], "executed")

    def test_baseline_continuation_is_not_anchor_resume(self):
        report = summary.phase_evidence(['flydelta_model_exchange {"phase":"resume_full","executed":true}'])
        self.assertEqual(report["phases"]["anchor_resume"]["status"], "not_reached")

    def test_selection_evidence_uses_guidance_arm_identity(self):
        path = self.root / "raw/low-rank-rank-sweep.log"
        records = path.read_text().splitlines()
        for index, line in enumerate(records):
            if line.startswith("flydelta_low_rank_arm "):
                arm = json.loads(line.split(" ", 1)[1])
                arm["guidance_arm_id"] = "guidance:1"
                records[index] = "flydelta_low_rank_arm " + json.dumps(arm)
        decision = {"wave": 1, "stage": "frontier", "compatible_history_rows": 2,
                    "posterior_computed": True, "selected": [{"arm_id": "guidance:1", "reason": "posterior_sample"}]}
        execution = {"wave": 1, "guidance_arm_id": "guidance:1", "arm_id": "arm:1", "executed": True}
        records.extend(["flydelta_progress_adviser " + json.dumps(decision),
                        "flydelta_progress_continuation " + json.dumps(execution)])
        path.write_text("\n".join(records) + "\n")
        evidence = summary.build_report(self.root)["conditions"]["raw"]["response_pairs"][0]["selection_evidence"]
        self.assertEqual(evidence[0]["reason"], "posterior_sample")
        self.assertEqual(evidence[1]["execution_arm_id"], "arm:1")

    def test_different_manifest_command_is_unmatched(self):
        path = self.root / "run-manifest.json"
        manifest = json.loads(path.read_text())
        manifest["commands"]["normalized"]["argv"].extend(["--threads", "1"])
        path.write_text(json.dumps(manifest))
        self.assertEqual(summary.build_report(self.root)["comparability"], "unmatched")

    def test_missing_baseline_repeats_are_unmatched(self):
        path = self.root / "raw/low-rank-rank-sweep.log"
        path.write_text(path.read_text().replace('"baseline_repeats": 1', '"baseline_repeats": 3'))
        report = summary.build_report(self.root)
        self.assertEqual(report["comparability"], "unmatched")
        self.assertTrue(any("repeats missing" in issue for issue in report["issues"]))

    def test_failed_condition_is_not_matched(self):
        (self.root / "normalized/process.exit").write_text("1\n")
        self.assertEqual(summary.build_report(self.root)["comparability"], "unmatched")

    def test_diagnostic_raw_limit_is_unknown_not_false(self):
        path = self.root / "raw/low-rank-rank-sweep.log"
        text = path.read_text().replace('"dose_safety_limited": true',
            '"dose_safety_limited": false, "proposal_norm_limit_known": false, "proposal_norm_limited": null')
        path.write_text(text)
        dose = summary.build_report(self.root)["conditions"]["raw"]["response_pairs"][0]["dose"]
        self.assertFalse(dose["proposal_norm_limit_known"])
        self.assertIsNone(dose["proposal_norm_limited"])

    def test_calibration_overlap_is_rejected(self):
        path = self.root / "normalized/low-rank-rank-sweep.log"
        path.write_text(path.read_text().replace('"source_scenarios": ["scenario-2"]',
                                                  '"source_scenarios": ["scenario-1"]'))
        self.assertEqual(summary.build_report(self.root)["status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
