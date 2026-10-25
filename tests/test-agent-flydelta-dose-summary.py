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
        for mode in summary.MODES:
            folder = self.root / mode
            folder.mkdir()
            (folder / "process.exit").write_text("0\n")
            arm = {"fixture_id": "resource://" + mode, "evaluation_scenario_id": "scenario-1",
                   "requested_rank": 1, "layer": 2, "basis_identity": "basis-1",
                   "mutation": "full_generation_top_arm", "outcome": "UNKNOWN",
                   "executed": True, "host_evaluated": True, "verifier_known": False,
                   "dose_experiment": mode}
            records = ["flydelta_low_rank_arm " + json.dumps(arm),
                       "flydelta_low_rank_rank_sweep group=test elapsed_ms=12"]
            if mode != "raw":
                records.append("flydelta_dose_calibration " + json.dumps({
                    "excluded_evaluation_scenario": "scenario-1", "source_scenarios": ["scenario-2"]}))
            (folder / "low-rank-rank-sweep.log").write_text("\n".join(records) + "\n")

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

    def test_calibration_overlap_is_rejected(self):
        path = self.root / "normalized/low-rank-rank-sweep.log"
        path.write_text(path.read_text().replace('"source_scenarios": ["scenario-2"]',
                                                  '"source_scenarios": ["scenario-1"]'))
        self.assertEqual(summary.build_report(self.root)["status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
