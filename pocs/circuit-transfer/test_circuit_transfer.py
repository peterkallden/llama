import importlib.util
import json
import shlex
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("circuit_transfer.py")
SPEC = importlib.util.spec_from_file_location("circuit_transfer", MODULE_PATH)
assert SPEC and SPEC.loader
circuit_transfer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = circuit_transfer
SPEC.loader.exec_module(circuit_transfer)

VERIFY_MODULE_PATH = Path(__file__).with_name("verify_interventions.py")
VERIFY_SPEC = importlib.util.spec_from_file_location("verify_interventions", VERIFY_MODULE_PATH)
assert VERIFY_SPEC and VERIFY_SPEC.loader
verify_interventions = importlib.util.module_from_spec(VERIFY_SPEC)
sys.modules[VERIFY_SPEC.name] = verify_interventions
VERIFY_SPEC.loader.exec_module(verify_interventions)


def result_row(
    intervention,
    delta,
    template_id="default",
    control_group="target",
    baseline=0.0,
):
    return {
        "schema_version": "circuit-transfer-result/v1",
        "recorded_at": "2026-06-03T12:00:00+00:00",
        "case_id": "capital-france",
        "behavior": "factual-recall",
        "model": "meta-llama/Llama-3.2-1B",
        "prompt": "The capital of Germany is",
        "target": " Paris",
        "distractor": " Berlin",
        "template_id": template_id,
        "intervention": intervention,
        "site": "blocks.12.feature.1234",
        "layer": 12,
        "feature_id": 1234,
        "graph_path": None,
        "baseline_logit_diff": baseline,
        "intervention_logit_diff": baseline + delta,
        "delta_logit_diff": delta,
        "control_group": control_group,
        "notes": "smoke test",
    }


class CircuitTransferTest(unittest.TestCase):
    def test_default_cases_are_valid(self):
        cases = circuit_transfer.load_cases(circuit_transfer.DEFAULT_CASES)
        self.assertEqual(3, len(cases))
        self.assertEqual("capital-france", cases[0].case_id)

    def test_tracer_command_contains_expected_arguments(self):
        command = circuit_transfer.tracer_command(
            "The capital of France is",
            "capital-france-clean",
            Path("graphs"),
            "llama",
        )
        self.assertIn("circuit-tracer attribute", command)
        self.assertIn("--transcoder_set llama", command)
        command_args = shlex.split(command)
        output_path = command_args[command_args.index("--graph_output_path") + 1]
        self.assertEqual(str(Path("graphs") / "capital-france-clean.pt"), output_path)

    def test_duplicate_case_ids_are_rejected(self):
        row = {
            "behavior": "factual-recall",
            "case_id": "duplicate",
            "clean_prompt": "clean",
            "corrupt_prompt": "corrupt",
            "target": " target",
            "distractor": " distractor",
            "templates": ["first", "second"],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate case ids"):
                circuit_transfer.load_cases(path)

    def test_result_row_validation_accepts_schema_v1(self):
        circuit_transfer.validate_result_row(result_row("inject-to-restore", 4.9))

    def test_evaluate_success_accepts_probable_success(self):
        rows = [
            result_row("inject-to-restore", 3.0, template_id="template-a"),
            result_row("ablate-to-fail", -1.5, template_id="template-a"),
            result_row("inject-to-restore", 2.5, template_id="template-b"),
            result_row("ablate-to-fail", -1.2, template_id="template-b"),
            result_row("inject-to-restore", 0.2, control_group="unrelated-control"),
        ]
        evaluation = circuit_transfer.evaluate_success(rows)
        self.assertEqual("probable_success", evaluation["status"])
        self.assertEqual(100, evaluation["score"])

    def test_evaluate_success_requires_controls_for_probable_success(self):
        rows = [
            result_row("inject-to-restore", 3.0, template_id="template-a"),
            result_row("ablate-to-fail", -1.5, template_id="template-a"),
            result_row("inject-to-restore", 2.5, template_id="template-b"),
            result_row("ablate-to-fail", -1.2, template_id="template-b"),
        ]
        evaluation = circuit_transfer.evaluate_success(rows)
        self.assertEqual("promising_needs_controls", evaluation["status"])
        self.assertEqual(80, evaluation["score"])

    def test_verification_plan_has_ablation_and_injection_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            candidates = Path(directory) / "candidates.jsonl"
            candidates.write_text(
                json.dumps({"case_id": "capital-france", "layer": 12, "feature_id": 1234}) + "\n",
                encoding="utf-8",
            )
            plan = verify_interventions.build_plan(
                circuit_transfer.DEFAULT_CASES,
                candidates,
                "meta-llama/Llama-3.2-1B",
            )
        self.assertEqual(2, len(plan))
        self.assertEqual("ablate-to-fail", plan[0]["intervention"])
        self.assertEqual("inject-to-restore", plan[1]["intervention"])
        self.assertEqual("blocks.12.feature.1234", plan[0]["site"])


if __name__ == "__main__":
    unittest.main()
