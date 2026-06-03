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

STEERING_MODULE_PATH = Path(__file__).with_name("steering_injection.py")
STEERING_SPEC = importlib.util.spec_from_file_location("steering_injection", STEERING_MODULE_PATH)
assert STEERING_SPEC and STEERING_SPEC.loader
steering_injection = importlib.util.module_from_spec(STEERING_SPEC)
sys.modules[STEERING_SPEC.name] = steering_injection
STEERING_SPEC.loader.exec_module(steering_injection)

DISTILL_MODULE_PATH = Path(__file__).with_name("distill_lora.py")
DISTILL_SPEC = importlib.util.spec_from_file_location("distill_lora", DISTILL_MODULE_PATH)
assert DISTILL_SPEC and DISTILL_SPEC.loader
distill_lora = importlib.util.module_from_spec(DISTILL_SPEC)
sys.modules[DISTILL_SPEC.name] = distill_lora
DISTILL_SPEC.loader.exec_module(distill_lora)


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
        "baseline_top_k": None,
        "intervention_logit_diff": baseline + delta,
        "intervention_top_k": None,
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

    def test_top_k_validation_accepts_token_distribution(self):
        row = result_row("inject-to-restore", 4.9)
        row["intervention_top_k"] = [
            {"token_id": 200, "token": " Paris", "logit": 3.9, "probability": 0.7}
        ]
        circuit_transfer.validate_result_row(row)

    def test_make_result_row_builds_site_and_delta(self):
        row = circuit_transfer.make_result_row(
            case_id="capital-france",
            behavior="factual-recall",
            model="meta-llama/Llama-3.2-1B",
            prompt="The capital of Germany is",
            target=" Paris",
            distractor=" Berlin",
            template_id="default",
            intervention="inject-to-restore",
            layer=12,
            feature_id=1234,
            baseline_logit_diff=-1.0,
            intervention_logit_diff=2.5,
        )
        self.assertEqual("blocks.12.feature.1234", row["site"])
        self.assertAlmostEqual(3.5, row["delta_logit_diff"])

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

    def test_load_steering_vector_accepts_example(self):
        vector = steering_injection.load_steering_vector(
            MODULE_PATH.parent / "data" / "steering_vector.example.json"
        )
        self.assertEqual("blocks.12.feature.1234", vector.site)
        self.assertEqual(4, len(vector.values))

    def test_logit_diff_from_logits(self):
        self.assertAlmostEqual(3.5, steering_injection.logit_diff_from_logits([0.0, 4.0, 0.5], 1, 2))

    def test_top_k_from_logits(self):
        try:
            torch = __import__("torch")
        except ModuleNotFoundError:
            self.skipTest("torch is not installed in the lightweight test environment")

        class FakeTokenizer:
            def decode(self, token_ids):
                return f"tok-{token_ids[0]}"

        entries = steering_injection.top_k_from_logits(torch.tensor([0.0, 2.0, 1.0]), FakeTokenizer(), 2)
        self.assertEqual([1, 2], [entry["token_id"] for entry in entries])
        self.assertEqual("tok-1", entries[0]["token"])

    def test_build_distillation_examples_filters_successful_injections(self):
        rows = [
            result_row("inject-to-restore", 3.0, template_id="template-a"),
            result_row("inject-to-restore", 0.4, template_id="template-b"),
            result_row("ablate-to-fail", -2.0, template_id="template-c"),
            result_row("inject-to-restore", 4.0, control_group="unrelated-control"),
        ]
        examples = distill_lora.build_distillation_examples(rows, min_delta=2.0)
        self.assertEqual(1, len(examples))
        self.assertEqual("template-a", examples[0].template_id)
        self.assertEqual(" Paris", examples[0].target)

    def test_distillation_examples_preserve_teacher_top_k(self):
        row = result_row("inject-to-restore", 3.0, template_id="template-a")
        row["intervention_top_k"] = [
            {"token_id": 200, "token": " Paris", "logit": 3.9, "probability": 0.7}
        ]
        examples = distill_lora.build_distillation_examples([row], min_delta=2.0)
        self.assertEqual(row["intervention_top_k"], examples[0].teacher_top_k)

    def test_distillation_dataset_round_trip(self):
        examples = [
            distill_lora.DistillationExample(
                prompt="The capital of Germany is",
                target=" Paris",
                case_id="capital-france",
                template_id="template-a",
                teacher_delta_logit_diff=4.9,
                source_site="blocks.12.feature.1234",
                teacher_top_k=[{"token_id": 200, "token": " Paris", "logit": 3.9, "probability": 0.7}],
            )
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            distill_lora.write_distillation_dataset(examples, path)
            loaded = distill_lora.load_distillation_dataset(path)
        self.assertEqual(examples, loaded)


if __name__ == "__main__":
    unittest.main()
