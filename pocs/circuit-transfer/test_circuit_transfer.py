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

VECTOR_IO_MODULE_PATH = Path(__file__).with_name("vector_io.py")
VECTOR_IO_SPEC = importlib.util.spec_from_file_location("vector_io", VECTOR_IO_MODULE_PATH)
assert VECTOR_IO_SPEC and VECTOR_IO_SPEC.loader
vector_io = importlib.util.module_from_spec(VECTOR_IO_SPEC)
sys.modules[VECTOR_IO_SPEC.name] = vector_io
VECTOR_IO_SPEC.loader.exec_module(vector_io)

EVALUATE_LORA_MODULE_PATH = Path(__file__).with_name("evaluate_lora.py")
EVALUATE_LORA_SPEC = importlib.util.spec_from_file_location("evaluate_lora", EVALUATE_LORA_MODULE_PATH)
assert EVALUATE_LORA_SPEC and EVALUATE_LORA_SPEC.loader
evaluate_lora = importlib.util.module_from_spec(EVALUATE_LORA_SPEC)
sys.modules[EVALUATE_LORA_SPEC.name] = evaluate_lora
EVALUATE_LORA_SPEC.loader.exec_module(evaluate_lora)

PIPELINE_MODULE_PATH = Path(__file__).with_name("run_pipeline.py")
PIPELINE_SPEC = importlib.util.spec_from_file_location("run_pipeline", PIPELINE_MODULE_PATH)
assert PIPELINE_SPEC and PIPELINE_SPEC.loader
run_pipeline = importlib.util.module_from_spec(PIPELINE_SPEC)
sys.modules[PIPELINE_SPEC.name] = run_pipeline
PIPELINE_SPEC.loader.exec_module(run_pipeline)

AFFINE_MODULE_PATH = Path(__file__).with_name("affine_stitch.py")
AFFINE_SPEC = importlib.util.spec_from_file_location("affine_stitch", AFFINE_MODULE_PATH)
assert AFFINE_SPEC and AFFINE_SPEC.loader
affine_stitch = importlib.util.module_from_spec(AFFINE_SPEC)
sys.modules[AFFINE_SPEC.name] = affine_stitch
AFFINE_SPEC.loader.exec_module(affine_stitch)

CASE_DRAFT_MODULE_PATH = Path(__file__).with_name("case_draft.py")
CASE_DRAFT_SPEC = importlib.util.spec_from_file_location("case_draft", CASE_DRAFT_MODULE_PATH)
assert CASE_DRAFT_SPEC and CASE_DRAFT_SPEC.loader
case_draft = importlib.util.module_from_spec(CASE_DRAFT_SPEC)
sys.modules[CASE_DRAFT_SPEC.name] = case_draft
CASE_DRAFT_SPEC.loader.exec_module(case_draft)


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
        circuit_transfer.validate_result_row(result_row("lora-restore", 3.1))

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
            result_row("lora-restore", 3.0, template_id="template-a"),
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

    def test_evaluate_success_can_skip_ablation_requirement_for_lora(self):
        rows = [
            result_row("lora-restore", 3.0, template_id="template-a"),
            result_row("lora-restore", 2.5, template_id="template-b"),
            result_row("lora-restore", 0.2, control_group="unrelated-control"),
        ]
        evaluation = circuit_transfer.evaluate_success(
            rows,
            circuit_transfer.SuccessCriteria(require_ablation=False),
        )
        self.assertEqual("probable_success", evaluation["status"])

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

    def test_vector_io_normalizes_values(self):
        row = {"layer": 1, "feature_id": 2, "vector": [1, 2.5], "source": "test"}
        vector = vector_io.normalize_vector_row(row)
        self.assertEqual({"layer": 1, "feature_id": 2, "source": "test", "notes": "", "values": [1.0, 2.5]}, vector)

    def test_evaluate_lora_builds_result_row(self):
        step = {
            "case_id": "capital-france",
            "behavior": "factual-recall",
            "prompt": "The capital of Germany is",
            "target": " Paris",
            "distractor": " Berlin",
            "template_id": "template-a",
            "layer": 12,
            "feature_id": 1234,
            "control_group": "target",
        }
        row = evaluate_lora.build_lora_result_row(step, "base", Path("adapter"), -1.0, 2.0)
        self.assertEqual("lora-restore", row["intervention"])
        self.assertAlmostEqual(3.0, row["delta_logit_diff"])

    def test_pipeline_plan_contains_heavy_and_light_steps(self):
        parser = run_pipeline.build_parser()
        with tempfile.TemporaryDirectory() as directory:
            args = parser.parse_args([
                "plan",
                "--work-dir",
                str(Path(directory) / "work"),
                "--output",
                str(Path(directory) / "plan.json"),
            ])
            steps = run_pipeline.build_pipeline(args)
        kinds = {step["kind"] for step in steps}
        self.assertIn("light", kinds)
        self.assertIn("heavy", kinds)
        self.assertIn("evaluate-lora", [step["name"] for step in steps])

    def test_affine_load_paired_prompts(self):
        prompts = affine_stitch.load_paired_prompts(
            MODULE_PATH.parent / "data" / "paired_prompts.example.jsonl"
        )
        self.assertEqual(4, len(prompts))
        self.assertEqual("capital-france-clean", prompts[0].prompt_id)

    def test_affine_fit_and_map_vector(self):
        numpy = __import__("numpy")
        donor = numpy.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        recipient = numpy.array([[2.0, 1.0], [0.0, 3.0], [2.0, 4.0]])
        weight, bias, mse = affine_stitch.fit_affine_numpy(donor, recipient, ridge_lambda=0.0)
        self.assertLess(mse, 1e-20)
        mapped = affine_stitch.map_vector_values([1.0, 0.0], weight.tolist())
        self.assertAlmostEqual(2.0, mapped[0])
        self.assertAlmostEqual(1.0, mapped[1])

    def test_affine_evaluate_plan_shape(self):
        parser = affine_stitch.build_parser()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            args = parser.parse_args([
                "evaluate-plan",
                "--stitch",
                str(base / "stitch.json"),
                "--vector-file",
                str(base / "donor-vector.json"),
                "--mapped-vector",
                str(base / "recipient-vector.json"),
                "--verification-plan",
                str(base / "verification.json"),
                "--run-file",
                str(base / "recipient-results.jsonl"),
                "--recipient-model",
                "recipient",
                "--output",
                str(base / "plan.json"),
            ])
            args.handler(args)
            plan = json.loads((base / "plan.json").read_text(encoding="utf-8"))
        self.assertEqual(["map-vector", "recipient-steering", "recipient-success"], [step["name"] for step in plan])

    def test_case_draft_prompt_mentions_jsonl_contract(self):
        prompt = case_draft.build_teacher_prompt("astronomy facts", "factual-recall", 3, "English")
        self.assertIn("Return JSONL only", prompt)
        self.assertIn("astronomy facts", prompt)
        self.assertIn('"behavior":"factual-recall"', prompt)

    def test_case_draft_normalizes_teacher_output(self):
        rows = case_draft.parse_json_or_jsonl(
            '{"case_id":"planet-mars","clean_prompt":"The red planet is","corrupt_prompt":"The largest planet is","answer":"Mars","distractors":["Jupiter"],"templates":["The red planet is","The planet called the red planet is"]}'
        )
        case = case_draft.normalize_case_row(rows[0], "factual-recall")
        self.assertEqual("planet-mars", case["case_id"])
        self.assertEqual(" Mars", case["target"])
        self.assertEqual(" Jupiter", case["distractor"])

    def test_case_draft_from_output_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "cases.jsonl"
            cases = [
                case_draft.normalize_case_row(
                    {
                        "case_id": "element-gold",
                        "clean_prompt": "The chemical symbol Au refers to",
                        "corrupt_prompt": "The chemical symbol Fe refers to",
                        "target": " gold",
                        "distractor": " iron",
                        "templates": [
                            "The chemical symbol Au refers to",
                            "Au is the chemical symbol for",
                        ],
                    },
                    "factual-recall",
                )
            ]
            case_draft.write_cases(cases, output)
            loaded = circuit_transfer.load_cases(output)
        self.assertEqual("element-gold", loaded[0].case_id)


if __name__ == "__main__":
    unittest.main()
