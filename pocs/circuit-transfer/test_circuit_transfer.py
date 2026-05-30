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


if __name__ == "__main__":
    unittest.main()
