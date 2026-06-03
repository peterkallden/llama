#!/usr/bin/env python3
"""Build a verification plan for candidate circuit-tracer features.

The script is intentionally runner-agnostic. It turns discovered candidates into
the exact two checks milestone 1 needs: ablate-to-fail on the clean prompt and
inject-to-restore on the corrupt prompt. A later iteration can replace the dry
run with direct calls into circuit-tracer once the research environment is
available.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("circuit_transfer.py")
SPEC = importlib.util.spec_from_file_location("circuit_transfer", MODULE_PATH)
assert SPEC and SPEC.loader
circuit_transfer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = circuit_transfer
SPEC.loader.exec_module(circuit_transfer)


@dataclass(frozen=True)
class CandidateFeature:
    case_id: str
    layer: int
    feature_id: int
    graph_path: str | None
    notes: str

    @property
    def site(self) -> str:
        return f"blocks.{self.layer}.feature.{self.feature_id}"


def load_candidates(path: Path) -> list[CandidateFeature]:
    candidates = []
    for row in circuit_transfer.load_jsonl(path):
        missing = sorted({"case_id", "layer", "feature_id"} - row.keys())
        if missing:
            raise ValueError(f"{path}: missing candidate fields: {', '.join(missing)}")
        candidates.append(
            CandidateFeature(
                case_id=row["case_id"],
                layer=int(row["layer"]),
                feature_id=int(row["feature_id"]),
                graph_path=row.get("graph_path"),
                notes=row.get("notes", ""),
            )
        )
    return candidates


def build_plan(cases_path: Path, candidates_path: Path, model: str) -> list[dict]:
    cases_by_id = {case.case_id: case for case in circuit_transfer.load_cases(cases_path)}
    plan = []
    for candidate in load_candidates(candidates_path):
        if candidate.case_id not in cases_by_id:
            raise ValueError(f"{candidates_path}: unknown case_id: {candidate.case_id}")
        case = cases_by_id[candidate.case_id]
        common = {
            "schema_version": "circuit-transfer-verification-plan/v1",
            "case_id": case.case_id,
            "behavior": case.behavior,
            "model": model,
            "target": case.target,
            "distractor": case.distractor,
            "site": candidate.site,
            "layer": candidate.layer,
            "feature_id": candidate.feature_id,
            "graph_path": candidate.graph_path,
            "candidate_notes": candidate.notes,
            "control_group": "target",
            "result_schema": circuit_transfer.RESULT_SCHEMA_VERSION,
        }
        plan.append(
            {
                **common,
                "intervention": "ablate-to-fail",
                "prompt": case.clean_prompt,
                "expected_direction": "delta_logit_diff < 0",
                "record_hint": "Run on the clean prompt; the target should become less preferred.",
            }
        )
        plan.append(
            {
                **common,
                "intervention": "inject-to-restore",
                "prompt": case.corrupt_prompt,
                "expected_direction": "delta_logit_diff > 0",
                "record_hint": "Run on the corrupt prompt; the target should become more preferred.",
            }
        )
    return plan


def command_plan(args: argparse.Namespace) -> int:
    plan = build_plan(args.cases, args.candidates, args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Wrote {len(plan)} verification steps to {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=circuit_transfer.DEFAULT_CASES)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    parser.set_defaults(handler=command_plan)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())

