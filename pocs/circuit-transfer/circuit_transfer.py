#!/usr/bin/env python3
"""Small utilities for a reproducible Llama circuit-discovery experiment."""

from __future__ import annotations

import argparse
import json
import shlex
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


POC_DIR = Path(__file__).resolve().parent
DEFAULT_CASES = POC_DIR / "data" / "factual_recall.jsonl"
RESULT_SCHEMA_VERSION = "circuit-transfer-result/v1"


@dataclass(frozen=True)
class PromptCase:
    case_id: str
    behavior: str
    clean_prompt: str
    corrupt_prompt: str
    target: str
    distractor: str
    templates: tuple[str, ...]


@dataclass(frozen=True)
class SuccessCriteria:
    min_restore_delta: float = 2.0
    min_ablation_drop: float = 1.0
    min_templates: int = 2
    max_control_abs_delta: float = 0.5


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def load_cases(path: Path) -> list[PromptCase]:
    cases = []
    required = {
        "case_id",
        "behavior",
        "clean_prompt",
        "corrupt_prompt",
        "target",
        "distractor",
        "templates",
    }
    for row in load_jsonl(path):
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"{path}: missing fields for a case: {', '.join(missing)}")
        templates = row["templates"]
        if not isinstance(templates, list) or len(templates) < 2:
            raise ValueError(f"{path}: {row['case_id']}: templates must contain at least two prompts")
        cases.append(
            PromptCase(
                case_id=row["case_id"],
                behavior=row["behavior"],
                clean_prompt=row["clean_prompt"],
                corrupt_prompt=row["corrupt_prompt"],
                target=row["target"],
                distractor=row["distractor"],
                templates=tuple(templates),
            )
        )
    duplicate_ids = sorted(case_id for case_id in {case.case_id for case in cases} if sum(case.case_id == case_id for case in cases) > 1)
    if duplicate_ids:
        raise ValueError(f"{path}: duplicate case ids: {', '.join(duplicate_ids)}")
    return cases


def tracer_command(prompt: str, slug: str, output_dir: Path, transcoder_set: str) -> str:
    graph_dir = output_dir / slug
    graph_path = output_dir / f"{slug}.pt"
    args = [
        "circuit-tracer",
        "attribute",
        "--prompt",
        prompt,
        "--transcoder_set",
        transcoder_set,
        "--slug",
        slug,
        "--graph_file_dir",
        str(graph_dir),
        "--graph_output_path",
        str(graph_path),
    ]
    return shlex.join(args)


def iter_tracer_commands(cases: Iterable[PromptCase], output_dir: Path, transcoder_set: str) -> Iterable[str]:
    for case in cases:
        yield tracer_command(case.clean_prompt, f"{case.case_id}-clean", output_dir, transcoder_set)
        yield tracer_command(case.corrupt_prompt, f"{case.case_id}-corrupt", output_dir, transcoder_set)


def command_validate(args: argparse.Namespace) -> int:
    cases = load_cases(args.cases)
    print(f"Validated {len(cases)} prompt cases from {args.cases}")
    for case in cases:
        print(f"- {case.case_id}: target={case.target!r}, distractor={case.distractor!r}")
    return 0


def command_render_tracer(args: argparse.Namespace) -> int:
    cases = load_cases(args.cases)
    for command in iter_tracer_commands(cases, args.output_dir, args.transcoder_set):
        print(command)
    return 0


def command_record(args: argparse.Namespace) -> int:
    site = args.site or f"blocks.{args.layer}.feature.{args.feature_id}"
    row = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "case_id": args.case_id,
        "behavior": args.behavior,
        "model": args.model,
        "prompt": args.prompt,
        "target": args.target,
        "distractor": args.distractor,
        "template_id": args.template_id,
        "intervention": args.intervention,
        "site": site,
        "layer": args.layer,
        "feature_id": args.feature_id,
        "graph_path": str(args.graph_path) if args.graph_path else None,
        "baseline_logit_diff": args.baseline_logit_diff,
        "intervention_logit_diff": args.intervention_logit_diff,
        "delta_logit_diff": args.intervention_logit_diff - args.baseline_logit_diff,
        "control_group": args.control_group,
        "notes": args.notes,
    }
    validate_result_row(row)
    args.run_file.parent.mkdir(parents=True, exist_ok=True)
    with args.run_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(row, indent=2, sort_keys=True))
    return 0


def validate_result_row(row: dict) -> None:
    required_types = {
        "schema_version": str,
        "recorded_at": str,
        "case_id": str,
        "behavior": str,
        "model": str,
        "prompt": str,
        "target": str,
        "distractor": str,
        "template_id": str,
        "intervention": str,
        "site": str,
        "layer": int,
        "feature_id": int,
        "baseline_logit_diff": (int, float),
        "intervention_logit_diff": (int, float),
        "delta_logit_diff": (int, float),
        "control_group": str,
        "notes": str,
    }
    for key, expected_type in required_types.items():
        if key not in row:
            raise ValueError(f"result row is missing required field: {key}")
        if not isinstance(row[key], expected_type):
            raise ValueError(f"result row field {key!r} has invalid type")
    if row["schema_version"] != RESULT_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {row['schema_version']}")
    if row["intervention"] not in {"ablate-to-fail", "inject-to-restore"}:
        raise ValueError(f"unsupported intervention: {row['intervention']}")
    if row["control_group"] not in {"target", "unrelated-control", "wrong-site-control"}:
        raise ValueError(f"unsupported control_group: {row['control_group']}")


def command_summarize(args: argparse.Namespace) -> int:
    rows = load_jsonl(args.run_file)
    if not rows:
        print(f"No records in {args.run_file}")
        return 0
    for row in rows:
        validate_result_row(row)
    deltas = [float(row["delta_logit_diff"]) for row in rows]
    print(f"Records: {len(rows)}")
    print(f"Mean delta logit diff: {statistics.fmean(deltas):.4f}")
    print(f"Median delta logit diff: {statistics.median(deltas):.4f}")
    for row in rows:
        print(f"- {row['case_id']} {row['intervention']} {row['site']}: {float(row['delta_logit_diff']):+.4f}")
    return 0


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def evaluate_success(rows: list[dict], criteria: SuccessCriteria = SuccessCriteria()) -> dict:
    for row in rows:
        validate_result_row(row)

    target_rows = [row for row in rows if row["control_group"] == "target"]
    control_rows = [row for row in rows if row["control_group"] != "target"]
    injection_rows = [row for row in target_rows if row["intervention"] == "inject-to-restore"]
    ablation_rows = [row for row in target_rows if row["intervention"] == "ablate-to-fail"]

    injection_deltas = [float(row["delta_logit_diff"]) for row in injection_rows]
    ablation_deltas = [float(row["delta_logit_diff"]) for row in ablation_rows]
    control_deltas = [float(row["delta_logit_diff"]) for row in control_rows]
    successful_templates = {
        row["template_id"]
        for row in target_rows
        if (
            row["intervention"] == "inject-to-restore"
            and float(row["delta_logit_diff"]) >= criteria.min_restore_delta
        )
        or (
            row["intervention"] == "ablate-to-fail"
            and float(row["delta_logit_diff"]) <= -criteria.min_ablation_drop
        )
    }

    checks = {
        "restore_effect": bool(injection_deltas) and mean(injection_deltas) >= criteria.min_restore_delta,
        "ablation_effect": bool(ablation_deltas) and mean(ablation_deltas) <= -criteria.min_ablation_drop,
        "template_coverage": len(successful_templates) >= criteria.min_templates,
        "controls_present": bool(control_deltas),
        "controls_stable": bool(control_deltas)
        and all(abs(delta) <= criteria.max_control_abs_delta for delta in control_deltas),
    }

    reasons = []
    if checks["restore_effect"]:
        reasons.append(f"mean inject-to-restore delta is {mean(injection_deltas):+.3f}")
    else:
        reasons.append("inject-to-restore effect is missing or below threshold")
    if checks["ablation_effect"]:
        reasons.append(f"mean ablate-to-fail delta is {mean(ablation_deltas):+.3f}")
    else:
        reasons.append("ablate-to-fail effect is missing or not negative enough")
    if checks["template_coverage"]:
        reasons.append(f"effect repeats across {len(successful_templates)} prompt templates")
    else:
        reasons.append(f"effect covers {len(successful_templates)} prompt templates; need {criteria.min_templates}")
    if not checks["controls_present"]:
        reasons.append("no unrelated or wrong-site controls recorded yet")
    elif checks["controls_stable"]:
        reasons.append("control deltas stay within threshold")
    else:
        reasons.append("one or more control deltas exceed threshold")

    score = sum(
        points
        for name, points in (
            ("restore_effect", 30),
            ("ablation_effect", 30),
            ("template_coverage", 20),
            ("controls_stable", 20),
        )
        if checks[name]
    )

    if checks["restore_effect"] and checks["ablation_effect"] and checks["template_coverage"] and checks["controls_stable"]:
        status = "probable_success"
    elif checks["restore_effect"] and checks["ablation_effect"] and checks["template_coverage"] and not checks["controls_present"]:
        status = "promising_needs_controls"
    elif checks["restore_effect"] or checks["ablation_effect"]:
        status = "partial_signal"
    else:
        status = "failed_or_insufficient"

    return {
        "schema_version": "circuit-transfer-success-evaluation/v1",
        "status": status,
        "score": score,
        "checks": checks,
        "criteria": {
            "min_restore_delta": criteria.min_restore_delta,
            "min_ablation_drop": criteria.min_ablation_drop,
            "min_templates": criteria.min_templates,
            "max_control_abs_delta": criteria.max_control_abs_delta,
        },
        "counts": {
            "target_rows": len(target_rows),
            "control_rows": len(control_rows),
            "injection_rows": len(injection_rows),
            "ablation_rows": len(ablation_rows),
            "successful_templates": len(successful_templates),
        },
        "means": {
            "injection_delta": mean(injection_deltas),
            "ablation_delta": mean(ablation_deltas),
            "control_abs_delta": mean([abs(delta) for delta in control_deltas]),
        },
        "reasons": reasons,
    }


def command_evaluate_success(args: argparse.Namespace) -> int:
    criteria = SuccessCriteria(
        min_restore_delta=args.min_restore_delta,
        min_ablation_drop=args.min_ablation_drop,
        min_templates=args.min_templates,
        max_control_abs_delta=args.max_control_abs_delta,
    )
    evaluation = evaluate_success(load_jsonl(args.run_file), criteria)
    print(json.dumps(evaluation, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate the contrastive prompt dataset")
    validate.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    validate.set_defaults(handler=command_validate)

    render = subparsers.add_parser("render-tracer", help="render circuit-tracer commands")
    render.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    render.add_argument("--output-dir", type=Path, required=True)
    render.add_argument("--transcoder-set", default="llama")
    render.set_defaults(handler=command_render_tracer)

    record = subparsers.add_parser("record", help="append a measured intervention to a JSONL run file")
    record.add_argument("--run-file", type=Path, required=True)
    record.add_argument("--case-id", required=True)
    record.add_argument("--behavior", default="factual-recall")
    record.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    record.add_argument("--prompt", required=True)
    record.add_argument("--target", required=True)
    record.add_argument("--distractor", required=True)
    record.add_argument("--template-id", default="default")
    record.add_argument("--intervention", choices=("ablate-to-fail", "inject-to-restore"), required=True)
    record.add_argument("--site")
    record.add_argument("--layer", type=int, required=True)
    record.add_argument("--feature-id", type=int, required=True)
    record.add_argument("--graph-path", type=Path)
    record.add_argument("--baseline-logit-diff", type=float, required=True)
    record.add_argument("--intervention-logit-diff", type=float, required=True)
    record.add_argument("--control-group", choices=("target", "unrelated-control", "wrong-site-control"), default="target")
    record.add_argument("--notes", default="")
    record.set_defaults(handler=command_record)

    summarize = subparsers.add_parser("summarize", help="summarize measured intervention effects")
    summarize.add_argument("--run-file", type=Path, required=True)
    summarize.set_defaults(handler=command_summarize)

    evaluate = subparsers.add_parser("evaluate-success", help="classify whether results look like probable success")
    evaluate.add_argument("--run-file", type=Path, required=True)
    evaluate.add_argument("--min-restore-delta", type=float, default=SuccessCriteria.min_restore_delta)
    evaluate.add_argument("--min-ablation-drop", type=float, default=SuccessCriteria.min_ablation_drop)
    evaluate.add_argument("--min-templates", type=int, default=SuccessCriteria.min_templates)
    evaluate.add_argument("--max-control-abs-delta", type=float, default=SuccessCriteria.max_control_abs_delta)
    evaluate.set_defaults(handler=command_evaluate_success)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
