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


@dataclass(frozen=True)
class PromptCase:
    case_id: str
    behavior: str
    clean_prompt: str
    corrupt_prompt: str
    target: str
    distractor: str
    templates: tuple[str, ...]


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
    row = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "case_id": args.case_id,
        "intervention": args.intervention,
        "site": args.site,
        "baseline_logit_diff": args.baseline_logit_diff,
        "intervention_logit_diff": args.intervention_logit_diff,
        "delta_logit_diff": args.intervention_logit_diff - args.baseline_logit_diff,
        "notes": args.notes,
    }
    args.run_file.parent.mkdir(parents=True, exist_ok=True)
    with args.run_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(row, indent=2, sort_keys=True))
    return 0


def command_summarize(args: argparse.Namespace) -> int:
    rows = load_jsonl(args.run_file)
    if not rows:
        print(f"No records in {args.run_file}")
        return 0
    deltas = [float(row["delta_logit_diff"]) for row in rows]
    print(f"Records: {len(rows)}")
    print(f"Mean delta logit diff: {statistics.fmean(deltas):.4f}")
    print(f"Median delta logit diff: {statistics.median(deltas):.4f}")
    for row in rows:
        print(f"- {row['case_id']} {row['intervention']} {row['site']}: {float(row['delta_logit_diff']):+.4f}")
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
    record.add_argument("--intervention", choices=("ablate-to-fail", "inject-to-restore"), required=True)
    record.add_argument("--site", required=True)
    record.add_argument("--baseline-logit-diff", type=float, required=True)
    record.add_argument("--intervention-logit-diff", type=float, required=True)
    record.add_argument("--notes", default="")
    record.set_defaults(handler=command_record)

    summarize = subparsers.add_parser("summarize", help="summarize measured intervention effects")
    summarize.add_argument("--run-file", type=Path, required=True)
    summarize.set_defaults(handler=command_summarize)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
