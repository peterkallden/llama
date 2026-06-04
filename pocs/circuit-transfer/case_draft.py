#!/usr/bin/env python3
"""Draft and normalize dynamic circuit-discovery prompt cases."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path


POC_DIR = Path(__file__).resolve().parent

MODULE_PATH = POC_DIR / "circuit_transfer.py"
SPEC = importlib.util.spec_from_file_location("circuit_transfer", MODULE_PATH)
assert SPEC and SPEC.loader
circuit_transfer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = circuit_transfer
SPEC.loader.exec_module(circuit_transfer)


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "case"


def ensure_leading_space(value: str) -> str:
    return value if value.startswith(" ") else f" {value}"


def build_teacher_prompt(context: str, behavior: str, count: int, language: str) -> str:
    return f"""You are drafting mechanistic-interpretability test cases for a Llama-family model.

Context/theme:
{context}

Generate {count} contrastive next-token factual-recall cases in {language}.

Return JSONL only. Each line must be one JSON object with exactly these fields:
- case_id: short stable slug
- behavior: "{behavior}"
- clean_prompt: prompt where the target answer should be the next token
- corrupt_prompt: closely matched prompt where the distractor should be the next token
- target: the desired next-token answer for clean_prompt, with a leading space
- distractor: the contrasting next-token answer for corrupt_prompt, with a leading space
- templates: at least 2 alternative clean prompt templates for the same target

Constraints:
- Keep target and distractor short; prefer single-token answers.
- The clean and corrupt prompts should differ minimally.
- Do not include explanations, Markdown, comments, or trailing prose.
- Avoid ambiguous facts.

Example JSONL line:
{{"case_id":"capital-france","behavior":"{behavior}","clean_prompt":"The capital of France is","corrupt_prompt":"The capital of Germany is","target":" Paris","distractor":" Berlin","templates":["The capital of France is","France's capital is"]}}
"""


def parse_json_or_jsonl(text: str) -> list[dict]:
    text = text.strip()
    if not text:
        raise ValueError("empty model output")
    if text.startswith("["):
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise ValueError("JSON array output must be a list")
        return parsed
    rows = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_number}: invalid JSONL: {exc}") from exc
    return rows


def normalize_case_row(row: dict, fallback_behavior: str) -> dict:
    target = row.get("target", row.get("answer"))
    distractor = row.get("distractor")
    if distractor is None and isinstance(row.get("distractors"), list) and row["distractors"]:
        distractor = row["distractors"][0]

    clean_prompt = row.get("clean_prompt", row.get("prompt", row.get("question_prompt")))
    corrupt_prompt = row.get("corrupt_prompt")
    if corrupt_prompt is None:
        raise ValueError(f"{row.get('case_id', '<unknown>')}: corrupt_prompt is required")
    if clean_prompt is None:
        raise ValueError(f"{row.get('case_id', '<unknown>')}: clean_prompt is required")
    if target is None:
        raise ValueError(f"{row.get('case_id', '<unknown>')}: target/answer is required")
    if distractor is None:
        raise ValueError(f"{row.get('case_id', '<unknown>')}: distractor is required")

    templates = row.get("templates")
    if templates is None:
        templates = [clean_prompt, row.get("alternate_prompt", clean_prompt)]
    if not isinstance(templates, list):
        raise ValueError(f"{row.get('case_id', '<unknown>')}: templates must be a list")
    if len(templates) < 2:
        templates = [clean_prompt, clean_prompt]

    case_id = row.get("case_id") or slugify(clean_prompt)
    return {
        "case_id": slugify(case_id),
        "behavior": row.get("behavior", fallback_behavior),
        "clean_prompt": clean_prompt,
        "corrupt_prompt": corrupt_prompt,
        "target": ensure_leading_space(str(target)),
        "distractor": ensure_leading_space(str(distractor)),
        "templates": templates,
    }


def write_cases(cases: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, sort_keys=True) + "\n")
    circuit_transfer.load_cases(output)


def command_prompt(args: argparse.Namespace) -> int:
    prompt = build_teacher_prompt(args.context, args.behavior, args.count, args.language)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(prompt, encoding="utf-8")
        print(f"Wrote teacher prompt to {args.output}")
    else:
        print(prompt)
    return 0


def command_from_output(args: argparse.Namespace) -> int:
    rows = parse_json_or_jsonl(args.input.read_text(encoding="utf-8-sig"))
    cases = [normalize_case_row(row, args.behavior) for row in rows]
    write_cases(cases, args.output)
    print(f"Wrote {len(cases)} cases to {args.output}")
    return 0


def command_check_tokenization(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    cases = circuit_transfer.load_cases(args.cases)
    failures = []
    for case in cases:
        for field_name, value in (("target", case.target), ("distractor", case.distractor)):
            token_ids = tokenizer.encode(value, add_special_tokens=False)
            if len(token_ids) != 1:
                failures.append({
                    "case_id": case.case_id,
                    "field": field_name,
                    "value": value,
                    "token_count": len(token_ids),
                    "token_ids": token_ids,
                })
    if failures:
        print(json.dumps({"status": "failed", "failures": failures}, indent=2, sort_keys=True))
        return 1
    print(json.dumps({"status": "ok", "cases": len(cases)}, indent=2, sort_keys=True))
    return 0


def command_generate_with_model(args: argparse.Namespace) -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompt = build_teacher_prompt(args.context, args.behavior, args.count, args.language)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map=args.device_map, torch_dtype=args.torch_dtype)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generated = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0,
        temperature=args.temperature if args.temperature > 0 else None,
    )
    text = tokenizer.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    if args.raw_output:
        args.raw_output.parent.mkdir(parents=True, exist_ok=True)
        args.raw_output.write_text(text, encoding="utf-8")
    rows = parse_json_or_jsonl(text)
    cases = [normalize_case_row(row, args.behavior) for row in rows]
    write_cases(cases, args.output)
    print(f"Wrote {len(cases)} generated cases to {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prompt = subparsers.add_parser("prompt", help="write a teacher prompt for dynamic case generation")
    prompt.add_argument("--context", required=True)
    prompt.add_argument("--behavior", default="factual-recall")
    prompt.add_argument("--count", type=int, default=8)
    prompt.add_argument("--language", default="English")
    prompt.add_argument("--output", type=Path)
    prompt.set_defaults(handler=command_prompt)

    from_output = subparsers.add_parser("from-output", help="normalize teacher JSON/JSONL output into prompt cases")
    from_output.add_argument("--input", type=Path, required=True)
    from_output.add_argument("--output", type=Path, required=True)
    from_output.add_argument("--behavior", default="factual-recall")
    from_output.set_defaults(handler=command_from_output)

    check = subparsers.add_parser("check-tokenization", help="verify target/distractor are single tokenizer tokens")
    check.add_argument("--cases", type=Path, required=True)
    check.add_argument("--model", required=True)
    check.set_defaults(handler=command_check_tokenization)

    generate = subparsers.add_parser("generate-with-model", help="ask a local/HF teacher model and normalize its JSONL output")
    generate.add_argument("--model", required=True)
    generate.add_argument("--context", required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--raw-output", type=Path)
    generate.add_argument("--behavior", default="factual-recall")
    generate.add_argument("--count", type=int, default=8)
    generate.add_argument("--language", default="English")
    generate.add_argument("--max-new-tokens", type=int, default=1024)
    generate.add_argument("--temperature", type=float, default=0.2)
    generate.add_argument("--device-map", default="auto")
    generate.add_argument("--torch-dtype", default="auto")
    generate.set_defaults(handler=command_generate_with_model)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())

