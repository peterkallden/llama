#!/usr/bin/env python3
"""Normalize candidate vectors into the steering vector JSON format."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


STEERING_MODULE_PATH = Path(__file__).with_name("steering_injection.py")
STEERING_SPEC = importlib.util.spec_from_file_location("steering_injection", STEERING_MODULE_PATH)
assert STEERING_SPEC and STEERING_SPEC.loader
steering_injection = importlib.util.module_from_spec(STEERING_SPEC)
sys.modules[STEERING_SPEC.name] = steering_injection
STEERING_SPEC.loader.exec_module(steering_injection)


def normalize_vector_row(row: dict, layer: int | None = None, feature_id: int | None = None) -> dict:
    values = row.get("values", row.get("vector"))
    if values is None:
        raise ValueError("vector row must contain either 'values' or 'vector'")
    if not isinstance(values, list) or not values:
        raise ValueError("vector values must be a non-empty list")
    if not all(isinstance(value, (int, float)) for value in values):
        raise ValueError("vector values must contain only numbers")

    resolved_layer = int(layer if layer is not None else row["layer"])
    resolved_feature_id = int(feature_id if feature_id is not None else row["feature_id"])
    return {
        "layer": resolved_layer,
        "feature_id": resolved_feature_id,
        "source": row.get("source", "vector-io"),
        "notes": row.get("notes", ""),
        "values": [float(value) for value in values],
    }


def write_vector(row: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    steering_injection.load_steering_vector(output)


def command_from_json(args: argparse.Namespace) -> int:
    row = json.loads(args.input.read_text(encoding="utf-8"))
    vector = normalize_vector_row(row, layer=args.layer, feature_id=args.feature_id)
    write_vector(vector, args.output)
    print(f"Wrote {vector['layer']=} {vector['feature_id']=} width={len(vector['values'])} to {args.output}")
    return 0


def command_inspect(args: argparse.Namespace) -> int:
    vector = steering_injection.load_steering_vector(args.vector_file)
    print(json.dumps({
        "site": vector.site,
        "width": len(vector.values),
        "source": vector.source,
        "notes": vector.notes,
    }, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    from_json = subparsers.add_parser("from-json", help="normalize a JSON vector into steering_vector format")
    from_json.add_argument("--input", type=Path, required=True)
    from_json.add_argument("--output", type=Path, required=True)
    from_json.add_argument("--layer", type=int)
    from_json.add_argument("--feature-id", type=int)
    from_json.set_defaults(handler=command_from_json)

    inspect = subparsers.add_parser("inspect", help="inspect a normalized steering vector")
    inspect.add_argument("--vector-file", type=Path, required=True)
    inspect.set_defaults(handler=command_inspect)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())

