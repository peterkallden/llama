#!/usr/bin/env python3
"""Verify structured DSL arguments through the resident daemon and real CSV tools.

No build is performed. Select the GPU through the normal backend environment;
this driver requests full layer offload and never enables FlyDelta workers.
"""
import argparse
import json
import re
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--daemon", type=Path, required=True)
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--work-dir", type=Path, required=True)
parser.add_argument("--timeout", type=int, default=300)
parser.add_argument("--fixture-only", action="store_true", help="Verify the real CSV adapters without model generation")
args = parser.parse_args()
root = args.work_dir.resolve()
if root.exists() and any(root.iterdir()):
    parser.error("work directory must be empty; preserve earlier evidence")
root.mkdir(parents=True, exist_ok=True)
(root / "records.csv").write_text("id,amount\nA,3\nB,5\nC,8\n", encoding="utf-8")
config = {
    "model": {"backend": "server-context", "path": str(args.model.resolve())},
    "runtime": {"default_mode": "agent", "agent_trace": True, "n_predict": 256},
    "stores": {kind: {"backend": "cozo", "path": str(root / (kind + ".db"))}
               for kind in ("memory", "plan", "data")},
    "resources": {"blob_backend": "fs", "blob_root": str(root / "blobs"),
                  "metadata_backend": "cozo", "metadata_db": str(root / "resources.db")},
    "tools": {"profile": "csv-contract", "repository_root": str(root),
              "capabilities": {"csv-contract": ["dataset.schema", "dataset.validate"]},
              "profiles": {"csv-contract": {"include_capabilities": ["csv-contract"]}}},
}
(root / "host.json").write_text(json.dumps(config), encoding="utf-8")
requests = [{"mode": "agent", "namespace_id": "dsl-smoke", "session_id": "csv-session",
             "turn_id": "structured-plan", "memory_scope": "session", "plan_scope": "session",
             "prompt": "First call dataset.schema on records.csv. Then call dataset.validate on records.csv "
                       "with two rules: id must be unique, and amount must have numeric_range minimum 0. "
                       "Execute both operations and report the actual validation result."},
            {"command": "shutdown"}]
if args.fixture_only:
    scope = {"command": "execute_tool", "namespace_id": "dsl-smoke", "session_id": "csv-session",
             "read_only_only": True}
    requests = [dict(scope, tool_name="dataset.schema", arguments={"path": "records.csv"}),
                dict(scope, tool_name="dataset.validate", arguments={"path": "records.csv", "rules": [
                    {"type": "unique", "column": "id"},
                    {"type": "numeric_range", "column": "amount", "min": 0}]}),
                {"command": "shutdown"}]
wire = "".join(json.dumps(item) + "\n" for item in requests)
(root / "requests.jsonl").write_text(wire, encoding="utf-8")
try:
    with (root / "stdout.jsonl").open("w") as out, (root / "stderr.log").open("w") as err:
        completed = subprocess.run(
            [str(args.daemon.resolve()), "--config", str(root / "host.json"), "-ngl", "99", "--generation-trace"],
            input=wire, text=True, stdout=out, stderr=err, timeout=args.timeout)
except subprocess.TimeoutExpired:
    raise SystemExit(f"FAIL: daemon timed out; evidence: {root}")
messages = []
for line in (root / "stdout.jsonl").read_text().splitlines():
    try:
        messages.append(json.loads(line))
    except json.JSONDecodeError:
        pass
if args.fixture_only:
    tools = [item["tool"] for item in messages if "tool" in item]
    if completed.returncode or len(tools) != 2 or not all(tool.get("ok") for tool in tools):
        raise SystemExit(f"FAIL: CSV adapter fixture failed; evidence: {root}")
    schema_result = tools[0].get("structured_content", {}).get("result", {})
    if [column.get("name") for column in schema_result.get("columns", [])] != ["id", "amount"]:
        raise SystemExit(f"FAIL: CSV adapter returned an unexpected schema; evidence: {root}")
    if not tools[1].get("structured_content", {}).get("result", {}).get("valid"):
        raise SystemExit(f"FAIL: CSV fixture failed its declared validation rules; evidence: {root}")
    print(f"PASS: actual CSV schema and validation adapters; no model generation; evidence: {root}")
    raise SystemExit(0)
turns = [item for item in messages if "trace" in item and item.get("message_type") != "event"]
if completed.returncode or len(turns) != 1 or not turns[0].get("ok"):
    raise SystemExit(f"FAIL: resident turn did not complete: {[turn.get('error') for turn in turns]}; evidence: {root}")
trace = turns[0]["trace"]
executed = [entry["tool_name"] for entry in trace
            if entry.get("kind") == "succeeded" and entry.get("tool_name")]
if executed != ["dataset.schema", "dataset.validate"]:
    raise SystemExit(f"FAIL: expected actual schema -> validation execution, got {executed}; evidence: {root}")
# A response sentence is not proof: require tool execution and a durable plan identity.
if not any(entry.get("plan_id") for entry in trace):
    raise SystemExit(f"FAIL: no published plan identity; evidence: {root}")
log = (root / "stderr.log").read_text()
outputs = "\n".join(re.findall(
    r"agent generation trace output: [^\n]* begin\n(.*?)\nagent generation trace output: end", log, re.S))
if 'rules[0].' not in outputs or 'rules[1].' not in outputs:
    raise SystemExit(f"FAIL: no two-item indexed DSL generation evidence; evidence: {root}")
# Reopen the same stores in a fresh daemon without another model turn.
restart_wire = json.dumps({"command": "list_plans", "namespace_id": "dsl-smoke",
                           "session_id": "csv-session", "turn_id": "structured-plan"}) + "\n"
restart_wire += json.dumps({"command": "shutdown"}) + "\n"
(root / "restart-requests.jsonl").write_text(restart_wire, encoding="utf-8")
with (root / "restart-stdout.jsonl").open("w") as out, (root / "restart-stderr.log").open("w") as err:
    restarted = subprocess.run([str(args.daemon.resolve()), "--config", str(root / "host.json")],
                               input=restart_wire, text=True, stdout=out, stderr=err, timeout=args.timeout)
plans = []
for line in (root / "restart-stdout.jsonl").read_text().splitlines():
    try:
        plans.extend(json.loads(line).get("plans", []))
    except json.JSONDecodeError:
        pass
plan_id = turns[0].get("plan_id")
if restarted.returncode or not any(plan.get("plan_id") == plan_id and
                                  plan.get("status") == "completed" and
                                  plan.get("observation_count", 0) >= 2 for plan in plans):
    raise SystemExit(f"FAIL: completed two-observation plan did not survive restart; evidence: {root}")
print(f"PASS: model-generated structured CSV checks executed and persisted through restart; evidence: {root}")
