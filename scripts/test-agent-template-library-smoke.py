#!/usr/bin/env python3
"""Exercise configured host libraries through the production daemon (no build)."""
import argparse
import json
import hashlib
import time
import re
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--daemon", required=True, type=Path)
parser.add_argument("--model", required=True, type=Path)
parser.add_argument("--work-dir", required=True, type=Path)
parser.add_argument("--timeout", type=int, default=240)
parser.add_argument("--declared-step", action="store_true", help="Use a declared host workflow tool transition; model still synthesizes the result")
args = parser.parse_args()
root = args.work_dir.resolve()
if root.exists() and any(root.iterdir()):
    parser.error("work directory must be empty; preserve prior evidence and choose a new directory")
root.mkdir(parents=True, exist_ok=True)
package = {
    "schema_version": 2, "name": "library-smoke", "version": "v1", "procedures": [],
    "workflows": [{"id": "clock", "source_revision": "clock@v1", "workflow_ref": "workflow://smoke/clock",
        "workflow_revision": "v1", "family": "clock", "graph_revision": "clock-g1", "allowed_tools": ["time.now"]}],
    "blueprints": [{"id": "clock", "source_revision": "clock@v1", "goal": "Read the current time with time.now.",
        "selection_description": "Read current time from the host clock.", "success_criteria": "A successful time.now result supports the answer.",
        "workflow_policy": "required", "workflow_bindings": [{"workflow_ref": "workflow://smoke/clock", "workflow_revision": "v1"}],
        "steps": [{"id": "read", "title": "Read clock", "objective": "Call time.now and answer from its result."}]}]}
if args.declared_step:
    package["workflows"][0].update({"start_state": "pending", "terminal_states": ["observed"],
        "transitions": [{"id": "read-clock", "from": "pending", "to": "observed", "tool": "time.now", "arguments": "{}"}]})
(root / "methods.json").write_text(json.dumps(package), encoding="utf-8")
config = {
    "model": {"backend": "server-context", "path": str(args.model.resolve())},
    "runtime": {"default_mode": "agent", "agent_blueprint": "smoke:clock", "agent_trace": True, "n_predict": 256},
    "bootstrap": {"libraries": [{"id": "smoke", "package": "methods.json", "level": "host"}]},
    "stores": {kind: {"backend": "cozo", "path": str(root / (kind + ".db"))} for kind in ("memory", "plan", "data")},
    "resources": {"blob_backend": "fs", "blob_root": str(root / "blobs"), "metadata_backend": "cozo", "metadata_db": str(root / "resources.db")},
    "tools": {"profile": "clock", "capabilities": {"clock": ["time.now"]}, "profiles": {"clock": {"include_capabilities": ["clock"]}}}}
(root / "host.json").write_text(json.dumps(config), encoding="utf-8")

def run(label, namespaces, resume_plan=None):
    requests = [{"mode": "agent", "prompt": ("Summarize the preserved clock observation without collecting a new observation." if resume_plan else "Read the current time using the host clock and report it."),
        "turn_id": label + "-turn",
        "namespace_id": ns, "session_id": "library-smoke-session", "memory_scope": "session", "plan_scope": "session"} for ns in namespaces]
    requests.append({"command": "shutdown"})
    wire = "".join(json.dumps(item) + "\n" for item in requests)
    (root / (label + "-requests.jsonl")).write_text(wire, encoding="utf-8")
    with (root / (label + "-stdout.jsonl")).open("w") as out, (root / (label + "-stderr.log")).open("w") as err:
        result = subprocess.run([str(args.daemon.resolve()), "--config", str(root / "host.json"), "-ngl", "99", "--generation-trace"],
            input=wire, text=True, stdout=out, stderr=err, timeout=args.timeout)
    if result.returncode:
        raise SystemExit(f"{label}: daemon failed; inspect {root}")
    messages = []
    for line in (root / (label + "-stdout.jsonl")).read_text().splitlines():
        try:
            messages.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    turns = [item for item in messages if "trace" in item and item.get("message_type") != "event"]
    if len(turns) != len(namespaces):
        raise SystemExit(f"{label}: expected {len(namespaces)} traced turns, got {len(turns)}; inspect {root}")
    for turn in turns:
        if not turn.get("ok"):
            raise SystemExit(f"{label}: turn failed: {turn.get('error')}; inspect {root}")
        trace = turn["trace"]
        if not resume_plan and not any(entry.get("tool_name") == "time.now" and entry.get("kind") == "succeeded" for entry in trace):
            raise SystemExit(f"{label}: no successful clock tool evidence; inspect {root}")
        if resume_plan and any(entry.get("tool_name") == "time.now" and entry.get("kind") == "succeeded" for entry in trace):
            raise SystemExit(f"{label}: resume repeated the already completed clock operation; inspect {root}")
        if resume_plan and preserved_time not in turn.get("response", ""):
            raise SystemExit(f"{label}: response did not consume preserved clock evidence; inspect {root}")
        if resume_plan and not any(entry.get("plan_id") == resume_plan and "resum" in entry.get("detail", "") for entry in trace):
            raise SystemExit(f"{label}: no preserved-plan resume evidence; inspect {root}")
        if not resume_plan and "smoke:clock" not in json.dumps(trace):
            raise SystemExit(f"{label}: no configured library route evidence; inspect {root}")
    return turns

initial = run("initial", ["library-a", "library-b"])
plan_ids = [next(entry["plan_id"] for entry in turn["trace"] if entry.get("plan_id")) for turn in initial]
if plan_ids[0] == plan_ids[1]:
    raise SystemExit("caller-owned task identities collided across namespaces")
# Interrupt only this smoke-owned daemon after durable tool evidence reaches the
# draft prompt. This keeps an active task for the existing auto-resume contract.
interrupt_request={"mode":"agent", "prompt":"Read the current time using the host clock and report it.",
    "namespace_id":"library-resume", "session_id":"library-smoke-session", "turn_id":"interrupted-turn",
    "memory_scope":"session", "plan_scope":"session"}
(root / "interrupted-requests.jsonl").write_text(json.dumps(interrupt_request)+"\n", encoding="utf-8")
with (root / "interrupted-stdout.jsonl").open("w") as out, (root / "interrupted-stderr.log").open("w") as err:
    proc=subprocess.Popen([str(args.daemon.resolve()), "--config", str(root / "host.json"), "-ngl", "99", "--generation-trace"],
        stdin=subprocess.PIPE, text=True, stdout=out, stderr=err)
    try:
        proc.stdin.write(json.dumps(interrupt_request)+"\n")
        proc.stdin.flush()
        deadline=time.monotonic()+args.timeout
        while time.monotonic()<deadline:
            evidence=(root / "interrupted-stderr.log").read_text()
            if "purpose=draft" in evidence and "agent generation trace prompt: end" in evidence:
                if "agent generation trace output:" in evidence:
                    raise SystemExit("interruption missed the active-draft window")
                match=re.search(r'time: "([^"]+)"', evidence)
                if not match:
                    raise SystemExit("interruption reached no verified clock observation")
                preserved_time=match.group(1)
                proc.kill()
                proc.wait(timeout=15)
                break
            if proc.poll() is not None:
                raise SystemExit("interrupted daemon exited before preserving tool evidence")
            time.sleep(0.05)
        else:
            raise SystemExit("interrupted daemon did not reach the evidence boundary")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=15)
        proc.stdin.close()
identity=json.dumps(["library-resume", "", "library-smoke-session", "interrupted-turn"], separators=(",",":"))
interrupted_id="agent-blueprint:"+hashlib.sha256(identity.encode()).hexdigest()
(root / "interruption.json").write_text(json.dumps({"plan_id":interrupted_id,"preserved_time":preserved_time}), encoding="utf-8")

# Keep the workflow reference/revision while changing its package origin and graph.
# Resume must use the preserved v1 definition rather than the newly current graph.
package["version"] = "v2"
package["blueprints"][0]["source_revision"] = "clock@v2"
package["workflows"][0]["source_revision"] = "clock@v2"
package["workflows"][0]["graph_revision"] = "clock-g2"
(root / "methods.json").write_text(json.dumps(package), encoding="utf-8")
run("restart", ["library-resume"], resume_plan=interrupted_id)
print(f"PASS: cross-namespace host library and interrupted v1 task resume after v2 import; evidence: {root}")
