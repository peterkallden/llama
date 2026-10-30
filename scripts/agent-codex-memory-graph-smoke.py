#!/usr/bin/env python3
"""Exercise Codex MCP project memory and graph persistence without a model."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

repo = Path(__file__).resolve().parent.parent
launcher = repo / "scripts/agent-codex-mcp-tools.sh"
source = json.loads((repo / "docs/examples/agent-host-config-codex-mcp-tools.json").read_text())
with tempfile.TemporaryDirectory(prefix="codex-memory-graph-") as temporary:
    root = Path(temporary)
    config = {"schema_version": 1, "stores": {
        "memory": {"backend": "cozo", "path": str(root / "memory.cozo")},
        "data": {"backend": "cozo", "path": str(root / "data.cozo")}},
        "tools": {"profile": source["tools"]["profile"],
                  "capabilities": source["tools"]["capabilities"], "profiles": source["tools"]["profiles"]}}
    path = root / "config.json"
    path.write_text(json.dumps(config))
    environment = os.environ.copy()
    environment["LLAMA_AGENT_CODEX_MCP_CONFIG"] = str(path)

    def start(project):
        process = subprocess.Popen([str(launcher), "--memory-project", project], cwd=repo, env=environment,
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        def rpc(ident, method, params):
            process.stdin.write(json.dumps({"jsonrpc": "2.0", "id": ident, "method": method, "params": params}) + "\n")
            process.stdin.flush()
            line = process.stdout.readline()
            if not line:
                raise RuntimeError("MCP server closed: " + process.stderr.read())
            response = json.loads(line)
            if response.get("id") != ident or "error" in response:
                raise RuntimeError(f"unexpected response: {response}")
            return response["result"]
        def call(ident, name, args):
            result = rpc(ident, "tools/call", {"name": name, "arguments": args})
            if result.get("isError") or not result.get("structuredContent", {}).get("ok"):
                raise RuntimeError(f"{name} failed: {result}")
            return result["structuredContent"]["result"]
        rpc(1, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                              "clientInfo": {"name": "memory-graph-smoke", "version": "1"}})
        process.stdin.write('{"jsonrpc":"2.0","method":"notifications/initialized"}\n')
        process.stdin.flush()
        return process, rpc, call

    def stop(process):
        process.stdin.close()
        process.wait(timeout=15)
        if process.returncode:
            raise RuntimeError("MCP server failed: " + process.stderr.read())

    process, rpc, call = start("smoke-project")
    try:
        names = {item["name"] for item in rpc(2, "tools/list", {})["tools"]}
        assert {"memory.remember", "graph.link.propose", "graph.retract.propose"} <= names
        memory = call(3, "memory.remember", {"kind": "decision", "content": "The smoke fixture uses the project graph for working relationships.",
                                                "importance": 0.8, "confidence": 0.9, "rationale": "project test fixture"})
        assert memory.get("decision") == "accept" and memory.get("id"), memory
        edge = call(4, "graph.link.propose", {"from": {"kind": "component", "name": "fixture-source"},
            "to": {"kind": "component", "name": "fixture-target"}, "relation": "depends_on",
            "rationale": "project test fixture"})["edge"]
    finally:
        stop(process)
    process, rpc, call = start("smoke-project")
    try:
        saved = call(5, "memory.get", {"id": memory["id"]})
        assert "project graph" in json.dumps(saved), saved
        found = call(6, "graph.search", {"from": edge["from"], "relation": "depends_on"})
        assert len(found["edges"]) == 1 and found["edges"][0]["id"] == edge["id"], found
    finally:
        stop(process)
    process, rpc, call = start("other-project")
    try:
        assert call(7, "graph.find", {"query": "fixture-source"})["nodes"] == []
        result = rpc(8, "tools/call", {"name": "memory.get", "arguments": {"id": memory["id"]}})
        assert result.get("isError") or not result.get("structuredContent", {}).get("ok"), result
    finally:
        stop(process)
    print("Codex MCP memory and graph: persistence, project isolation and proposals passed (no model)")
