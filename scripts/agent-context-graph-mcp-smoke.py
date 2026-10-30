#!/usr/bin/env python3
"""Exercise project graph tools through a fresh MCP server and Cozo store."""
import json
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading

server = Path(sys.argv[1]).resolve()
repository = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else None
with tempfile.TemporaryDirectory(prefix="agent-context-graph-") as temporary:
    root = Path(temporary)
    config = {"schema_version": 1, "model": {"backend": "server-context"},
              "stores": {"data": {"backend": "cozo", "path": str(root / "data.cozo")}},
              "tools": {"profile": "graph-project", "repository_root": str(repository or root)}}
    if repository:
        config["diagnostics"] = {"semantic_backend": "clangd", "clangd_executable": "clangd",
                                 "compile_commands": str(repository / "build-agent-vulkan-cozo")}
    (root / "config.json").write_text(json.dumps(config))

    def start(project, profile="graph-project"):
        p = subprocess.Popen([str(server), "--config", str(root / "config.json"),
                              "--repository-root", str(repository or root), "--memory-project", project,
                              "--tool-profile", profile],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        lines = queue.Queue()
        threading.Thread(target=lambda: [lines.put(line) for line in p.stdout], daemon=True).start()
        def send(value):
            p.stdin.write(json.dumps(value) + "\n")
            p.stdin.flush()
        def rpc(ident, method, params):
            send({"jsonrpc": "2.0", "id": ident, "method": method, "params": params})
            response = json.loads(lines.get(timeout=180 if repository else 30))
            if response.get("id") != ident or "error" in response:
                raise RuntimeError(f"unexpected response: {response}")
            return response["result"]
        def call(ident, name, args):
            result = rpc(ident, "tools/call", {"name": name, "arguments": args})
            if result.get("isError") or not result.get("structuredContent", {}).get("ok"):
                raise RuntimeError(f"{name} failed: {result}")
            return result["structuredContent"]["result"]
        rpc(1, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                               "clientInfo": {"name": "graph-test", "version": "1"}})
        send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return p, rpc, call

    def stop(p):
        p.terminate()
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill(); p.wait()

    p, rpc, call = start("project-a")
    try:
        names = {tool["name"] for tool in rpc(2, "tools/list", {})["tools"]}
        assert {"graph.find", "graph.get", "graph.expand", "graph.search", "graph.source", "graph.link.propose", "graph.retract.propose"} <= names, names
        denied = rpc(20, "tools/call", {"name": "graph.source", "arguments": {
            "source": "code", "arguments": {"symbol": "f", "path_hint": "a.cpp", "line": 1}}})
        assert denied.get("isError") or denied.get("structuredContent", {}).get("ok") is False, denied
        created = call(3, "graph.link.propose", {
            "from": {"kind": "component", "name": "planning"},
            "to": {"kind": "file", "name": "selection", "path": "src/selection.cpp"},
            "relation": "depends_on", "rationale": "check the runtime boundary"})
        edge = created["edge"]
        assert edge["status"] == "proposed" and edge["source"] == "model"
        expanded = call(4, "graph.expand", {"id": edge["from"]})
        assert len(expanded["edges"]) == 1 and expanded["visualization"]["kind"] == "relationship"
        actor = call(21, "graph.link.propose", {"from": {"kind": "actor", "name": "Alice", "source_ref": "case:a/entity:alice"},
            "to": {"id": edge["from"]}, "relation": "associated_with", "rationale": "investigate the association"})
        actor_id = actor["edge"]["from"]
        authored = call(24, "graph.link.propose", {"from": {"id": actor_id},
            "to": {"kind": "resource", "name": "Paper", "source_ref": "paper:one"},
            "relation": "authored", "rationale": "source credits Alice"})["edge"]
        assert len(call(25, "graph.search", {"from": actor_id, "relation": "authored"})["edges"]) == 1
        assert len(call(26, "graph.search", {"to": authored["to"], "relation": "authored"})["edges"]) == 1
        assert len(call(27, "graph.expand", {"id": authored["to"]})["edges"]) == 1
    finally:
        stop(p)
    p, rpc, call = start("project-a")
    try:
        assert len(call(5, "graph.find", {"query": "planning"})["nodes"]) == 1
        assert call(22, "graph.get", {"id": actor_id})["node"]["source_ref"] == "case:a/entity:alice"
        assert len(call(28, "graph.search", {"to": authored["to"], "relation": "authored"})["edges"]) == 1
        retracted = call(6, "graph.retract.propose", {"id": edge["id"], "expected_version": 1, "rationale": "recheck"})
        assert retracted["edge"]["status"] == "retraction_proposed"
    finally:
        stop(p)
    p, rpc, call = start("project-b")
    try:
        assert call(7, "graph.find", {"query": "planning"})["nodes"] == []
    finally:
        stop(p)
    p, rpc, call = start("project-a", "developer-read")
    try:
        names = {tool["name"] for tool in rpc(8, "tools/list", {})["tools"]}
        assert {"graph.find", "graph.get", "graph.expand"} <= names, names
        assert "graph.link.propose" not in names and "graph.retract.propose" not in names, names
        assert len(call(9, "graph.find", {"query": "planning"})["nodes"]) == 1
    finally:
        stop(p)
    if repository:
        source = repository / "common/agent/tooling/adapters/families/graph-adapters.cpp"
        symbol = "common_context_graph_prompt_projection"
        line = next(i for i, text in enumerate(source.read_text().splitlines(), 1) if f"std::string {symbol}(" in text)
        p, rpc, call = start("project-a", "code-analysis")
        try:
            observed = call(23, "graph.source", {"source": "code", "arguments": {
                "symbol": symbol, "definition_path": str(source.relative_to(repository)), "definition_line": line}})
            assert observed["source_tool"] == "code.calls" and observed["data"]["semantic"] is True
            assert observed["visualization"]["kind"] == "relationship" and observed["persisted"] is False
        finally:
            stop(p)
        print("project graph MCP: live clangd source delegation passed")
    print("project graph MCP: proposal, navigation, restart, retraction, isolation and read-only profile passed")
