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
        assert {"graph.find", "graph.get", "graph.expand", "graph.search", "graph.source", "graph.link.propose", "graph.describe.propose", "graph.retract.propose"} <= names, names
        denied = rpc(20, "tools/call", {"name": "graph.source", "arguments": {
            "source": "code", "arguments": {"symbol": "f", "path_hint": "a.cpp", "line": 1}}})
        assert denied.get("isError") or denied.get("structuredContent", {}).get("ok") is False, denied
        created = call(3, "graph.link.propose", {
            "from": {"kind": "component", "name": "planning"},
            "to": {"kind": "file", "name": "selection", "path": "src/selection.cpp"},
            "relation": "depends_on", "rationale": "check the runtime boundary"})
        edge = created["edge"]
        assert edge["status"] == "proposed" and edge["source"] == "model"
        described = call(60, "graph.describe.propose", {"id": edge["from"], "expected_revision": created["revision"],
                         "description": "Coordinates bounded planner orchestration", "source_revision": "fixture-v1",
                         "evidence_refs": ["source:planning"]})
        assert described["node"]["description"]["status"] == "proposed"
        repeated = call(63, "graph.describe.propose", {"id": edge["from"], "expected_revision": described["revision"],
                        "description": "Coordinates bounded planner orchestration", "source_revision": "fixture-v1",
                        "evidence_refs": ["source:planning"]})
        assert repeated["reused"] and repeated["revision"] == described["revision"]
        assert call(61, "graph.find", {"query": "bounded planner orchestration"})["nodes"][0]["id"] == edge["from"]
        extra = call(33, "graph.link.propose", {"from": {"id": edge["from"]}, "to": {"id": edge["to"]},
                     "relation": "depends_on", "rationale": "another source", "evidence_refs": ["source:two"]})["edge"]
        assert extra["id"] == edge["id"] and extra["version"] == 2 and len(extra["assertions"]) == 2
        duplicate = call(34, "graph.link.propose", {"from": {"id": edge["from"]}, "to": {"id": edge["to"]},
                         "relation": "depends_on", "rationale": "another source", "evidence_refs": ["source:two"]})["edge"]
        assert duplicate["version"] == 2 and len(duplicate["assertions"]) == 2
        relations = call(35, "graph.search", {"from": edge["from"], "relation": "depends_on"})["edges"]
        assert len(relations) == 1 and relations[0]["id"] == edge["id"] and len(relations[0]["assertions"]) == 2
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
        symbol = {"kind": "symbol", "name": "ns::f(int)", "path": "src/f.cpp", "line": 10,
                  "source_revision": "r1", "symbol_key": "clang:repo-a:usr:f-int"}
        symbol_edge = call(29, "graph.link.propose", {"from": symbol, "to": {"id": edge["from"]},
                           "relation": "references", "rationale": "inspect symbol"})["edge"]
        symbol.update({"line": 20, "source_revision": "r2"})
        symbol_edge_again = call(30, "graph.link.propose", {"from": symbol, "to": {"id": edge["from"]},
                                 "relation": "references", "rationale": "inspect symbol"})["edge"]
        assert symbol_edge_again["from"] == symbol_edge["from"]
        for i in range(8):
            symbol.update({"line": 21 + i, "source_revision": f"r{i + 3}"})
            call(40 + i, "graph.link.propose", {"from": symbol, "to": {"id": edge["from"]},
                 "relation": "references", "rationale": "inspect symbol"})
        for i in range(6):
            call(70 + i, "graph.link.propose", {"from": {"kind": "component", "name": f"fixture-candidate-{i}"},
                 "to": {"id": edge["from"]}, "relation": "related_to", "rationale": "selection fixture"})
        limited = call(76, "graph.find", {"query": "fixture-candidate", "limit": 5})
        assert len(limited["nodes"]) == 5 and limited["truncated"] and limited["notice"]
    finally:
        stop(p)
    p, rpc, call = start("project-a")
    try:
        assert len(call(5, "graph.find", {"query": "planning"})["nodes"]) == 1
        saved_description = call(62, "graph.get", {"id": edge["from"], "description_offset": 0})
        assert saved_description["description_page"]["items"][0]["text"] == "Coordinates bounded planner orchestration"
        assert call(22, "graph.get", {"id": actor_id})["node"]["source_ref"] == "case:a/entity:alice"
        assert len(call(28, "graph.search", {"to": authored["to"], "relation": "authored"})["edges"]) == 1
        saved_symbol = call(31, "graph.get", {"id": symbol_edge["from"], "observation_offset": 0})
        assert saved_symbol["node"]["observations_truncated"] and saved_symbol["node"]["line"] == 28
        first_page = saved_symbol["observation_page"]
        assert first_page["total"] == 10 and first_page["items"][0]["line"] == 10 and first_page["next_offset"] == 8
        last_page = call(32, "graph.get", {"id": symbol_edge["from"], "observation_offset": 8})["observation_page"]
        assert len(last_page["items"]) == 2 and last_page["items"][-1]["line"] == 28 and last_page["next_offset"] is None
        retracted = call(6, "graph.retract.propose", {"id": edge["id"], "expected_version": 2, "rationale": "recheck"})
        assert retracted["edge"]["status"] == "retraction_proposed"
        assert retracted["edge"]["rationale"] == "check the runtime boundary"
        assert retracted["edge"]["retraction_rationale"] == "recheck"
        blocked = rpc(36, "tools/call", {"name": "graph.link.propose", "arguments": {
            "from": {"id": edge["from"]}, "to": {"id": edge["to"]},
            "relation": "depends_on", "rationale": "new assertion during retraction"}})
        assert blocked.get("isError") or not blocked.get("structuredContent", {}).get("ok"), blocked
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
        saved_edge = call(10, "graph.get", {"id": edge["id"]})["edge"]
        assert saved_edge["status"] == "retraction_proposed" and saved_edge["rationale"] == "check the runtime boundary"
        assert len(saved_edge["assertions"]) == 2 and saved_edge["assertions"][1]["evidence_refs"] == ["source:two"]
        assert saved_edge["retraction_rationale"] == "recheck"
        first_assertion = call(11, "graph.get", {"id": edge["id"], "assertion_offset": 0})["assertion_page"]
        assert first_assertion["total"] == 2 and first_assertion["next_offset"] == 1
        second_assertion = call(12, "graph.get", {"id": edge["id"], "assertion_offset": 1})["assertion_page"]
        assert second_assertion["items"][0]["evidence_refs"] == ["source:two"] and second_assertion["next_offset"] is None
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
