#!/usr/bin/env python3
"""Exercise clangd through MCP with two translation units and a persistent session."""
import json
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import time

server = Path(sys.argv[1]).resolve()
with tempfile.TemporaryDirectory(prefix="agent-clangd-workspace-") as temporary:
    root = Path(temporary)
    (root / "api.h").write_text("int fixture_target(int value);\n")
    (root / "target.cpp").write_text('#include "api.h"\nint fixture_target(int value) { return value + 1; }\n')
    (root / "caller.cpp").write_text('#include "api.h"\nint fixture_caller() { return fixture_target(2); }\n')
    commands = [{"directory": str(root), "file": str(root / name),
                 "arguments": ["c++", "-std=c++17", "-I", str(root), "-c", str(root / name)]}
                for name in ("target.cpp", "caller.cpp")]
    (root / "compile_commands.json").write_text(json.dumps(commands))
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "target.cpp"], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Agent Test", "-c", "user.email=agent@example.invalid", "commit", "-qm", "fixture"], check=True)
    config = {"schema_version": 1, "model": {"backend": "server-context"},
              "tools": {"profile": "developer-read", "repository_root": str(root)},
              "diagnostics": {"semantic_backend": "clangd", "compile_commands": str(root),
                              "semantic_request_timeout_ms": 20000},
              "limits": {"tool_timeout_ms": 30000}}
    (root / "config.json").write_text(json.dumps(config))
    invalid = dict(config)
    invalid["diagnostics"] = dict(config["diagnostics"], semantic_request_timeout_ms=99)
    (root / "invalid-config.json").write_text(json.dumps(invalid))
    rejected = subprocess.run([str(server), "--config", str(root / "invalid-config.json"),
                               "--repository-root", str(root)], input="", capture_output=True,
                              text=True, timeout=10)
    assert rejected.returncode != 0 and "semantic_request_timeout_ms" in rejected.stderr, rejected.stderr
    p = subprocess.Popen([str(server), "--config", str(root / "config.json"), "--repository-root", str(root)],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    lines = queue.Queue()
    threading.Thread(target=lambda: [lines.put(line) for line in p.stdout], daemon=True).start()
    def send(message):
        p.stdin.write(json.dumps(message) + "\n")
        p.stdin.flush()
    def rpc(ident, method, params):
        send({"jsonrpc": "2.0", "id": ident, "method": method, "params": params})
        response = json.loads(lines.get(timeout=45))
        if response.get("id") != ident or "error" in response:
            raise RuntimeError(f"unexpected RPC response: {response}")
        return response["result"]
    def call(ident, name, args):
        result = rpc(ident, "tools/call", {"name": name, "arguments": args})
        if result.get("isError") or not result.get("structuredContent", {}).get("ok"):
            raise RuntimeError(f"{name} failed: {result}")
        return result["structuredContent"]["result"]
    try:
        rpc(1, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                               "clientInfo": {"name": "workspace-test", "version": "1"}})
        send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        advertised = rpc(40, "tools/list", {})
        diff_tool = next(tool for tool in advertised["tools"] if tool["name"] == "repository.diff")
        assert "path" in diff_tool["inputSchema"]["properties"] and "include_untracked" in diff_tool["inputSchema"]["properties"]
        diff = call(41, "repository.diff", {"path": "caller.cpp", "include_untracked": True})
        assert diff["include_untracked"] and "+int fixture_caller" in diff["summary"]
        inspect = call(2, "code.inspect", {})
        assert inspect["compile_commands"] == str(root) and inspect["index_coverage"] == "unknown"
        assert inspect["compilation_database_unique_sources"] == 2 and inspect["provider_status"]["session"] == "not_running"
        inspected_file = call(19, "code.inspect", {"path": "caller.cpp"})
        assert inspected_file["compilation_database_status"] == "available" and inspected_file["source_in_database"] is True
        inspected_header = call(20, "code.inspect", {"path": "api.h"})
        assert inspected_header["source_in_database"] is False
        # A no-hint query starts the project index without the caller opening
        # either source. Index completion is asynchronous, so allow it time.
        deadline = time.monotonic() + 30
        while True:
            cold = call(21, "code.symbols", {"symbol": "fixture_caller"})
            if any(item["path"] == "caller.cpp" for item in cold["items"]): break
            if time.monotonic() >= deadline: raise AssertionError(f"project index did not find caller: {cold}")
            time.sleep(0.2)
        started = call(42, "code.inspect", {})
        assert started["provider_status"]["session"] == "initialized" and started["provider_status"]["project_index_requested"]
        assert started["provider_status"]["observation_may_be_stale"] is True
        caller = call(3, "code.symbols", {"symbol": "fixture_caller", "path_hint": "caller.cpp"})
        assert caller["items"] and caller["complete"] is False
        target = call(4, "code.symbols", {"symbol": "fixture_target", "path_hint": "target.cpp"})
        assert any(item["path"] == "target.cpp" for item in target["items"])
        refs = call(5, "code.references", {"symbol": "fixture_target", "definition_path": "target.cpp", "definition_line": 2})
        assert refs["complete"] is False and any(item["path"] == "caller.cpp" for item in refs["items"]), refs
        calls = call(6, "code.calls", {"symbol": "fixture_target", "path_hint": "caller.cpp", "line": 2, "direction": "callers"})
        assert calls["complete"] is False and any("fixture_caller" in node["label"] for node in calls["visualization"]["nodes"]), calls
        index_dir = root / ".cache" / "clangd" / "index"
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not list(index_dir.glob("*")):
            time.sleep(0.1)
        assert list(index_dir.glob("*")), "clangd did not persist a workspace index"
        print("clangd workspace MCP: symbols, cross-file references, calls and coverage passed")
    finally:
        p.terminate()
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()

    assert list(index_dir.glob("*")), "clangd index was lost after server shutdown"
    print("clangd workspace MCP: disk index remains after server shutdown")

    p = subprocess.Popen([str(server), "--config", str(root / "config.json"), "--repository-root", str(root)],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    lines = queue.Queue()
    threading.Thread(target=lambda: [lines.put(line) for line in p.stdout], daemon=True).start()
    try:
        rpc(30, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                "clientInfo": {"name": "restart-test", "version": "1"}})
        send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        deadline = time.monotonic() + 30
        while True:
            found = call(31, "code.symbols", {"symbol": "fixture_caller"})
            if any(item["path"] == "caller.cpp" for item in found["items"]): break
            if time.monotonic() >= deadline: raise AssertionError(f"restart lookup failed: {found}")
            time.sleep(0.2)
        print("clangd workspace MCP: project lookup after server restart passed")
    finally:
        p.terminate()
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()

    # A fake provider hangs once and then succeeds after the host restarts it.
    fake = root / "fake-clangd"
    fake.write_text('''#!/usr/bin/env python3
import json, os, sys, time
marker = os.path.join(os.path.dirname(__file__), "timed-out")
while True:
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line: sys.exit(0)
        if line in (b"\\r\\n", b"\\n"): break
        key, value = line.decode().split(":", 1)
        headers[key.lower()] = value.strip()
    msg = json.loads(sys.stdin.buffer.read(int(headers["content-length"])))
    if "id" not in msg: continue
    method = msg.get("method")
    if method == "workspace/symbol" and not os.path.exists(marker):
        open(marker, "w").close()
        time.sleep(5)
    result = {"capabilities": {}} if method == "initialize" else [] if method == "workspace/symbol" else None
    body = json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": result}).encode()
    sys.stdout.buffer.write(("Content-Length: %d\\r\\n\\r\\n" % len(body)).encode() + body)
    sys.stdout.buffer.flush()
''')
    fake.chmod(0o755)
    config["diagnostics"]["clangd_executable"] = str(fake)
    config["diagnostics"]["semantic_request_timeout_ms"] = 300
    (root / "timeout-config.json").write_text(json.dumps(config))
    p = subprocess.Popen([str(server), "--config", str(root / "timeout-config.json"), "--repository-root", str(root)],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    lines = queue.Queue()
    threading.Thread(target=lambda: [lines.put(line) for line in p.stdout], daemon=True).start()
    try:
        rpc(10, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                "clientInfo": {"name": "timeout-test", "version": "1"}})
        send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        timed_out = rpc(11, "tools/call", {"name": "code.symbols", "arguments": {"symbol": "fixture_target"}})
        error = timed_out.get("structuredContent", {}).get("error", {})
        assert timed_out.get("isError") and error.get("class") == "timeout", timed_out
        recovered = call(12, "code.symbols", {"symbol": "fixture_target"})
        assert recovered["complete"] is False
        print("clangd workspace MCP: timeout classification and session recovery passed")
    finally:
        p.terminate()
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
