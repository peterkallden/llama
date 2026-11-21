#!/usr/bin/env python3
"""Exercise project graph scope through the daemon's MCP HTTP dispatcher, without model inference."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request


def main():
    daemon = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory(prefix="agent-daemon-graph-") as temporary:
        root = Path(temporary)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        tokens = [
            {"id": name, "token_env": "GRAPH_SMOKE_" + name.upper(), "audience": "llama-agent",
             "namespace": namespace, "project": project,
             "tool_profile": "graph-project" if name == "writer" else "developer-read",
             "allow_writes": name == "writer"}
            for name, namespace, project in [("writer", "tenant-a", "project-a"),
                                               ("reader", "tenant-a", "project-a"),
                                               ("other", "tenant-a", "project-b"),
                                               ("tenant", "tenant-b", "project-a")]
        ]
        config = {"schema_version": 1, "model": {"path": str(root / "unused.gguf")},
                  "stores": {"data": {"backend": "cozo", "path": str(root / "data.cozo")}},
                  "tools": {"profile": "graph-project", "repository_root": str(root)},
                  "mcp": {"inbound": {"enabled": True, "listen": "127.0.0.1", "port": port,
                                      "path": "/mcp", "tokens": tokens}}}
        path = root / "config.json"
        path.write_text(json.dumps(config))
        environment = os.environ.copy()
        for token in tokens:
            environment[token["token_env"]] = token["id"] + "-secret"
        with (root / "daemon.log").open("w+") as log:
            daemon_process = subprocess.Popen([str(daemon), "--config", str(path)], stdin=subprocess.PIPE,
                                              stdout=subprocess.DEVNULL, stderr=log, env=environment)
            def request(token, method, params, headers=None):
                body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
                request_headers = {"Content-Type": "application/json", "Authorization": "Bearer " + token + "-secret"}
                request_headers.update(headers or {})
                message = urllib.request.Request(f"http://127.0.0.1:{port}/mcp", data=body, headers=request_headers)
                with urllib.request.urlopen(message, timeout=10) as response:
                    return json.load(response), response.headers

            def connect(token):
                value, headers = request(token, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                    "clientInfo": {"name": "graph-scope-smoke", "version": "1"}})
                if "error" in value:
                    raise RuntimeError(f"initialize failed: {value}")
                return {"Mcp-Session-Id": headers["Mcp-Session-Id"], "MCP-Protocol-Version": "2025-06-18"}

            def call(token, headers, tool, arguments):
                value, _ = request(token, "tools/call", {"name": tool, "arguments": arguments}, headers)
                result = value["result"]
                if result.get("isError") or not result.get("structuredContent", {}).get("ok"):
                    raise RuntimeError(f"{tool} failed: {result}")
                return result["structuredContent"]["result"]

            try:
                for attempt in range(100):
                    try:
                        writer = connect("writer")
                        break
                    except urllib.error.URLError:
                        if daemon_process.poll() is not None:
                            raise RuntimeError("daemon exited before MCP startup")
                        time.sleep(0.1)
                else:
                    raise RuntimeError("daemon MCP startup timed out")
                reader, other, tenant = (connect(name) for name in ("reader", "other", "tenant"))
                created = call("writer", writer, "graph.link.propose", {
                    "from": {"kind": "component", "name": "scope-smoke-source"},
                    "to": {"kind": "component", "name": "scope-smoke-target"},
                    "relation": "depends_on", "rationale": "exercise daemon graph scope"})
                edge = created["edge"]
                same = call("reader", reader, "graph.search", {"from": edge["from"], "relation": "depends_on"})
                if len(same["edges"]) != 1 or same["edges"][0]["id"] != edge["id"]:
                    raise RuntimeError(f"same-project client did not see the proposal: {same}")
                denied, _ = request("reader", "tools/call", {"name": "graph.link.propose", "arguments": {
                    "from": {"id": edge["from"]}, "to": {"id": edge["to"]},
                    "relation": "related_to", "rationale": "read-only policy probe"}}, reader)
                if not denied["result"].get("isError"):
                    raise RuntimeError(f"read-only client was allowed to propose a relation: {denied}")
                for name, headers in (("other", other), ("tenant", tenant)):
                    result = call(name, headers, "graph.find", {"query": "scope-smoke-source"})
                    if result["nodes"]:
                        raise RuntimeError(f"graph leaked across {name} scope: {result}")
                print("daemon graph MCP: shared project, read-only policy and namespace/project isolation passed (no model inference)")
            except Exception:
                log.flush()
                log.seek(0)
                print(log.read(), file=sys.stderr)
                raise
            finally:
                daemon_process.terminate()
                try:
                    daemon_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    daemon_process.kill()
                    daemon_process.wait()


if __name__ == "__main__":
    main()
