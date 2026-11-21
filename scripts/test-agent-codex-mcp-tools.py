#!/usr/bin/env python3
"""Probe standard MCP stdio and record real Codex tool calls, independently of llama-agent's client."""

import argparse
import json
from pathlib import Path
import queue
import shutil
import subprocess
import threading
import time


def receive_lines(stream, lines):
    for line in stream:
        lines.put(line)
    lines.put(None)


def probe(launcher, workdir, check_web_limit=False, resource_uri=None, timeout=180,
          edge_tools=False, repository_status=False, clangd=False):
    with (workdir / "server-stderr.log").open("w") as diagnostics, \
            (workdir / "protocol.jsonl").open("w") as trace:
        process = subprocess.Popen(
            ["/bin/bash", str(launcher), "--memory-project", "codex-audit-smoke"], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=diagnostics, text=True, encoding="utf-8")
        lines = queue.Queue()
        threading.Thread(target=receive_lines, args=(process.stdout, lines), daemon=True).start()

        def rpc(request_id, method, params):
            request = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
            trace.write(json.dumps({"direction": "request", "message": request}) + "\n")
            trace.flush()
            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()
            line = lines.get(timeout=timeout)
            if line is None:
                raise RuntimeError("MCP server closed stdout; see server-stderr.log")
            response = json.loads(line)
            trace.write(json.dumps({"direction": "response", "message": response}) + "\n")
            trace.flush()
            if response.get("id") != request_id or "error" in response:
                raise RuntimeError(f"unexpected MCP response: {response}")
            return response["result"]

        try:
            initialized = rpc(1, "initialize", {
                "protocolVersion": "2025-11-25", "capabilities": {},
                "clientInfo": {"name": "codex-audit-preflight", "version": "1"}})
            process.stdin.write('{"jsonrpc":"2.0","method":"notifications/initialized"}\n')
            process.stdin.flush()
            rpc(2, "ping", {})
            inventory = rpc(3, "tools/list", {})["tools"]
            (workdir / "tools.json").write_text(json.dumps(inventory, indent=2) + "\n")
            names = {tool["name"] for tool in inventory}
            expected = {f"openalex.{operation}" for operation in (
                "listWorks", "getWork", "listAuthors", "getAuthor",
                "listInstitutions", "getInstitution", "listTopics", "getTopic")}
            if not expected <= names:
                raise RuntimeError(f"missing configured OpenAlex tools: {sorted(expected - names)}")
            math = rpc(4, "tools/call", {"name": "math.calculate", "arguments": {"expression": "2+3"}})
            if math.get("isError") or math.get("structuredContent", {}).get("result", {}).get("value") != 5:
                raise RuntimeError(f"unexpected calculator result: {math}")
            graph = rpc(19, "tools/call", {"name": "graph.find", "arguments": {"query": "planning"}})
            if graph.get("isError") or not graph.get("structuredContent", {}).get("ok"):
                raise RuntimeError(f"graph lookup failed: {graph}")
            if check_web_limit:
                bounded = rpc(5, "tools/call", {"name": "web.fetch", "arguments": {
                    "url": "https://example.com/", "max_bytes": 32}})
                failure = bounded.get("structuredContent", {}).get("error", {})
                if not bounded.get("isError") or failure.get("code") != "tool.web_fetch.byte_limit":
                    raise RuntimeError(f"web byte limit was not classified as a limit: {bounded}")
                recovered = rpc(6, "tools/call", {"name": "math.calculate", "arguments": {"expression": "2+3"}})
                if recovered.get("isError"):
                    raise RuntimeError("server failed after the bounded fetch error")
            if resource_uri:
                read = rpc(7, "tools/call", {"name": "resource.read", "arguments": {
                    "uri": resource_uri, "representation": "text", "max_bytes": 2000}})
                print(f"resource_read_probe={json.dumps(read, ensure_ascii=False)}", flush=True)
            if edge_tools:
                memory = rpc(8, "tools/call", {"name": "memory.get", "arguments": {
                    "id": "codex-mcp-audit-missing-memory"}})
                memory_error = memory.get("structuredContent", {}).get("error", {})
                if not memory.get("isError") or memory_error.get("code") != "tool.memory_get.unavailable":
                    raise RuntimeError(f"unexpected memory.get missing-record result: {memory}")
                crash = rpc(9, "tools/call", {"name": "diagnostics.native_crash", "arguments": {
                    "executable": "agent-resource://resource/missing-audit-executable",
                    "dump": "agent-resource://resource/missing-audit-dump"}})
                crash_error = crash.get("structuredContent", {}).get("error", {})
                if not crash.get("isError") or crash_error.get("code") != "tool.diagnostics.native_crash.invalid_dump":
                    raise RuntimeError(f"unexpected native_crash invalid-fixture result: {crash}")
                recovered = rpc(10, "tools/call", {"name": "math.calculate", "arguments": {"expression": "2+3"}})
                if recovered.get("isError"):
                    raise RuntimeError("server failed after the missing-memory and invalid-dump calls")
                print("edge_tool_probes=passed memory.get=bounded_not_found diagnostics.native_crash=invalid_fixture", flush=True)
            if repository_status:
                changed = rpc(11, "tools/call", {"name": "repository.changed_files", "arguments": {}})
                changed_result = changed.get("structuredContent", {}).get("result", {})
                if changed.get("isError") or not isinstance(changed_result.get("files"), list):
                    raise RuntimeError(f"repository.changed_files did not return a bounded file list: {changed}")
                print("repository_changed_files=passed files=" + str(len(changed_result["files"])) +
                      " truncated=" + str(changed_result.get("truncated")), flush=True)
            if clangd:
                inspected = rpc(18, "tools/call", {"name": "code.inspect", "arguments": {}})
                inspected_data = inspected.get("structuredContent", {}).get("result", {})
                if inspected.get("isError") or inspected_data.get("index_coverage") != "unknown" or not inspected_data.get("compile_commands"):
                    raise RuntimeError(f"clangd configuration or coverage not reported: {inspected}")
                # Exercise the canonical code tools in one long-lived MCP
                # session. The file hint must work on a cold clangd index.
                if "code.impact" not in names:
                    raise RuntimeError("code.impact is missing from the Codex audit profile")
                source = "common/agent/adaptation/flydelta/flydelta-deep-search.cpp"
                symbol_name = "common_flydelta_run_deep_search_batched"
                found = rpc(16, "tools/call", {"name": "code.symbols", "arguments": {
                    "symbol": symbol_name, "path_hint": source, "max_results": 5}})
                found_data = found.get("structuredContent", {}).get("result", {})
                locations = [item for item in found_data.get("items", []) if item.get("path") == source]
                if found.get("isError") or not locations or found_data.get("complete") is not False:
                    raise RuntimeError(f"cold clangd symbol lookup or coverage failed: {found}")
                callers = rpc(17, "tools/call", {"name": "code.calls", "arguments": {
                    "symbol": symbol_name, "path_hint": source, "line": locations[0]["line"],
                    "direction": "callers", "max_results": 16}})
                callers_data = callers.get("structuredContent", {}).get("result", {})
                if callers.get("isError") or callers_data.get("complete") is not False:
                    raise RuntimeError(f"clangd call hierarchy or coverage failed: {callers}")
                print("clangd_code_probe=passed cold_file_hint=true coverage=unknown", flush=True)
                outside = rpc(12, "tools/call", {"name": "code.calls", "arguments": {
                    "symbol": "root", "path_hint": "../../../../etc/passwd",
                    "line": 1, "direction": "both", "max_results": 3}})
                outside_error = outside.get("structuredContent", {}).get("error", {})
                if not outside.get("isError") or outside_error.get("code") != "tool.code.call_hierarchy.invalid_arguments":
                    raise RuntimeError(f"call hierarchy accepted a source outside the repository: {outside}")
                mismatched = rpc(13, "tools/call", {"name": "code.calls", "arguments": {
                    "symbol": "symbol_fallback",
                    "path_hint": "common/agent/tooling/adapters/families/diagnostics-adapters.cpp",
                    "line": 1, "direction": "both", "max_results": 3}})
                mismatch_error = mismatched.get("structuredContent", {}).get("error", {})
                if not mismatched.get("isError") or mismatch_error.get("code") != "tool.code.call_hierarchy.invalid_arguments":
                    raise RuntimeError(f"call hierarchy accepted a source line without its symbol: {mismatched}")
                hierarchy = rpc(14, "tools/call", {"name": "code.calls", "arguments": {
                    "symbol": symbol_name, "path_hint": source,
                    "line": locations[0]["line"], "direction": "both", "max_results": 3}})
                if hierarchy.get("isError"):
                    raise RuntimeError(f"call hierarchy failed: {hierarchy}")
                symbol = rpc(15, "tools/call", {"name": "code.symbols", "arguments": {
                    "symbol": "common_schema_normalize_and_validate_object", "max_results": 3}})
                print("clangd_symbol_probe=" + json.dumps(symbol, ensure_ascii=False), flush=True)
                print("clangd_call_hierarchy_probe=" + json.dumps(hierarchy, ensure_ascii=False), flush=True)
            print(f"stdio_probe=passed protocol={initialized['protocolVersion']} tools={len(names)} openalex=8", flush=True)
            return names
        finally:
            process.stdin.close()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--check-web-limit", action="store_true", help="Live, bounded example.com byte-limit regression")
    parser.add_argument("--probe-resource-uri", help="Call resource.read directly and capture server-side diagnostics")
    parser.add_argument("--probe-timeout", type=int, default=180,
                        help="Maximum seconds to wait for each stdio MCP reply (default: 180)")
    parser.add_argument("--probe-edge-tools", action="store_true",
                        help="Exercise safe missing-record/invalid-resource paths for memory.get and diagnostics.native_crash")
    parser.add_argument("--probe-repository-status", action="store_true",
                        help="Exercise repository.changed_files against the configured checkout")
    parser.add_argument("--probe-clangd", action="store_true",
                        help="Exercise clangd-backed symbol and call-hierarchy calls with a known source location")
    parser.add_argument("--suite", choices=("connection", "tools"), default="connection")
    parser.add_argument("--group", choices=("openalex", "eurostat", "resources", "data", "repository", "diagnostics", "memory"))
    parser.add_argument("--prompt-file", type=Path, help="Use a reviewed task prompt for one selected group")
    parser.add_argument("--require-pdf-text", action="store_true", help="Require successful PDF fetch and derived text read")
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--codex", default=shutil.which("codex"))
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--workdir", type=Path, default=Path.home() / "llama-agent-smokes" / "codex-mcp" / time.strftime("%Y%m%d-%H%M%S"))
    args = parser.parse_args()
    if args.group and args.suite != "tools":
        parser.error("--group requires --suite tools")
    if args.prompt_file and args.suite == "tools" and not args.group:
        parser.error("--prompt-file requires one --group when using --suite tools")
    repo = Path(__file__).resolve().parent.parent
    launcher = repo / "scripts/agent-codex-mcp-tools.sh"
    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    names = probe(launcher, workdir, args.check_web_limit, args.probe_resource_uri,
                  args.probe_timeout, args.probe_edge_tools, args.probe_repository_status,
                  args.probe_clangd)
    if args.probe_only:
        return
    if not args.codex:
        raise RuntimeError("Codex executable was not found")

    connection_prompt = """Use only the llama-agent-tool-audit MCP server for this audit. Do not use shell, browser, files, or other servers.
Call math.calculate for 2+3 and openalex.listWorks to search for 'Attention Is All You Need', requesting at most one record.
Report tool name, arguments, actual result and expected result. An unavailable tool is a failure; do not invent results.
"""
    audit_prompt = """Use only the llama-agent-tool-audit MCP server for this audit. Do not use shell, browser, files, or other servers.
Exercise every advertised tool below. Use its actual schema; batch independent calls where possible.
Keep results bounded. The isolated audit database/resources may contain derived test data.
Use only these repository paths: README.md, CMakeLists.txt, common/agent/tooling/contracts/schema-contract.cpp,
pocs/agent/smoke/data/fixtures/document-table/TAB6695_sv.csv and TAB6695_sv.docx in that same directory,
and tools/ui/tests/stories/fixtures/assets/example.pdf. Do not inspect home directories, credentials or arbitrary repository diffs.
Repository status/log/diff/changed_files operate at repository scope when their schema has no path filter.
For file reads, searches and directory listings prefer the named paths. Read at most 40 lines of source.
For dataset tools inspect the CSV schema first, then choose actual columns and not_null/unique rules. dataset.select must use only a real dataset returned by dataset.list; never pass OpenAPI operation names such as listWorks as dataset names.
Use returned dataset/resource references for data/statistics/document/resource tools; never invent a successful reference. resource.inspect and resource.read use the `uri` field, not `id`; never pass a repository path to either tool. document.tables/table need a compatible document representation, not merely a PDF's OCR text resource.
OpenAlex: search works/authors/institutions/topics with limit 1, then get the corresponding returned entity ID.
Web: one bounded search; fetch a public text or PDF URL returned by OpenAlex or https://example.com/.
Use returned resource refs for inspect/read and document tables when supported.
Memory queries may return an empty set; memory.get/plan.get can report missing fixture without implying a server crash.
For diagnostics.compile, use compiler error output; diagnostics.format expects formatter output such as a 'would reformat' line;
diagnostics.include_graph expects clang -H lines ('. header.h', '.. nested.h') or explicit 'parent -> child' edges, not compiler errors or flat clang -M output.
For this fixture, use exact symbol common_schema_normalize_and_validate_object at line 152 in common/agent/tooling/contracts/schema-contract.cpp for references/call hierarchy; line must actually contain the symbol.
If diagnostics need source positions, use the named source file. Do not execute a compiler/build or modify workspace files.
For tools lacking required host fixtures, state precisely which input or capability is missing.
Every advertised tool must have a recorded call or an explicit not_tested reason. Do not call a catalog-only check a successful tool test.
End with one row per tool: name, pass/fail/not_tested, expected versus actual, and actionable deficiency.
"""
    eurostat_prompt = """Use only the llama-agent-tool-audit MCP server for this end-to-end Eurostat evaluation. Do not use shell, browser, files, or other servers.
User request: Find a Eurostat table about population density, verify its metadata, then retrieve Sweden's 2020 observation and inspect the resulting dataset.

Follow the evidence chain; do not begin with a hard-coded dataset code or API URL:
1. Call web.fetch on https://ec.europa.eu/eurostat/api/dissemination/catalogue/toc/txt?lang=en with accept text/plain and max_bytes 4194304. Use its returned resource_ref (not uri) with dataset.inspect to materialize the catalogue. The inline preview is intentionally bounded; query the materialized dataset rather than trying to read the full TSV into the model.
2. Use data.query on that returned dataset, selecting code/title and filtering title for population density. Prefer this exact canonical predicate: `where: [{"field":"title","operator":"contains","value":"density"}]`. If the MCP validator rejects that documented shape, retry using the simple expression `where: "title contains \"density\""` and report the schema/validator mismatch. Rows are positional arrays: map them using the returned columns array. Choose only a code/title actually returned by the catalogue; report the evidence supporting the choice.
3. Call eurostat_sdmx.getStructure for the returned candidate code (artifactType dataflow, agencyID ESTAT, version 1.0, references children, detail full). Do not send `lang`: the live SDMX endpoint rejects that query parameter. Keep reference expansion at `children` so the structural response remains within the tool-result budget; use versions returned by the dataflow for versioned structures. Verify the English dataflow title/description and dimension/codelist references.
4. Call eurostat_sdmx.getStructure again for that same catalogue-returned code with artifactType contentconstraint, agencyID ESTAT, version 1.0. Use the returned constraint values and the dataflow's overall period annotations to verify whether geography SE and year 2020 are actually available. Do not fetch the global GEO code list; it is large and the dataset-specific constraint is the relevant evidence.
5. Only if these metadata responses verify the concept and both requested codes, call eurostat.getData with the selected datasetCode and verified geography/time dimensions. Inspect the returned resource using dataset.inspect, then dataset.schema and dataset.sample using the returned dataset reference. Report actual row/column counts, relevant values and retained source/dimension/value provenance. Otherwise report exactly which metadata condition was absent and stop rather than guessing.

Keep calls bounded. Report each tool, arguments, result or exact error, and distinguish verified facts from unresolved metadata. Never invent a returned reference or claim success from a preview alone.
"""
    calls = []
    groups = [("connection", {"math.calculate", "openalex.listWorks"})]
    if args.suite == "tools":
        groups = [(group, {name for name in names if any(name.startswith(prefix) for prefix in prefixes)})
                  for group, prefixes in (
                      ("openalex", ("openalex.",)),
                      ("eurostat", ("eurostat.", "eurostat_sdmx.")),
                      ("resources", ("web.", "resource.", "document.")),
                      ("data", ("dataset.", "data.", "statistics.")),
                      ("repository", ("repository.", "workspace.")),
                      ("diagnostics", ("diagnostics.",)),
                      ("memory", ("memory.", "math.", "time.")))]
        assigned = set().union(*(group_names for _, group_names in groups))
        if names - assigned:
            groups.append(("other", names - assigned))
        dependencies = {
            "openalex": {"dataset.sample", "resource.read"},
            "eurostat": {"web.fetch", "resource.inspect", "resource.read",
                         "dataset.inspect", "dataset.schema", "dataset.sample", "data.query"},
            "resources": {"openalex.listWorks", "openalex.getWork", "dataset.sample"},
        }
        groups = [(group, group_names | (dependencies.get(group, set()) & names))
                  for group, group_names in groups if not args.group or group == args.group]
    exit_codes = {}
    for group, group_names in groups:
        group_dir = workdir if args.suite == "connection" else workdir / group
        group_dir.mkdir(exist_ok=True)
        prompt = (connection_prompt if args.suite == "connection" else
                  eurostat_prompt if group == "eurostat" else audit_prompt)
        if args.prompt_file:
            prompt = args.prompt_file.read_text()
        prompt += "\nAdvertised tools for this group:\n" + "\n".join(sorted(group_names)) + "\n"
        (group_dir / "prompt.txt").write_text(prompt)
        server_key = "mcp_servers.llama-agent-tool-audit"
        command = [args.codex, "exec", "--ignore-user-config", "--ephemeral",
                   "--model", args.model, "--sandbox", "read-only", "--json", "--cd", str(repo),
                   "-c", f'{server_key}.command="/bin/bash"',
                   "-c", f"{server_key}.args={json.dumps([str(launcher)])}",
                   "-c", f'{server_key}.env.LLAMA_AGENT_MCP_TRACE_TOOL_ERRORS="1"',
                   "-c", f"{server_key}.enabled_tools={json.dumps(sorted(group_names))}",
                   "-c", f"{server_key}.startup_timeout_sec=30",
                   "-c", f"{server_key}.tool_timeout_sec=180",
                   "-c", 'model_reasoning_effort="low"',
                   "-c", "project_doc_max_bytes=0",
                   "-o", str(group_dir / "answer.txt"), prompt]
        with (group_dir / "codex-events.jsonl").open("w") as events, \
                (group_dir / "codex-stderr.log").open("w") as diagnostics:
            environment = dict(__import__("os").environ)
            environment["LLAMA_AGENT_MCP_TRACE_TOOL_ERRORS"] = "1"
            completed = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=events,
                                       stderr=diagnostics, timeout=args.timeout, env=environment)
        exit_codes[group] = completed.returncode
        group_calls = []
        for line in (group_dir / "codex-events.jsonl").read_text().splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            item = event.get("item", {})
            if event.get("type") == "item.completed" and item.get("type") == "mcp_tool_call":
                group_calls.append(item)
        (group_dir / "calls.json").write_text(json.dumps(group_calls, indent=2) + "\n")
        calls.extend(group_calls)
        print(f"group={group} codex_exit={completed.returncode} completed_mcp_calls={len(group_calls)}", flush=True)
    (workdir / "calls.json").write_text(json.dumps(calls, indent=2) + "\n")
    called = {call.get("tool") for call in calls if call.get("server") == "llama-agent-tool-audit"}
    (workdir / "coverage.json").write_text(json.dumps({
        "advertised": sorted(names), "called": sorted(called),
        "not_called": sorted(names - called), "codex_exit_codes": exit_codes}, indent=2) + "\n")
    print(f"completed_mcp_calls={len(calls)} distinct_tools={len(called)} trace={workdir}")
    required = {"math.calculate", "openalex.listWorks"} if not args.group else set()
    successful = {call.get("tool") for call in calls
                  if call.get("status") == "completed" and not call.get("error")}
    if any(exit_codes.values()) or not calls or not required <= successful:
        raise RuntimeError("Codex did not complete the required actual MCP calls; inspect trace")
    if args.require_pdf_text:
        by_tool = {call.get("tool"): call for call in calls}
        fetch = (by_tool.get("web.fetch", {}).get("result", {}).get("structured_content", {})
                 .get("result", {}))
        read_call = by_tool.get("resource.read", {})
        read = read_call.get("result", {}).get("structured_content", {}).get("result", {})
        if (by_tool.get("web.fetch", {}).get("status") != "completed" or
                fetch.get("content_type", "").split(";")[0].strip().lower() != "application/pdf" or
                fetch.get("truncated") or not fetch.get("resource_ref") or
                read_call.get("status") != "completed" or
                not read.get("content") or
                "attention is all you need" not in read.get("content", "").lower()):
            raise RuntimeError("PDF smoke did not prove a complete PDF fetch and derived text containing the article title")


if __name__ == "__main__":
    main()
