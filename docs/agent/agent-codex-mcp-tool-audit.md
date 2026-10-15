# Codex MCP tool audit

This is a local Codex connection for exercising llama-agent's MCP-exported
native tools. Codex (GPT-5.6-luna) is the MCP client and chooses/calls tools;
the optional Qwen daemon is a separate llama-agent runtime and is not the model
behind these Codex calls. The profile excludes policy-gated writes, workspace
patching and build/test execution. Dataset materialization is confined to the
audit-specific Cozo store under `var/codex-mcp-tool-audit/`.

## Setup

Build `llama-agent-mcp-stdio-server` in the existing Vulkan/Cozo build tree.
Add the entry from
[`codex-mcp-tools.config.toml`](../examples/codex-mcp-tools.config.toml) to
`~/.codex/config.toml`, replacing the checkout path with its absolute path.
The launcher pins the server to this checkout's tool profile and repository
root. Restart or reload Codex after changing MCP configuration.

The host profile is
[`agent-host-config-codex-mcp-tools.json`](../examples/agent-host-config-codex-mcp-tools.json).
It uses the existing OpenAlex provider fragment, isolated Cozo data/resource
stores, local DOCX/XLSX processor policies and sets
`allow_policy_gated_writes=false`.

## Verified setup state (2026-10-15)

- The stdio server built in `build-agent-vulkan-cozo` and its registered
  `llama-agent-mcp-stdio-client-ctest` passed.
- A direct MCP `initialize` + `tools/list` protocol check advertised 52 tools,
  including `openalex.*`, `dataset.*`, `data.*`, `statistics.*`, `document.*`,
  `resource.*`, `web.*`, diagnostics, repository/workspace read tools and
  `math.calculate`. This check verified registration/schema discovery, not
  successful execution of every operation.
- The Codex global MCP entry `llama-agent-tool-audit` was added and reports
  enabled. A separate `codex exec` did not expose it to the model: the first
  attempt timed out during MCP startup/handshake, and the next run selected
  unrelated built-in surfaces before any audit call. A final one-call
  `math.calculate` probe also reported the server tool unavailable and emitted
  no llama-agent tool-call event. Stop and reload/restart the active Codex
  runtime, then verify the server is callable before running the model audit.
  Do not count those attempts as tool tests.
- A narrow CSV/OpenAlex audit prompt and trace locations are prepared, but no
  Codex-driven tool call is yet verified. No raw repository diff, CSV rows or
  PDF full text were sent in the attempted runs.
- Local document tools require host-owned `resource_ref` values. The stdio
  server has no local-file-to-resource acquisition call, so checked-in DOCX/PDF
  fixtures cannot yet be passed to `document.tables` / `resource.read` directly.
  Public URLs can use `web.fetch`; OpenAlex can provide public OA locations.
- No `.xls`/`.xlsx` fixture was found. XLSX processor configuration is present,
  but workbook ingestion remains unverified until a workbook fixture and
  resource reference are available.

## Suggested audit cases

Run these cases with `llama_agent_tool_audit` selected and request a brief
per-case record of tool name, arguments, result, expected result and any
failure layer:

1. Inspect and validate
   `pocs/agent/smoke/data/fixtures/document-table/TAB6695_sv.csv`; discover its
   schema first, then validate a real column with `not_null`.
2. After a host-owned resource acquisition path is available, read tables from
   `pocs/agent/smoke/data/fixtures/document-table/TAB6695_sv.docx` and inspect
   `tools/ui/tests/stories/fixtures/assets/example.pdf` through the supported
   document/resource route. Until then, mark these as unavailable rather than
   passing filesystem paths to tools that require opaque resource references.
3. Query OpenAlex for a topic or a known work, inspect the selected record's
   open-access location, then fetch/read the PDF through the resource pipeline
   if the provider supplies a usable public URL.
4. Run one bounded `web.search` and a separate direct `web.fetch` against a
   public text resource. Do not treat search and fetch as interchangeable.
5. Inspect `repository.status` and `repository.diff` for a read-only code
   comparison. Use `diagnostics.symbol` or `diagnostics.references` on a
   symbol with a known source location.
6. Feed real compiler dependency output from `clang -M` to
   `diagnostics.include_graph`; record that this tool parses compiler output
   rather than executing clang itself.
7. Exercise a simple arithmetic request with `math.calculate` and report that
   the former `calculator` name is not advertised.

There is currently no checked-in `.xls`/`.xlsx` fixture in the searched
agent/docs/tools/test paths. Add a dedicated workbook fixture before claiming
spreadsheet ingestion is covered; do not silently substitute CSV.

## Trace

Use Codex's JSON event output to preserve actual MCP calls, arguments, results
and model responses. Keep generated traces outside Git (for example under
`var/codex-mcp-tool-audit/`) and redact any personal or external data before
sharing them.
