# Codex MCP tool audit

This is a local Codex connection for exercising llama-agent's MCP-exported
native tools. Codex (GPT-5.6-luna) is the MCP client and chooses/calls tools;
the optional Qwen daemon is a separate llama-agent runtime and is not the model
behind these Codex calls. The profile excludes policy-gated writes, workspace
patching and build/test execution. Dataset materialization is confined to the
audit-specific Cozo store under `var/codex-mcp-tool-audit/`.

This is a selected integration-test profile, not every native catalog tool
and not a side-effect-free profile. Acquisition, dataset derivation and
session selection can write isolated audit state. The capability name
`codex.audit.tools` deliberately makes no read-only claim. The eight
`openalex.*` tools come from `openalex-provider.d/openalex.json`; they are
loaded separately from the native capability list.

## Setup

Build `llama-agent-mcp-stdio-server` in the existing Vulkan/Cozo build tree.
Add the entry from
[`codex-mcp-tools.config.toml`](../examples/codex-mcp-tools.config.toml) to
`~/.codex/config.toml`, replacing the checkout path with its absolute path.
The launcher pins the server to this checkout's tool profile and repository
root. A running Codex UI may need to reload its MCP connections to see a
newly added entry. The reproducible smoke starts a fresh Codex client and
does not depend on a UI reload.

The host profile is
[`agent-host-config-codex-mcp-tools.json`](../examples/agent-host-config-codex-mcp-tools.json).
It uses the existing OpenAlex provider fragment and the checked-in Eurostat
Statistics/SDMX adapter contracts, isolated Cozo data/resource stores, local
DOCX/XLSX processor policies and sets
`allow_policy_gated_writes=false`.

## Verified setup state (2026-10-15)

- The stdio server built in `build-agent-vulkan-cozo` and its registered
  `llama-agent-mcp-stdio-client-ctest` passed.
- A fresh MCP `initialize` + `tools/list` protocol check advertised 54 tools,
  including `openalex.*`, `dataset.*`, `data.*`, `statistics.*`, `document.*`,
  `resource.*`, `web.*`, diagnostics, repository/workspace read tools and
  `math.calculate`. This check verified registration/schema discovery, not
  successful execution of every operation.
  The audit profile now additionally configures `eurostat.getData` and
  `eurostat_sdmx.getStructure`; a running Codex MCP connection must be reloaded
  before those provider tools appear in its advertised list.
- The initial Codex attempts failed because our stdio transport incorrectly
  expected LSP `Content-Length` headers. A literal newline-delimited
  `initialize` reproduced the wait. Both client and server now use standard
  MCP JSON lines; self-client listing alone is no longer acceptance evidence.
  See [the framing bug](bugs.md#mcp-stdio-handshake-blocked-by-lsp-style-framing).
- A fresh Codex/GPT-5.6-luna run completed actual `math.calculate` and
  `openalex.listWorks` calls. `2+3` returned 5; the OpenAlex search returned
  one row and host-owned resource/dataset references. The first connection
  run exposed a second bug: stdio ignored `limits.tool_timeout_ms` and used
  the internal 1-second default. Passing the configured timeout fixed the
  false timeout. An independent delayed-provider test covers this boundary.
- A fresh Codex/GPT-5.6-luna Eurostat group run made nine calls across eight
  distinct tools. It fetched the English catalogue resource, queried
  `eurostat.getData`, fetched an SDMX structure through
  `eurostat_sdmx.getStructure`, then successfully inspected/sampled the
  returned JSON-stat resource as a one-row, nine-column dataset. The query
  returned Sweden's 2020 population density as 25.2 persons/km² and retained
  the source resource reference. Trace:
  `var/codex-mcp-tool-audit/eurostat-codex-20261015-retry/eurostat/`.
- Qwen's earlier model-selected first `web.fetch` reproduced the contract
  problem: it invented a different URL and left `accept` at `*/*`, yielding
  HTTP 404. The start transition already declared the TSV URL,
  `accept=text/plain`, and a 4 MiB bound, but initial route materialization
  asked the model to regenerate those arguments. The host now executes a
  complete schema-valid static start transition directly. A bounded Qwen
  rerun confirms `deterministic_graph_entry`, then fetches the declared URL
  with HTTP 200, `text/plain;charset=UTF-8`, 1,969,964 bytes, and a host
  resource reference. That run intentionally allowed one tool call, so the
  following `dataset.inspect` was rejected by the configured call limit; the
  full catalogue materialization remains unverified at model-runtime scale.
  Trace:
  `var/codex-mcp-tool-audit/eurostat-qwen-20261015/static-accept-rerun/qwen-cpu-workflow.log`.
- The 2026-10-16 full Qwen Instruct run exercised the new linked workflow
  chain through catalogue materialization, SDMX dataflow/constraint fetches,
  JSON-stat retrieval and `dataset.inspect`. The catalogue import completed
  with 12,241 rows in 48 batches. Qwen selected `demo_r_d3dens`, whose title
  refers to NUTS-3 regions. The initial audit incorrectly treated that title
  as evidence that the choice could not answer a national question. Subsequent
  inspection of the saved resources shows that its ContentConstraint includes
  `geo=SE`, `TIME_PERIOD=2020`, `unit=PER_KM2` and `freq=A`. The returned
  JSON-stat explicitly contains Sweden, 2020, persons per square kilometre
  and value 25.2. Regional tables can contain national aggregates; actual
  dimensions, definitions and observations determine suitability.
  This offline check supports the retrieved value. That run predates the new
  declared constraint/coverage gates in starter package v24, which are added in
  source but deliberately await build, model-free and model verification.
  Final answer generation stopped at
  context pressure (`decoded 0 tokens`), so model end-to-end completion is
  still unproven. The final-context budget also needs follow-up.
  Full trace:
  `var/codex-mcp-tool-audit/eurostat-qwen-20261016-v16-run22/qwen-e2e.log`.
- Full connection traces are under
  `/home/prbm/llama-agent-smokes/codex-mcp-20261015/connection-fixed/`:
  `prompt.txt`, `codex-events.jsonl`, `calls.json` and `answer.txt` contain
  exact input/output and actual calls. No repository contents, CSV rows or
  PDF full text were sent by this two-call connection check.
- The native `web.fetch` default and maximum are 10 MiB (10,485,760 bytes).
  A direct MCP preflight accepted a 2,215,244-byte PDF without a `max_bytes`
  argument; a separate 32-byte bound was rejected as
  `tool.web_fetch.byte_limit`, after which `math.calculate` still succeeded.
- The full public OpenAlex → PDF → OCR path was completed by Codex/GPT-5.6-luna
  on 2026-10-15. OpenAlex search returned work
  `https://openalex.org/W2626778328` (“Attention Is All You Need”, listed as
  2025); its PDF URL failed. The model reported that failure and used the
  prompt's declared arXiv fallback. `web.fetch` returned an untruncated
  `application/pdf` resource (2,215,244 bytes); `resource.read` returned a
  derived text resource with 39,969 bytes and OCR coverage of 15/15 pages.
  The extracted text contained the title and author names. The test asserted
  the PDF MIME, non-truncation, materialized source reference and title in the
  derived text. Trace: `/home/prbm/llama-agent-smokes/codex-mcp-20261015/pdf-e2e-after-workspace-fix/`.
- This run exposed and fixed two host-config seams. Relative sandbox workspace
  and artifact roots are now resolved against the repository root before
  processor construction; otherwise a local renderer's operation working
  directory caused the relative path to be applied twice. The audit profile
  also gives only PDF page rendering and Tesseract OCR a bounded 4 MiB output
  allowance and `artifact_write` scope, matching their declared output
  artifacts without widening the default policy for unrelated tools.
- The diagnostics follow-up found a separate clangd/LSP chain of failures:
  clangd was not searched on `PATH`; the pipe reader waited for a full 4 KiB
  buffer instead of returning on a complete short response; shutdown closed
  the same stdin `FILE*` both directly and in `subprocess_join`; and
  call-hierarchy requests did not send `textDocument/didOpen`. These are now
  corrected. Location queries open only canonical files inside the configured
  repository and cap source text at 4 MiB. The MCP probe now returns a real
  caller/callee result for a known in-repository function. A zero-result
  clangd workspace-symbol response is explicitly downgraded to the existing
  `text-fallback` search (`semantic:false`) so an index-not-ready response is
  not presented as semantic coverage. Trace:
  `/home/prbm/llama-agent-smokes/codex-mcp-20261015/diagnostics-clangd-real-calls/`.
- The Codex diagnostics-group run then exposed two model-facing contract gaps:
  the call-hierarchy description did not say that a source path and line are
  required, and the include-graph prompt incorrectly suggested flat `clang -M`
  output even though that format cannot encode parent/child edges. The contract
  now documents the accepted source-location forms; the include parser accepts
  clang `-H` nesting and explicit `parent -> child` edges, rejects unrecognized
  input, and the format checker recognizes common `formatting differs` and
  clang-format violation lines. The audit prompt now distinguishes compiler
  diagnostics, formatter output, and include traces. These specific cases are
  rerun in the diagnostics model smoke. The same review also found that
  `call_hierarchy.max_depth` was accepted but not implemented; the unused field
  was removed and the tool is now explicit that it returns one-hop callers and
  callees. The host verifies that the supplied source line contains the exact
  symbol and locates its token instead of silently querying an unrelated
  cursor. The final Codex/GPT-5.6-luna diagnostics run made 8 calls across all
  8 advertised diagnostics tools. Compile, format, include graph, test-failure
  grouping, references, symbol and call hierarchy returned usable bounded
  output; the crash tool was called but no compatible core fixture was
  available, so its `parse_failed` result remains a fixture limitation rather
  than a successful crash-analysis test. Trace:
  `/home/prbm/llama-agent-smokes/codex-mcp-20261015/diagnostics-codex-final/`.
- The direct PDF text parser can reject malformed UTF-8 in a PDF text layer;
  the host then uses the configured OCR fallback. Both PDF/OCR text processors
  normalize invalid sequences before JSON/resource materialization. Direct
  MCP verification read all 15 pages successfully; its trace is under
  `var/mcp-debug/direct-resource-long-wait/` in the checkout.
- The model completed the PDF pipeline with five distinct tool names called
  (`openalex.listWorks`, `openalex.getWork`, `web.fetch`, `resource.inspect`,
  `resource.read`); the failed first PDF URL is a recorded network failure,
  not a failed extraction. The more general previous audit recorded 50/52
  advertised tools as called. `memory.get` lacked a returned memory ID and
  `diagnostics.native_crash` lacked a safe dump fixture, so neither should be
  counted as behaviorally verified without those inputs.
- Discovery is not complete coverage: the current profile advertises 54
  tools. Thirteen native catalog tools are excluded by the profile:
  `artifact.export`, `development.build`, `development.test`,
  `workspace.patch`, the five memory proposal/link tools,
  `plan.propose` and the three `oracle.*` tools. In addition, `plan.get`
  and `openapi.connect` are requested by the profile but have no bound
  implementation in this standalone host and are absent from its listing.
  These fifteen cases must remain explicitly untested until their required
  host state/policy/bindings are provided.
- Local document tools require host-owned `resource_ref` values. The stdio
  server has no local-file-to-resource acquisition call, so checked-in DOCX/PDF
  fixtures cannot yet be passed to `document.tables` / `resource.read` directly.
  Public URLs can use `web.fetch`; OpenAlex can provide public OA locations.
- No `.xls`/`.xlsx` fixture was found. XLSX processor configuration is present,
  but workbook ingestion remains unverified until a workbook fixture and
  resource reference are available.

## Suggested audit cases

Run these cases with `llama-agent-tool-audit` selected and request a brief
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
4. For Eurostat, fetch the English catalogue TOC from
   `https://ec.europa.eu/eurostat/api/dissemination/catalogue/toc/txt?lang=en` with `web.fetch` and
   `accept=text/plain`, inspect the resulting dataset, resolve a returned
   candidate with `eurostat_sdmx.getStructure`, then query validated codes and
   dimensions through `eurostat.getData`. Keep metadata discovery separate
   from data retrieval.
5. Run one bounded `web.search` and a separate direct `web.fetch` against a
   public text resource. Do not treat search and fetch as interchangeable.
6. Inspect `repository.status` and `repository.diff` for a read-only code
   comparison. Use `diagnostics.symbol` or `diagnostics.references` on a
   symbol with a known source location.
7. Feed real compiler dependency output from `clang -M` to
   `diagnostics.include_graph`; record that this tool parses compiler output
   rather than executing clang itself.
8. Exercise a simple arithmetic request with `math.calculate` and report that
   the former `calculator` name is not advertised.

There is currently no checked-in `.xls`/`.xlsx` fixture in the searched
agent/docs/tools/test paths. Add a dedicated workbook fixture before claiming
spreadsheet ingestion is covered; do not silently substitute CSV.

## Trace

Use Codex's JSON event output to preserve actual MCP calls, arguments, results
and model responses. Keep generated traces outside Git (for example under
`var/codex-mcp-tool-audit/`) and redact any personal or external data before
sharing them.

Run the independent protocol preflight and then the real Codex connection
test from the checkout:

```sh
python3 scripts/test-agent-codex-mcp-tools.py --suite connection \
  --workdir "$HOME/llama-agent-smokes/codex-mcp-connection"
python3 scripts/test-agent-codex-mcp-tools.py --suite tools \
  --workdir "$HOME/llama-agent-smokes/codex-mcp-tools"
python3 scripts/test-agent-codex-mcp-tools.py --probe-only --check-web-limit \
  --probe-edge-tools --workdir "$HOME/llama-agent-smokes/codex-mcp-preflight"
```

`--probe-only` omits the model call; `--model` defaults to `gpt-5.6-luna`.
`--probe-edge-tools` safely verifies the expected missing-record response from
`memory.get` and invalid-resource response from `diagnostics.native_crash`, then
checks that a later calculator call still succeeds. These are failure-path
checks, not proof of successful memory retrieval or dump analysis. The stdio
preflight reply timeout defaults to 180 seconds to accommodate bounded local
resource conversion; it is configurable with `--probe-timeout`.
The preflight uses only Python's standard library and sends literal JSON
lines, verifies OpenAlex discovery and calls the calculator. The model run
loads only this MCP configuration, keeping its full prompt in `prompt.txt`,
Codex events in `codex-events.jsonl`, server/client stderr separately, and
actual call records in `calls.json`. `coverage.json` distinguishes advertised,
called and uncalled tools. The script fails if the model produces no actual
MCP call, even if its final answer sounds successful.

The broad `--suite tools` audit uses one fresh Codex run per tool family and
filters each client view with `enabled_tools`. This avoids repeatedly
injecting the entire catalog. Group prompts, events and answers are saved in
per-group subdirectories. Local-content model runs need authorization for
their specific fixture scope; registration or a successful connection does
not authorize exporting arbitrary repository contents.

For clangd-backed diagnostics, the direct stdio probe can be run with
`--probe-clangd`. It verifies both `diagnostics.call_hierarchy` against a
known source position and `diagnostics.symbol`; the latter reports whether
the result came from clangd or the non-semantic text fallback. This is an
integration check through the real server, not just the LSP codec unit test.
