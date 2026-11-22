# Source Code Analysis

Llama-agent exposes source navigation through the `code` tool family. The
family is separate from `diagnostics`, which covers compiler output, test
failures, formatting, include traces and native crash reports.

## Current operations

| Tool | Behavior |
| --- | --- |
| `code.inspect` | Reports whether the host has an authorized repository and which semantic operations are available. |
| `code.refresh` | Reconciles watched files and schedules a fresh clangd session for the next semantic query. |
| `code.symbols` | Finds symbol definitions through the semantic provider, or bounded text matches when no provider is configured. |
| `code.references` | Finds references through the semantic provider, or bounded text matches when no provider is configured. |
| `code.calls` | Returns one-hop callers/callees from a semantic provider and a relationship projection using the shared visualization contract. |
| `code.impact` | Traverses a bounded multi-hop call neighborhood from an explicit source location and returns the same generic relationship projection. |

Text matches are not resolved symbols. Static call relationships describe
provider-reported source structure; they do not establish path feasibility,
runtime behavior, execution, or complete change impact. `code.impact` is
bounded to depth 1–3, 2–64 nodes and 1–128 edges. It requires a semantic call
provider and source locations (including symbol names) from that provider to
continue traversal. If a provider omits a name or reports partial coverage, the
result is marked incomplete rather than filled with guessed relationships.

Traversal expands only the opposite endpoint of each incoming/outgoing relation,
resolves the starting selection to the provider's canonical location, and merges
repeated edge observations without losing call-site provenance. A self-call does
not consume two node slots. Reaching a node or edge count exactly does not imply
truncation; excluded relationships or unexpanded depth boundaries do. Provider
calls are bounded by the node budget and the tool's configured timeout, checked
between calls and after each response. An in-flight provider call remains subject
to the provider's own timeout; traversal does not interrupt that call.

Each provider page and the final tool response are limited to 64 KiB. Provider
failures retain their failure class and retryability. Result-level coverage and
limitations come from the final validated graph so that projection exclusions
cannot be presented as complete analysis. Responses declare UTF-16 columns and
the number of provider requests. Accumulated limitations are bounded explicitly.

The host owns the repository root and provider configuration. The current
clangd provider is enabled through the existing
`diagnostics.semantic_backend: clangd` host setting for compatibility with
deployed configurations. It is wired through the shared host tool resolver, so
CLI and daemon tool execution use the same bindings and resolved repository
root, including when that root comes from the host tool context. The provider and its LSP
transport are implemented under `tools/agent/code/`; native compiler and crash
diagnostics remain under `tools/agent/diagnostics/`.

The stdio MCP server retains its clangd provider while refreshing tool policy
for each call. The daemon also keeps one host-owned provider across turns so the
watcher and LSP session do not restart with every per-turn tool view. Indexing is asynchronous and the provider cannot establish
project-wide index coverage: clangd results therefore report `complete: false`
and an explicit coverage limitation even when all returned entries were parsed.
An empty result does not prove that a symbol or caller is absent. The Codex MCP
audit profile includes `code.impact`; its bounded graph has the same limitation.
For a regular source file supplied as `code.symbols.path_hint`, the provider
opens and parses that file before querying the workspace index. Directory hints
and searches without a hint remain subject to asynchronous index availability.
`code.inspect` reports the configured `compile_commands` setting and marks
project index coverage as unknown; the setting `auto` is not a resolved path.
With an explicitly configured compilation database it reports whether that
database can be read, and an optional repository-relative `path` reports
whether the source is present, or null when this cannot be determined. It also
reports the number of unique source paths and whether invalid entries were
encountered; that count is not an index coverage measurement. `provider_status` reports `start_on: semantic_query`: inspection does not start clangd.
A `not_running` session before the first semantic query is expected.
`provider_status` is a non-blocking snapshot of the session and last observed clangd index
activity. Progress notifications are read during requests, so the observation
may be stale even when an activity reports that it ended.

The host-owned file-watch service is backend-neutral and can be shared by
multiple consumers registered for the same canonical project root. Linux uses
recursive inotify watches; other platforms use bounded polling, and polling
also reconciles Linux snapshots periodically. It watches C/C++ source and
header files plus common build inputs, while skipping `.git`, build, cache and
agent data directories. It does not follow symlinks. New directories, atomic
renames, deletions, branch changes and inotify queue loss are handled as file
events or a request for reconciliation. Callbacks run outside watcher locks.

When clangd has a document open, source changes are sent as versioned LSP
`didChange` notifications and deletion closes the document. Other watched
files are sent through `workspace/didChangeWatchedFiles`. A delivered event
means only that the host observed and forwarded a file change; it does not mean
clangd's asynchronous project index has completed. `code.inspect` reports the
watcher backend, whether reconciliation is needed, and the latest observed and
consumer-acknowledged change generations. A pending delivery does not imply
clangd has consumed the change, and neither acknowledgement nor delivery proves
background indexing has completed. `index_coverage` remains `unknown`. `code.refresh`
schedules filesystem reconciliation, validates and publishes a configured
compilation database, and stops the current clangd session so the next semantic
query starts it with current inputs. It does not launch CMake or compile.

`diagnostics.compile_commands_source` and
`diagnostics.compile_commands_destination` are optional host-owned paths. The
source may be a JSON file or its containing directory. On change, the host
validates the bounded compilation database and atomically publishes it to the
destination; invalid input leaves the previous working copy in place. If no
destination is set, `diagnostics.compile_commands` is used. A configured source
can live in a build tree excluded from recursive project watching because it is
registered as an explicit file. Changes to CMake or other build inputs set
`compile_commands_may_be_stale`; the host never generates build commands.
When a configured source cannot be published, semantic queries can use the
last validated destination while `compile_commands_sync_error` and the stale
flag report the failure. `code.refresh` itself reports the failed publication.
`code.inspect` can report whether a requested source has an entry in the
selected database, including `false` when it is known to be missing.

For example, an agent host can keep its reusable clangd database outside a
frequently rebuilt checkout:

```json
{
  "diagnostics": {
    "semantic_backend": "clangd",
    "compile_commands": "/home/agent/agent-data/clangd",
    "compile_commands_source": "/home/agent/llama/build-agent-vulkan-cozo/compile_commands.json",
    "compile_commands_destination": "/home/agent/agent-data/clangd/compile_commands.json"
  }
}
```

The remote Codex/MCP client remains an ordinary client of the existing
llama-agent MCP server; watcher ownership and configuration belong to that
server or daemon host.
Clangd's background index persists under the compilation database directory's
`.cache/clangd/index` when writable and is reused across process restarts.
`diagnostics.semantic_request_timeout_ms` bounds an individual clangd request
(default 180000, range 100–180000 ms); the host tool budget also applies. A
multi-request operation can take longer than one request, and its outer budget
is checked separately. Index contents depend on the build configuration and
do not prove that every translation unit has been indexed.
For an unhinted symbol search with an explicit readable compilation database,
the provider opens one authorized translation unit to trigger clangd's own
background project indexing. Queries can return before the index is ready.

Backlog: propagate one shared operation deadline and cancellation through MCP,
the tool queue, multi-step code operations and in-flight clangd requests. The
current independent limits do not guarantee a hard end-to-end time bound.

## Provider boundary and source positions

`agent-clangd-projection` normalizes LSP locations and incoming/outgoing calls
before exposing them to tools. Incoming calls obtain their selected target;
outgoing calls obtain their selected caller. Call-site ranges are retained as
line-and-column spans in the relationship provenance references.
Symbol selections use `selectionRange` (or `targetSelectionRange` for links),
and file URIs are encoded/decoded before checking the host repository boundary.
External locations and malformed entries are excluded with incomplete coverage.
The shapes follow the [LSP call hierarchy contract](https://github.com/microsoft/language-server-protocol/blob/gh-pages/_specifications/lsp/3.17/language/callHierarchy.md).

All positions are one-based. Semantic results declare `column_encoding: utf-16`;
text fallback declares `column_encoding: utf-8-bytes`. An explicit semantic
column is honored; an ambiguous repeated symbol requires a column. The source
text used to locate the symbol is also the text supplied to clangd. Reopening a
changed file sends `didChange`; compound open/query operations are serialized.
At most eight source documents of up to 4 MiB each are retained by the session.
Failed sessions are closed and may be initialized again on a later request.

Coverage remains separate from truncation: excluded or invalid entries can make
`complete` false while `truncated` remains false. The adapter preserves provider
limitations. `code.calls.count` counts projected unique directed relationships;
`provider_count` records the number returned by the provider. The result limit
applies across both call directions. Legacy diagnostics callbacks remain bound
independently from canonical code callbacks.

## Workflows

The starter bootstrap package defines:

- `workflow://code/overview` for bounded repository orientation and capability
  reporting.
- `workflow://code/entry-point-investigation` for selecting a definition and
  inspecting its immediate semantic call neighborhood.

Their blueprints are `code-overview-v1` and
`code-entry-point-investigation-v1`. Both workflows preserve source locations,
provider limitations and the distinction between text matches and semantic
relationships. `code.impact` is optional and is used only when `code.inspect`
reports bounded semantic call support. The package parser rejects duplicate JSON
object keys so that required procedure lists and other contracts cannot be
silently replaced during import.

## Source-bound development evidence

`repository.snapshot` captures the current Git revision and a bounded manifest of tracked
and non-ignored untracked files. The manifest points to immutable scoped resources for
each file's exact bytes and mode. A complete `code-source-snapshot.v1` can be supplied to
`workspace.patch`, `development.build` or `development.test` through
`source_snapshot_ref`. The host verifies that a patched file still matches the selected
baseline. After the final patch, capture a new snapshot and supply that post-change
snapshot to build and test operations; the pre-change snapshot only identifies the
baseline and cannot demonstrate regression safety of the patch. The host verifies and
materializes the selected snapshot under `/workspace/source` before invoking the target. Incomplete snapshots are retained for
inspection but rejected for mutation binding or materialization. Successful patches
retain an immutable `workspace-change-receipt.v1` with before/after hashes and applied
operations.

With host resource storage configured, development runs retain a
`development-execution-receipt.v1` resource and bounded stdout/stderr resources.
Existing profiles without resource storage can still execute without inputs, but cannot
provide retained verification evidence or consume source snapshots. The receipt binds the logical target, selected
configuration/filter, terminal status, runner identity, usage and source manifest hash
when a snapshot was supplied. This proves which source bytes were staged, not that every
acceptance criterion passed, and does not by itself pin external dependencies, generated
inputs, build images or compiler/toolchain state. `code.impact` also remains a bounded
semantic call-neighborhood projection rather than complete change-impact analysis.

Repository mutation and sandbox materialization share the typed source-snapshot validator
under `common/agent/sandbox/source-snapshot.h`. It validates JSON types, paths, file counts,
sizes and identities before host adapters perform resource reads or filesystem effects.
Execution artifacts remain available on both successful and failed results. Output storage
failures are recorded explicitly in the receipt and never reported as complete evidence.

## Presentation and current limits

`code.impact` returns `continuations` with a reason and an `arguments` object
that can be passed directly to another `code.impact` call for a bounded or
failed branch. If a provider fails after relations
have already been collected, the response keeps the graph, marks it incomplete,
and reports bounded safe `provider_failures`; a failure on the initial request
still returns the ordinary tool error. Continuations do not guarantee that the
remaining graph is complete or that a failed provider will succeed on retry.

`code.calls` and `code.impact` include a `relationship` visualization projection in their results.
The runtime validates visualization projections against the shared contract
and forwards only that projection in the completed tool event; it does not
expose the full tool result in the event stream. The web client renders it with
the domain-neutral `RelationshipGraph` component in the assistant turn. Research
uses the same component through a thin semantic wrapper. Completed graph events
are scoped to the current session and deduplicated by delivery sequence. This is a
transient presentation: it is not yet a persisted code-analysis artifact and
cannot be reopened independently of the chat turn. Source locations refer to
the host's current repository state; results do not yet pin a commit or content
snapshot, so they should not be treated as revision-stable evidence.

The current provider offers symbol navigation, references and one-hop call
hierarchy; `code.impact` composes bounded repeated call-hierarchy requests and
does not add a deeper semantic index or analysis engine. It does not yet provide
AST/CFG/data-flow analysis, external-effect classification, a persistent code
index, binary analysis, symbolic execution or dynamic sandbox evidence. Those require
separate host-bound adapters and evidence contracts; they must not be inferred
from a call graph.

## Review verification status

Deterministic contracts cover raw LSP call shapes, encoded paths, symbol positions,
call-site ownership and repository boundaries. Adapter contracts cover canonical
seed handling, repeated call-site provenance, self-calls, node and depth limits,
malformed names, projection coverage, response bounds and preserved provider
failures. Bootstrap contracts verify the code overview's complete procedure list
and installed workflows, and reject duplicate package keys. Frontend contracts
cover shared rendering, session isolation and duplicate event delivery. Run the
focused contracts with CTest (`test-tool-adapters`, `test-agent-bootstrap`,
`test-agent-code-projection`) and the web graph tests with
`npm test -- --run src/components/code-analysis-graph.test.ts` from the web client.
These contracts do not establish model or browser smoke coverage.

`repository.diff` compares tracked changes with HEAD. An optional repository-relative
`path` limits the result, and `include_untracked: true` adds bounded previews of
new files. `truncated` and `omitted_files` report incomplete output; binary and
non-UTF-8 untracked files are identified without exposing their bytes.

### Repository observations

`code.inspect` and canonical `code.symbols`, `code.references`, `code.calls` and
`code.impact` responses include a `source` observation before the query and a
`source_after` observation afterwards. Available Git observations contain the
canonical local repository ID (`id`, the Git common directory), checkout root,
HEAD (`revision`), branch, requested location, local-change flag and Unix
observation time. Failure to inspect Git produces `status: unknown` and does
not prevent an otherwise authorized analysis. `revision_unchanged_during_query`
compares HEAD before and after; it does not promise an atomic content snapshot.
`source_files` adds bounded observations for returned paths, with an explicit
truncation flag. Dirty worktrees are identified rather than hashed.

The response also retains the provider's available status: compilation-database
configuration and stale/sync diagnostics, file-watch generations and last
observed index activity. A database source count, delivered file event or ended
progress notification never establishes complete semantic index coverage.
Use these observations to qualify results and to populate graph `sources`;
verify source contents when stronger evidence is required.

### Optional reading context

`repository.search` and `workspace.search` accept `context_lines` from zero to
five (default zero). Matches retain `preview` and add `preview_truncated`.
With nonzero context, a `context` object contains numbered lines around the
match, `start_line`, `end_line` and an explicit truncation flag. Preview and
context line text is capped at 256 UTF-8 bytes without splitting a character.
Context uses the existing search result budget; narrow the query or reduce
context when the response reports partial coverage.

`code.symbols` retains provider-reported `container_name` and `detail` when
available. A file hint allows the provider to reuse that file's document AST
for namespace/class context and detail. `detail` is the provider's declaration
or type label; it is not a reconstructed source declaration. Missing metadata
is omitted. Text matches do not acquire semantic metadata.

`code.references` accepts `include_context` (default false). When enabled, each
reference includes up to two source lines before and after its position and
`context_status`. The semantic provider also queries document symbols once per
returned file to identify an enclosing function, method or constructor by its
reported source range. `enclosing_function` is an object or null;
`enclosing_function_status` distinguishes `found`, `not_reported` and
`unavailable`. Text fallback uses `unavailable_text_fallback` and never guesses
functions from braces or names. Supplementary analysis is bounded to 16 files;
source reads are capped at 512 KiB per file and the excerpt cache retains at
most 4 MiB of line text. Missing files or exhausted context budgets do not remove
references. Large enriched responses drop optional context from the end and
mark it `omitted_output_budget`, preserving the reference locations. Base
results can still require a lower `max_results` under existing output limits.
These additions do not improve or establish project-wide index coverage.

The extended reference contract is opt-in through `include_context: true`; the
ordinary result remains compact. The clangd client advertises hierarchical
`DocumentSymbol` support. If a provider nevertheless returns flat
`SymbolInformation`, the reported function range is used and its projected
position is marked `position_kind: range_start`; it is not a name position.

## Change impact brief

`code.change_impact` is a read-only member of `developer-read` and `code-analysis`, available through the same native registry and MCP projection as other code tools.

```json
{"path":"src/example.cpp","symbol":"execute_plan","line":42,"column":1,"intent":"Modify execution behavior","max_depth":2}
```

`path` and `symbol` are required. Locations use one-based lines and UTF-16 columns. Without an explicit line, the brief proceeds with relationship analysis only when a complete semantic symbol result identifies exactly one matching definition in the requested file. Text fallback does not resolve identity. Intent is returned unchanged for model interpretation and never selects host behavior.

The host composes existing read adapters using the same repository root, scope and providers. It collects symbol definitions, references, bounded call relationships, a target excerpt, graph node/edge candidates and bounded symbol text matches under `tests`. Each retained observation records its tool and arguments. Graph proposals and text test candidates are observations, not verified contracts, test coverage or safety conclusions. `contracts` remains empty when no authoritative contract extraction is available.

The brief limits traversal to 16 nodes and 24 edges, references to 16, graph nodes to two and outgoing graph edges to four per node. At most 12 subqueries are attempted and 48,000 bytes of observations retained. The shared deadline prevents starting further subqueries; an active provider call retains its own timeout and cannot be forcibly cancelled by this composition. Provider errors and budget omissions appear in `knowledge_gaps`. Partial results never assert complete project coverage. Tests elsewhere than `tests` are an explicit coverage gap.

The target file is read before and after collection using the repository adapter's exact-byte SHA-256, together with Git observations. `target_content_unchanged_during_query` requires matching hashes and observed Git revisions. This detects endpoint changes but does not establish an atomic snapshot or bind a semantic index to those bytes; `coverage.revision_bound` remains false. Retained child observations preserve their own source metadata. Do not interpret stable HEAD or a dirty-worktree flag as proof of stable contents of all related files.

The model interprets the returned evidence and identifies invariants. The tool does not execute tests, modify files, promote graph proposals or declare a change safe. Existing development and comprehension-refresh workflows remain separate explicit actions.

Change-impact traversal collects callers before callees with separate per-direction budgets (12 nodes and 16 edges each). This prevents outgoing library calls from consuming the caller budget; it does not guarantee complete incoming coverage. Each affected symbol records its discovery direction and evidence reference. The top-level `sections` array exposes child completeness, truncation, limitations and continuation arguments. Partial child coverage is also listed in knowledge gaps. Graph discovery searches the file basename and retains only exact repository-path matches before expanding saved relations; broad rejected candidates remain visible only in raw discovery evidence. Test candidates combine bounded symbol text matches with semantically resolved references located under `tests`, with their evidence references. Neither source establishes behavioral test coverage.

## Development evidence tools

`development.test_select` is a read-only selection helper. Supply `changed_files` as up to 32 normalized repository-relative paths. It returns only host-approved targets with declared `source_paths` matching a changed file or its directory prefix. Returned filters are registered aliases, not generated commands or regular expressions. It never runs tests or grants execution authority. Missing mappings produce `unmapped_files`; semantic dependency-based selection and historical result ranking are not implemented in this first version.

Host configuration can add optional source mappings to existing `sandbox.development_targets` entries:

```json
{"agent-contracts":{"build_program":"cmake","build_arguments":["--build","build"],"test_program":"ctest","test_arguments":["--test-dir","build"],"source_paths":["common/agent","tools/agent"],"test_filters":{"contracts":["-R","contract"]}}}
```

This snippet illustrates the target entry only. Ordinary host sandbox policy and test execution authorization still apply. Existing configurations without `source_paths` remain valid and supply no mappings.

`diagnostics.failure_summary` accepts `receipt_ref` and optional `previous_receipt_ref`. References must resolve under the current resource authority to retained `development-execution-receipt.v1` resources with development receipt metadata/lineage. It reads at most two retained output resources of 64 KiB each per receipt and reuses compiler and failure-group parsers. It compares normalized retained diagnostics, source snapshots, target, configuration and filter. A missing snapshot is not a matching context. Output size is bounded; oversized inputs or summaries fail explicitly. Missing diagnostics are not proof of repair, and stream order does not identify a root cause. Hypothesis synthesis remains model-owned. Crash artifacts are not executed or reanalyzed here; native crash analysis remains a separate tool.

`code.contracts` uses the bounded change-impact evidence collector. It adds provider signature candidates and text occurrences of `assert` or `requires` in the retained target excerpt. These are explicitly source-expression candidates: function scope, macro expansion and active compilation are not established. It retains test and graph candidates for model inspection. It is not an AST contract verifier, does not enumerate an entire function body, and does not claim to extract all language constraints or documented preconditions. Text fallback symbols are labeled separately from semantic signature candidates. Source hashes and coverage gaps have the same limitations as `code.change_impact`.

All three tools are included in the `developer-read` and `code-analysis` profiles. Hosts with explicit custom capability lists must opt them in. The native registry supplies the same implementation to MCP; no new execution authority or workflow engine is added.
