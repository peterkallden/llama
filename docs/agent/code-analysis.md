# Source Code Analysis

Llama-agent exposes source navigation through the `code` tool family. The
family is separate from `diagnostics`, which covers compiler output, test
failures, formatting, include traces and native crash reports.

## Current operations

| Tool | Behavior |
| --- | --- |
| `code.inspect` | Reports whether the host has an authorized repository and which semantic operations are available. |
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

The stdio MCP server retains its clangd session while refreshing tool policy
for each call. Indexing is asynchronous and the provider cannot establish
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
whether the source is present, or null when this cannot be determined.
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
