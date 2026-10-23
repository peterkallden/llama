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

Text matches are not resolved symbols. Static call relationships describe
provider-reported source structure; they do not establish path feasibility,
runtime behavior or execution.

The host owns the repository root and provider configuration. The current
clangd provider is enabled through the existing
`diagnostics.semantic_backend: clangd` host setting for compatibility with
deployed configurations. It is wired through the shared host tool resolver, so
CLI and daemon tool execution use the same bindings and resolved repository
root, including when that root comes from the host tool context. The provider and its LSP
transport are implemented under `tools/agent/code/`; native compiler and crash
diagnostics remain under `tools/agent/diagnostics/`.

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
relationships.

## Presentation and current limits

`code.calls` includes a `relationship` visualization projection in its result.
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
hierarchy. It does not yet provide AST/CFG/data-flow analysis, external-effect
classification, deeper graph traversal, a persistent code index, binary
analysis, symbolic execution or dynamic sandbox evidence. Those require
separate host-bound adapters and evidence contracts; they must not be inferred
from a call graph.

## Review verification status

The follow-up review added `test-agent-code-projection` to CTest for raw LSP
call shapes, encoded paths, symbol positions, call-site ownership and repository
boundaries. Adapter contracts cover malformed results and incomplete coverage;
frontend contracts cover session isolation and duplicate event delivery. These
follow-up checks were authored but not executed, and no build was performed.
