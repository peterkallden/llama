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
CLI and daemon tool execution use the same bindings. The provider and its LSP
transport are implemented under `tools/agent/code/`; native compiler and crash
diagnostics remain under `tools/agent/diagnostics/`.

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
the existing relationship graph component in the assistant turn. This is a
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
