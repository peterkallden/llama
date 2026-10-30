# Project context graph

The project context graph stores explicitly proposed relationships separately
from memory records, research workspaces and investigation cases. A proposal
is not a verified fact. Nodes contain names and optional source locations;
they do not contain copies of source files. Evidence references and revisions
are recorded as submitted and require independent verification.

The `graph` tool family offers `graph.find`, `graph.get`, `graph.expand`, `graph.source`,
`graph.link.propose` and `graph.retract.propose`. New relationships start as
`proposed`; a withdrawal changes the current status to `retraction_proposed`
while preserving previous versions. Cozo persistence is scoped by the
host-provided namespace and project, and graph operations fail if a project is
not configured. File and symbol paths are repository-relative. Symbol identity
includes its source location and stated revision; ambiguous symbols are not
silently combined across revisions. Actor, claim, event and plan nodes require
an opaque source reference, which distinguishes objects with the same name and
does not grant access to their sources. Returned freshness is `unverified`.

Read-only graph tools are added to the Codex tool-audit profile; the audit
profile still disallows policy-gated writes. The separate `graph-project`
profile allows graph proposals under the existing memory-proposal policy.
`scripts/agent-codex-mcp-graph.sh` starts that restricted profile and requires
`LLAMA_AGENT_GRAPH_PROJECT`. Set `LLAMA_AGENT_GRAPH_CONFIG` to a host config
using the intended Cozo data store; the default audit config uses an isolated
store. The project and namespace must match the agent host to share data.
Reload the MCP connection after changing its profile or tool schema.

The saved graph contains proposals only. `graph.source` delegates an explicit
request to `code.calls`, `visualize.research_graph` or
`investigate.graph.expand` through the same resolved native tool view. The
source tool must itself be authorized and model-visible, receives its normal
arguments, and consumes a separate tool-call slot. Its scope, validation and
result limits still apply. The response labels the source and includes the
source result and, where available, a bounded visualization. It does not
persist or silently merge those observations with proposals. Source revisions
and evidence references supplied in proposals are not verified.

The web chat can search and reopen the current project's saved graph, and shows
graph expansions in the shared relationship renderer. With the corresponding
read tools authorized, selecting a project graph node can
expand that neighborhood or read a bounded excerpt around its repository
location. This uses the currently selected session and project.

When authorized graph reads are available, ordinary planning performs a bounded
host-side search and includes a small projection as untrusted context. It never
turns a proposal into verified evidence or grants new tools. Graph tool results
also travel through the existing plan-observation path. A missing project or
unavailable graph does not prevent ordinary planning.

`llama-agent-context-graph-smoke --no-model` exercises proposals and lookup
through the native tool view. Pass `--model PATH` to exercise the same lookup
with a model-produced tool call; `--n-gpu-layers` selects offloading. The MCP
smoke in `scripts/agent-context-graph-mcp-smoke.py` additionally checks Cozo
persistence, project isolation and the read-only profile. Pass the repository
root as its second argument to exercise the configured clangd provider too.
