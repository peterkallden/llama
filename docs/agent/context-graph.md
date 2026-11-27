# Project context graph

The project context graph stores explicitly proposed relationships separately
from memory records, research workspaces and investigation cases. A proposal
is not a verified fact. Nodes contain names and optional source locations;
they do not contain copies of source files. Evidence references and revisions
are recorded as submitted and require independent verification.

The `graph` tool family offers `graph.find`, `graph.get`, `graph.expand`, `graph.search`, `graph.source`,
`graph.link.propose` and `graph.retract.propose`. New relationships start as
`proposed`; a withdrawal changes the current status to `retraction_proposed`
while preserving previous versions. New retraction proposals keep their
reason in `retraction_rationale`, separately from the original proposal's
`rationale`. Older saved retractions may only have the earlier combined field;
their previous versions remain in the storage history. Cozo persistence is scoped by the
host-provided namespace and project, and graph operations fail if a project is
not configured. File and symbol paths are repository-relative. A symbol may
include a provider-qualified `symbol_key`: observations with the same key in
the same project share a node and retain their individual locations and stated
revisions. A key is a caller-supplied identifier, not independently verified
evidence; it must also distinguish repositories within a project. Without a
key, symbol identity includes its source location and stated revision, so
ambiguous symbols are not silently combined across revisions. Existing node
IDs remain unchanged. Tool responses show the most recently submitted
observation as the location and include up to eight recent observations, with
an explicit truncation flag. Submission order does not establish which
revision is newest. To read the full bounded observation list, call `graph.get`
with `observation_offset: 0` and follow `observation_page.next_offset` until it
is null. Actor, claim, event and plan nodes require
an opaque source reference, which distinguishes objects with the same name and
does not grant access to their sources. Legacy references have unknown freshness. Structured sources can carry per-source currentness as described below.

Planning context prioritizes proposed relationships before relationships with
a pending retraction. A pending retraction remains visible and is not an
approved withdrawal.

Relation names are strings, up to 64 bytes. Suggested names include `calls`,
`depends_on`, `related_to`, `contains`, `supports` and `references`; arbitrary names
are accepted. Names are compared exactly: `authored` and `authored_by` are not
automatically treated as synonyms or inverses.

New relations use a stable identity derived from the project scope, endpoints
and exact relation name. Repeating a proposal with different evidence or a
different rationale adds an unverified model assertion to that relation;
repeating identical evidence reuses the existing assertion. A pending
retraction prevents adding further assertions until it has been resolved.
Each relation retains up to 64 assertions; the host reports an error if the
limit is reached.
The response includes a bounded assertion list and an explicit truncation flag.
Use `graph.get` with `assertion_offset: 0` and follow `assertion_page.next_offset`
to inspect all saved assertions for a relation.
Older test graphs are not migrated to the new relation identity.

Use `graph.find` to resolve names to project graph IDs. Then `graph.search`
accepts any combination of `from`, `to`, `relation`, `from_kind`, `to_kind`
and `status` (at least one filter is required); `limit` defaults to 32 and is
at most 100. Both proposed and retraction-proposed edges are included unless a
status is specified. Results contain the edges and their endpoints, and mark
truncation. For example, `{ "from": "<alice-id>", "relation": "authored" }`
finds Alice's saved authorship proposals. Use the paper ID as `to` to search
in the other direction, or `graph.expand` to browse its other relationships.
`graph.find` also accepts a `limit` from 1 to 32 (default 32); its `truncated`
flag indicates that further matching nodes exist. The graph investigation
workflow requests five, offers those returned nodes or an option to finish,
and binds the selected node ID from the host observation before expansion.
`graph.describe.propose` adds a sourced description to an existing node using
the graph revision from a read as `expected_revision`. Descriptions are model
proposals, not verified facts. The node ID does not change; prior descriptions
are retained. The latest proposal participates in `graph.find` and bounded
planning projection. Use `graph.get` with `description_offset` to page through
description history. Full tool responses include `text_bytes` and
`text_complete: true`: its text is the complete literal stored value, not a
preview. Description pagination selects records, not fragments of one text.
The surrounding tool/context projection may still report omitted coverage.
Existing nodes without descriptions remain valid.
The current adapter still loads the bounded project graph before applying
endpoint-kind and status filters; the indexes narrow candidate edges but do
not yet eliminate that graph load.

Opening an older Cozo database builds the graph search indexes from its saved
edges in an atomic migration. Stop older server processes writing to the same
database before upgrading; older versions do not maintain these indexes. An
incomplete index schema fails closed instead of returning incomplete results.

Graph reads and policy-gated relationship proposals are available in the Codex
tool-audit profile. The separate `graph-project`
profile allows graph proposals under the existing memory-proposal policy.
For a standalone server, `scripts/agent-codex-mcp-graph.sh` accepts the normal
server options `--memory-project PROJECT` and `--memory-namespace NAMESPACE`.
`LLAMA_AGENT_GRAPH_CONFIG` selects the host config; the default audit config
uses an isolated store. Matching project and namespace IDs do not connect
different stores.

To use the running daemon's graph, connect the MCP client to the daemon's
inbound HTTP endpoint. Configure an inbound token profile with the same
`namespace` and `project` as the daemon request, and a tool profile that permits
the desired graph operations. The daemon's tool profile must expose those tools
as well. Calls use the daemon dispatcher and its configured data store. The
token profile is host authority; the client cannot choose another project by
passing tool arguments. Separate MCP client sessions can access the same
project graph when their authorized namespace and project match. See
[inbound MCP HTTP setup](agent-daemon-usage.md#inbound-mcp-http-in-the-daemon-host).
Reload the MCP connection after changing its profile or tool schema.

For example, add a token profile to the daemon's existing `mcp.inbound.tokens`:

```json
{
  "id": "codex-project-reader",
  "token_env": "LLAMA_AGENT_CODEX_TOKEN",
  "audience": "llama-agent",
  "namespace": "<session namespace>",
  "project": "<session project ID>",
  "tool_profile": "developer-read",
  "allow_writes": false
}
```

Replace the placeholders with the values used by the daemon session and
configure the client with the daemon's MCP URL and the same bearer token. This
profile can read graph data; proposing relations requires a separately
authorized profile and the existing write policy. A token identifies a fixed
authorized project, not whichever project a different client last selected.

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

The general starter package includes separate project graph investigation and
maintenance blueprints and workflows. Investigation starts with `graph.find`
and can finish without a match or expand a relevant node. Maintenance searches
first and can finish without writing or propose a description, relation or
retraction through the existing host tool policy. These definitions do not
grant tools or turn proposals into verified facts; availability depends on the
active project scope and authorized tool view.

`llama-agent-context-graph-smoke --no-model` exercises proposals and lookup
through the native tool view. Pass `--model PATH` to exercise the same lookup
with a model-produced tool call; `--n-gpu-layers` selects offloading. The MCP
smoke in `scripts/agent-context-graph-mcp-smoke.py` additionally checks Cozo
persistence, project isolation and the read-only profile. Pass the repository
root as its second argument to exercise the configured clangd provider too.

The daemon scope smoke can be run without inference using
`python3 scripts/agent-daemon-graph-mcp-smoke.py build-agent-vulkan-cozo/bin/llama-agent-daemon`.
It uses a temporary data store and verifies shared project access, read-only
policy, and namespace and project isolation through the daemon dispatcher.

### Standalone descriptions and comprehension refresh

`graph.describe.propose` accepts exactly one of an existing `id` or a `node`
descriptor using the same identity rules as relationship endpoints. The host
resolves the entity and appends its description in the same scoped publication,
protected by `expected_revision`. No edge is created. Existing id-based callers
remain supported. Descriptions remain model proposals and retain the existing
2048-byte bound and evidence references.

The starter `comprehension-refresh-v1` blueprint applies to code, research,
documents and investigations. It inspects at most ten candidates and proposes
at most three compact Purpose/Invariant/Dependencies/Change hazard notes.
Research artifacts remain the detailed evidence workspace; graph notes link to
them. Unknown facts must remain explicit. This blueprint is explicitly selected
through normal planning; post-verification opt-in variants are described below.

### Phased comprehension and opt-in development methods

The description limit is now 2048 bytes in the adapter and Cozo store, with the
same bound in the tool schema. Retrieval retains the existing bounded context
projection; longer stored notes do not increase the prompt allocation.

`execution.tool_limits.v1` is an imported hard method constraint whose JSON
object maps exact tool names to reservation limits. Method progress validation
counts persisted tool steps and new unique steps together, including failed
steps, before accepting another slice. The comprehension starter reserves at
most three description steps and ten graph lookups. These limits do not replace
host authorization or research budgets. They bound tool reservations, not the
number of entities returned by one call; the ten-subject frontier and depth-one
selection remain model instructions.

For opt-in post-verification maintenance select
`repository-change-v5-with-comprehension` or
`regression-repair-v5-with-comprehension`. These configuration variants append
the existing comprehension phases after the development phases. Normal method
review and continuation own ordering, saved progress and partial completion.
No autonomous worker is started. Failed or unavailable verification is reported
as a maintenance blocker. Source and research references remain plan evidence;
this feature creates no second research workspace.

The method-phase contract includes persisted reservation checks. The existing
CLI graph smoke now covers standalone creation, a description over 512 bytes,
idempotent replay and reading an isolated node through a separate tool view.
These source changes have not yet been compiled or executed.

## Source revisions and currentness

Node endpoints, descriptions and relation assertions accept an optional `sources`
array with at most 16 entries. Each entry records `type`, `id`, optional
`location`, `revision`, `observed_at` (Unix seconds), `branch` and
`local_changes`. Existing `source_ref`, `source_revision` and `evidence_refs`
remain supported. Old records retain empty sources and unknown currentness;
no source identity is inferred from their free-form references. Source metadata
is submitted evidence, not proof that the submitter actually read that revision.
Node source references are append-only, including when evidence is attached to
a legacy node. Different assertion or description source metadata creates a distinct record rather
than replacing historical evidence. Keyed symbols retain it per observation.

The resolved native tool view checks sources on `graph.get`, `graph.find`,
`graph.search` and `graph.expand`. Every checked entry has a `currentness`
object. The view requires the corresponding model-visible, read-only tool:
`repository.read`, `document.open`, `dataset.inspect` or `resource.inspect`.
The existing host root and resource/dataset scope checks still apply. A source
reference never grants access. A bare adapter registry cannot perform these
checks and returns the saved sources with unknown freshness.

Checks admit at most 16 distinct references within a two-second admission
window, reuse duplicate checks within the response and report unknown for
unavailable, unsupported or budget-exhausted sources. An admitted provider read
may finish after that window. No network fetch is initiated. Results describe
the check time and do not change saved graph data or proposal status.

For Git, use the `id`, `revision`, `branch`, `location`, `observed_at` and
`local_changes` returned by a code tool's `source` or `source_files` entry.
The ID is the canonical local Git common directory, shared by linked worktrees;
`root` in code observations identifies the particular checkout. A different
local clone has a different ID. Currentness compares against the host-bound
checkout's HEAD, not a remote branch tip. Branch names are context only.
Full commit IDs are required. The response distinguishes `same_revision`,
`newer_source`, `diverged` and `unknown`, with `commits_behind` only when the
observed commit is an ancestor. `file_changed` compares the referenced path
between commits; `local_changes` reports its current worktree/index state.
An observation made with local changes has unknown currentness because its
bytes cannot be reconstructed from the commit alone. A current local change
is reported even when the commit is unchanged. No hash is required.

For `document`, `id` is the host document ID and `revision` its numeric published
revision. The existing authorized document-head lookup provides the current
revision and, when newer, `revisions_behind`. Document sections belong in
`location`. For `resource`, use an authorized immutable resource URI as both
`id` and `revision` when referring to that exact revision. Neither check claims
that no newer independent resource exists.

Dataset descriptors currently expose retrieval time, not a mutable source
revision or modification time. Authorized dataset sources therefore return
`unknown`; retrieval time must not be treated as change time. The shared
comparison helper supports `possibly_stale` when an authoritative modification
time is later than `observed_at`, and `not_modified_since_observation` otherwise.
Providers without comparable revision or modification metadata stay unknown.
Currentness does not establish the truth or completeness of a graph claim.

The functional `llama-agent-context-graph-smoke` supports `--no-model` and
`--model PATH --n-gpu-layers -1`, with `--repository PATH` pointing to a local
checkout. `--trace-file PATH` retains JSONL requests and results for all tool
calls and the optional model generation; without it, those events go to stdout.
Run either mode from the intended environment and retain backend diagnostics
alongside the trace. The smoke exercises sourced graph search through the
native view; it is not a model-quality or project-index coverage benchmark.

Restart graph writers and reconnect MCP clients when upgrading so that both
storage writers and advertised input schemas understand `sources`. Older
writers do not preserve fields they do not understand.

## Project context in development workflows

`context.resolve` exposes the same bounded graph projection used by automatic
project orientation. It adds optional, model-selected source reads through the
resolved native tool view. It is read-only and never starts research, selects a
workflow, updates memory or promotes graph proposals.

```json
{
  "query": "plan recovery",
  "budget_chars": 6000,
  "sources": [
    {"tool": "memory.search", "arguments": {"query": "plan recovery", "limit": 2}},
    {"tool": "repository.read", "arguments": {"path": "common/agent/thinking/deliberation-policy.cpp", "start_line": 130, "end_line": 170}}
  ]
}
```

The query is model-selected focus, not a semantic routing rule. At most four
source reads are admitted. Supported sources are `graph.get`, `graph.search`,
`memory.search`, `memory.get`, `repository.read`, `code.symbols`, `plan.get`,
`resource.inspect` and `resource.read`. Each must also be exposed, read-only and
not policy-gated in the same view. Each delegated read uses its own input
validation, source scope and the shared tool-call budget. A bare adapter registry
can return the graph projection but rejects delegated source requests.

The output has schema `project-context.v1`, graph nodes, optional graph revision,
`sections`, `limitations`, `truncated`, `complete: false` and `persisted: false`.
Sections preserve provider data and call IDs, or report unavailable, failed or
omitted reads. The character budget bounds the serialized result, not tokens;
the existing inference context allocator remains responsible for token limits.
Whole source observations are omitted rather than returning fragments as complete
evidence. A provider call already admitted can finish after the admission timeout.
No atomic source snapshot or complete project coverage is claimed.

Saved descriptions and assertions retain their proposal status. Authorized source
checks add currentness without establishing truth. Automatic startup orientation
retains its small 1800-character graph footprint and does not perform extra source
reads; the resolved tool view checks structured graph sources when explicitly
resolving context. The initial projection includes nodes only; choose `graph.search`
explicitly for relationships. Missing graph results can mean no match, unavailable
data or exhausted space; they do not establish absence. Source payloads remain
untrusted data, including any instructions found in repository files or memories.

The starter workflow procedures use the following ownership boundaries:

| Component | Responsibility |
| --- | --- |
| Workflow | Phase objectives, evidence requirements and handoffs |
| Deliberate | Interpret evidence, compare options and identify material gaps |
| Research | Investigate selected gaps and return source-bound observations |
| Context resolution | Assemble explicitly requested existing observations |
| Outer plan | Retain task evidence and research completion references |
| Graph and memory | Preserve useful knowledge through existing authorized proposals |
| Host | Authorization, scope, validation, budgets and provenance |

Reuse adequate evidence from the same source state before requesting more. Carry
references, limitations and open questions through the existing phase handoff.
Research completion continues through the existing outer-plan observation bridge;
there is no separate context plan, cache or persistence lifecycle. Following an
edit or failure, the model decides which affected sources need another read.
Checking freshness does not delete notes or prove every claim about a changed file
wrong. Durable capture remains an optional, authorized `comprehension-refresh-v1`
proposal after useful evidence has been established. A passing test only supports
its observed behavior, not every explanation offered for it.

### Verification scope

The initial implementation passed these eight focused CTests on 2026-11-23:
`test-tool-adapters`, `test-tool-catalog`, `test-agent-bootstrap`,
`test-agent-mcp-app-bridge`, `test-agent-research-driver`,
`test-agent-context-allocation`, `test-agent-research-contract` and
`test-agent-workflow-recovery`.

The added cases check shared graph projection, read-only behavior, source
visibility, delegated input validation, shared call limits, rejection of recursive
resolution, currentness without truth promotion, bounded output and attachment of
the existing shared procedures to the development methods. The research and
workflow tests exercise existing contracts with controlled fixtures. These checks
do not establish that a real model uses context effectively, or demonstrate a full
model-driven development task with deliberate and research escalation. That
behavioral evaluation remains separate.

### Source-read retention correction

Repository and workspace reads now use the same bounded durable retention path
as resource reads (up to 65536 serialized bytes). Prompt projection owns the
smaller model-visible range. Legacy observations marked `result_truncated` request
a reread from their original start line rather than claiming the retained prefix
is complete; source identity must be checked on reread.

The source projection regression test passed. In the Luna retest, the model-visible
runtime excerpt covered all 116 requested lines (3090-3205) with
`model_content_complete: true`. The overall investigation still exhausted its
continuation budget while remaining in orientation. This validates the corrected
source delivery, not completion of the full development workflow.

### Phase handoff follow-up

Starter v54 defines repository-investigation orientation as identifying the bounded
question, candidate source entry points and open questions. Full source tracing
belongs to the following phase. Method phase reviews retain optional bounded
`next_work_kind` and `next_work_goal` annotations as untrusted planner guidance;
they do not select tools, grant authority or alter dependency eligibility.

The method-phase and bootstrap CTests passed. The Luna v54 run accepted orientation
at its second review and subsequently planned source tracing. The overall run still
exhausted its continuation budget: later plans kept searching without the source
reads needed to establish implementation behavior. This is evidence of phase
progress in one run, not a completed investigation or a general efficiency result.

### Remaining integrated evaluation

The resource-inventory investigation reproduced a scope mismatch: a web turn
requested its test project, but both execution scope and inventory authority used
`__llama_agent_web_catalog__`. Authenticated daemon scope binding overrides the
request with the caller's configured fixed project. The new explicit project
selection policy is intended to correct that binding for authorized web clients;
live verification with the setting enabled is still pending.

Before evaluating model behavior again, verify separate project resources and
catalog records through creation, listing, reading and turn execution. Confirm
that graph, memory and plan identities follow the same selected project. Inventory
presentation also still needs distinct roles for current attachments, prior scoped
resources and internal navigation metadata using trusted provenance.

The proposal to replace intermediate full answer drafts with compact phase work
summaries is analysis only. It must reuse the existing reflection and continuation
path while preserving final-answer verification, tool repair, research escalation
and budget handling. Longer phase-guidance retention passed focused method and
recovery tests, but has not been evaluated in Luna or after a real daemon restart.
The full Luna investigation remains incomplete; source discovery still needs to
lead to source reads and a verified final answer.

### Bounded method work reviews

Phased methods reserve up to two continuation slices per retained phase (at most
16 phases) in addition to the configured execution allowance. Normal method
continuation is counted separately from plan repair; tool-call and plan-capacity
limits still apply. Intermediate ready phases with dependent phases use a compact
host work-review prompt through the existing reflection engine instead of a full
answer draft. That prompt cannot be accepted as a final response. Terminal work
continues through ordinary draft generation and answer verification.

The focused method-phase and workflow-recovery tests passed after this change.
The changed runtime has not yet been deployed to the Luna daemon or evaluated for
end-to-end investigation completion. The two-slices-per-phase allowance is an
initial bounded policy, not an empirically established optimal budget.

### Plan capacity rollover

When a method continuation exhausts durable plan step capacity, the runtime can
create a successor through the existing `prior-work.v2` checkpoint protocol.
The successor retains eligible, current phase evidence and receives the remaining
execution, tool and revision allowances. Storage rollover does not replenish
these task budgets and remains bounded by the driver's continuation limit.
Without remaining budget the driver preserves a checkpoint without a verified
final answer.

`test-agent-workflow-recovery` exercises the actual outer driver with a two-step
plan limit: the first phase fills the plan, a successor executes the second phase,
and ordinary final drafting and reflection accept the response. A separate case
exhausts the tool budget and checks that no final response is accepted.
`test-agent-method-phases` also checks the capacity boundary and carried budget.
These deterministic checks validate runtime integration, not model task quality.

### Runtime verification on 2026-11-27

The CPU-only Luna daemon was rebuilt and restarted with the changes above.
Three focused CTests passed: `test-agent-workflow-recovery`,
`test-agent-method-phases`, and `test-agent-chat-routing`.
The driver test reaches final response acceptance after rollover; a separate
budget case prevents acceptance when further tool work cannot be admitted.

Live web-API runs exposed additional prompt-budget defects. Reasoning allocation
omitted the structured schema and shared adapter framing (cognitive mode and
investigation context). Reflection allocated its evidence before adding validation
feedback on regeneration. These paths now account for complete adapter framing;
reasoning reduces optional context after rendering, and reflection reallocates
its evidence on each attempt while retaining the complete answer under review.
Regression coverage includes growing evidence, injected investigation context,
and reflection regeneration with the configured output reservation intact.

The final focused Luna run lasted 62 seconds. It executed the original plan and
one successor, completing 16 tool operations: eight repository reads, six searches,
and two status queries. It stopped with `continuation tool budget is insufficient
for the proposed slice`, without a final response. The budget was not replenished
at rollover. The earlier broad investigation also rolled over, then stopped after
15 successful tool operations because its next slice exceeded the remaining budget.
Neither run establishes successful model-level investigation completion. The
focused run supplied source entry points and is not equivalent to the original
source-discovery task. These were web API checks, not browser interaction tests.

The remaining observed limitation is model/tool efficiency within the existing
budget. Repeated searches and incomplete tracing still consume the available
allowance. Passing deterministic runtime tests must not be reported as evidence
that Luna has completed the source investigation or produced a verified answer.

### Budget-limited partial answers

When the host cannot admit further tool work or execution/revision iterations,
it can close execution and reserve the configured answer-review allowance for
synthesis over retained evidence. The same draft generator, grounding checks,
reflection engine and evidence-reference validation remain in use. The closing
pass cannot execute tools, reopen research or apply plan repairs. Cancellation,
source authorization and generation context/output limits still apply.

The host sets `partial_allowed` with `stop_reason: execution_budget_exhausted`.
`mandatory_remaining` and pending method phases remain truthful. The answer must
separate supported findings, uncertainties, unfinished work and the budget limit.
The reviewer can accept that answer with `task_outcome: blocked`; this accepts the
partial response, not task completion. Acceptance retains an active saved plan,
reports `limit_reached`, and records an accepted, paused review. It does not mark
pending tool steps or method phases complete. Unsupported claims or missing
material limitations can still cause response revision or rejection within the
reserved review allowance. No accepted response is manufactured after rejection.

Runtimes without reflection retain their existing checkpoint behavior rather
than returning an unreviewed partial answer. A cancelled generation does not
start finalization. The reserve cannot extend an expired turn deadline.

Validation: all four focused CTests passed (`test-agent-workflow-recovery`,
`test-agent-synthesis-state`, `test-agent-method-phases`, `test-agent-chat-routing`).
The recovery test covers accepted and rejected budget-limited answers through the
outer runtime driver, checks the tool-call count, and verifies that unfinished
method work is not marked complete.

A live Luna run of the original investigation through the web API returned
`ok: true`, `turn_completed`, a nonempty partial answer, `limit_reached: true`,
and an accepted, paused review. It lasted 80 seconds and used 14 successful tool
calls (nine searches, three reads, two context resolutions), one final draft and
one successor plan. The next proposed work slice did not fit the remaining tool
allowance, so execution closed before all 16 calls were consumed. The response
reported inspected context-adapter findings and explicitly stated that the full
research-to-plan handoff still needed tracing. This validates delivery of a
reviewed partial answer, not completion of the original investigation. The check
used the web API; browser rendering was not tested.
