# Agent workflows

This page describes runtime workflows (not GitHub Actions workflows).

## Contracts

- A **blueprint** describes a task goal, constraints, assumptions and success
  criteria. It may be useful without a bound executable workflow.
- A **workflow** describes a host-approved method and its execution authority.
  Definitions are stored as plan-store records and referenced by immutable
  workflow identity and revision.
- A **procedure** is reusable methodology or guidance; it does not itself grant
  tool access or define an executable path.
- A **workflow graph** is a bounded host-owned transition system. It may name
  only route-resolved tools, terminal states and dataflow from recorded host
  observations. It is not a new domain-specific planner.

Workflow definitions declare semantic `required_capabilities`, optional
`optional_capabilities`, and host-state `required_context`. Required
capabilities gate route eligibility. Optional capabilities are included only
when the host can resolve them in the active tool view; those resolved tools
are part of the route envelope and its fingerprint from route selection.
Context requirements are checked separately and do not grant tool authority.

`allowed_tools` remains accepted for v1 package compatibility. When semantic
capabilities are also present, `allowed_tools` can only narrow the resolved
tool set. New workflow packages should prefer capabilities.

## Bounded workflow graphs

A workflow without `transitions` retains the v1 one-action behavior. A graph
adds `start_state`, `terminal_states`, and transitions with an `id`, `from`,
`to`, `cost`, and either a `tool` or `kind: "model_choice"`.

```json
{
  "start_state": "author_candidates",
  "terminal_states": ["works_retrieved"],
  "transitions": [
    {"id":"choose-author","from":"author_candidates","to":"author_selected",
     "kind":"model_choice","candidates":"/results"},
    {"id":"retrieve-author-works","from":"author_selected","to":"works_retrieved",
     "tool":"openalex.listWorks","arguments":"{\"filter\":\"author.id:$choice\"}"}
  ]
}
```

The host validates the graph at import: IDs and JSON templates must be valid,
terminal states have no outgoing transition, and every declared state must
have a path to a terminal state. Route compilation then rejects any graph tool
outside the resolved execution envelope.

At runtime, bounded Dijkstra search (A* with a zero admissible heuristic)
chooses the least-cost host-declared path. A `model_choice` is the only model
decision inside that path: it receives a bounded list of IDs from the previous
host observation and must return one of them. The selected ID is recorded on
the next tool step and may replace `$choice`; `$from_step: "$previous"` binds
the preceding completed host step. The model cannot invent an identifier,
tool, URL, or transition.

## Current implementation boundary

The route compiler evaluates workflow capabilities and host-context
requirements generically. The starter library covers these host-tool domains:

- Dataset analysis and quality, including `data.query`, `data.filter`,
  `data.aggregate`, `data.transform`, and the descriptive/statistics tools.
- Multi-dataset joins through a separate route that is eligible only when the
  host reports at least two scoped datasets. Join inputs must resolve to two
  distinct host dataset identities.
- Repository inspection, with read-only search/read capabilities.
- Repository change and regression repair, with workspace patching plus
  sandbox-backed build/test capabilities.
- Resource/document analysis, including host-resolved resource reads and
  structured document tables.
- Web research, gated by the host's network policy and limited to the safe
  search/fetch tools.
- OpenAPI onboarding, which fetches a JSON specification into a host resource,
  admits anonymous read-only operations, and exposes those operations on the
  next turn. The route's optional `openapi.read` capability resolves against
  the newly registered provider so the same workflow can continue safely.
- Configured scholarly research through OpenAlex. A static provider is exposed
  at host startup; its workflows therefore require `context.openapi.available`
  and do not repeat dynamic API onboarding.

The starter blueprint, procedure and workflow for this path are imported from
[`agent-bootstrap-workflows-v1.json`](../examples/agent-bootstrap-workflows-v1.json).
The blueprint binds `workflow://openapi/resource-onboarding`; the workflow
requires `web.fetch` and `openapi.connect`, and adds available read-only API
operations through an optional capability after admission. The model receives
the fetched resource handle from `web.fetch`; the host owns operation discovery,
policy filtering and the next-turn tool view.

OpenAPI HTTP failures retain retryability and a status-specific failure class:
invalid request/input (400/405/415/422), host credentials or access policy
(401/403), missing resource (404/410), timeout/conflict/size-limit, and
transient service/rate-limit failures (408/425/429/5xx). Retry guidance is
host-derived from `Retry-After` (delta seconds or HTTP date), standard
`RateLimit`/`RateLimit-Reset`, and common `X-RateLimit-*` / `X-Rate-Limit-*`
headers. The runtime reports the observed delay and headers; it does not
automatically replay a request or let the model choose a retry time.

An imported package without a `project_id` is host-wide within its namespace:
its procedures use global memory scope, and its blueprints/workflows use global
template scope. Supplying a `project_id` installs the package in that project
instead. Global template visibility does not make task plans global; instantiated
plans still use the caller's requested turn/session/project scope. The plan
scope matcher for active task plans remains exact.

## OpenAlex starter library

[`agent-host-config-openalex.json`](../examples/agent-host-config-openalex.json)
is a least-privilege static provider example: it loads the generated OpenAlex
spec and provider fragment at host startup and exposes only list/get operations
for works, authors, institutions and topics. This is distinct from `workflow://openapi/resource-onboarding`,
which remains the path for an API supplied during a conversation.

For a host that wants both the static provider and the starter library, use
the host config together with the package import. With no project identity the
package import is host-wide in its namespace; the OpenAlex provider itself is
also host configuration, not a session registration.

```sh
llama-agent --config docs/examples/agent-host-config-openalex.json \
  --agent-import docs/examples/agent-bootstrap-workflows-v1.json \
  --agent-runtime "Find recent work on retrieval-augmented generation"
```

The starter package supplies four configured-OpenAlex workflows and matching
blueprints:

- `workflow://openalex/work-search` for bounded scholarly-work discovery.
- `workflow://openalex/work-lookup` for an OpenAlex ID or DOI-backed metadata
  check.
- `workflow://openalex/literature-review` for a transparent candidate set that
  remains available as a host resource and can optionally be exported.
- `workflow://openalex/entity-filtered-work-search` for title/text discovery
  and bounded works search by a resolved author, institution or research topic,
  optionally combined with publication-year filters.

The entity workflow resolves a human name through the corresponding OpenAlex
collection and uses only an ID returned by that operation. For research areas,
it searches the Topics collection, then filters works with `topics.id:T…`.
Work records expose up to three assigned `topics` plus `primary_topic`; use
`primary_topic.id` only when the request specifically asks about the top-ranked
topic. The works `search` parameter searches title, abstract and other text,
and `display_name` is the returned work title. The host bounds each request and
the workflow reports that the resulting candidate set is not exhaustive.

They share the OpenAPI read capability but carry different procedures,
constraints and plan steps. In particular, search ranking is not treated as a
quality judgement, and a bounded result page is never presented as an
exhaustive literature review. The route compiler admits them only when the
host has already registered an OpenAPI read tool; a host without OpenAlex sees
neither route as executable.

The live smoke has three independent modes: resource acquisition,
conversation-time OpenAPI admission, and automatic configured-provider
startup. The last proves that a host configuration loads the OpenAlex contract
and exposes `openalex.*` tools before a model turn:

```sh
./build-agent-vulkan-cozo/bin/llama-agent-web-openapi-live-smoke \
  --mode configured \
  --config docs/examples/agent-host-config-openalex.json
```

Dataset analysis/quality/join, resource/document analysis and web research
declare `artifact.export` as an optional capability. The route receives it only
when the selected tool profile exposes the host-backed exporter. Export remains
an explicit user-requested action; it is not an automatic terminal step.

The dataset A* adapter accepts every data operation above when explicit
host-bound operation arguments are available. It does not invent predicates,
aggregate measures, transforms, or join keys. If those semantic arguments are
not already bound, materialization yields to the ordinary planner, which can
select among all operations allowed by the route envelope. Join materialization
adds separate host-resolved left/right dataset selections and schema checks.
In this case the selector does not persist the blueprint's generic template as
a task. It defers plan creation to the ordinary planner and carries the selected
route binding into that new plan, while the route-scoped tool view remains the
authority for both model-visible tools and execution. The plan also retains the
selected workflow definition and blueprint ancestry so route revalidation and
workflow continuation remain available after the fallback.

The resource/document workflow keeps table discovery and materialization
host-owned. After successful tool results, its A* continuation may append a
host-bound data operation using a typed reference to the completed
`document.table` dataset output. It never chooses table names or operation
arguments. Without a host-bound operation, ordinary planning continues within
the same route and envelope.

Repository and web workflows currently use the common route compiler and
execution envelope, with ordinary planning for request-specific paths.
Dataset and document continuation remain compatibility bridges while their
request-specific inventory/materialization facts are moved into graph
definitions; the execution envelope and generic graph runner are already
shared with OpenAlex.

## Next adapter seams

When adding another domain, put reusable method structure in the workflow
graph. Add a small domain validator only where the host must establish an
inventory, materialize a typed resource, or verify a domain invariant. The
common route compiler validates capability/context requirements, resolves the
tool view, and produces the envelope; it never gains a domain branch. Tool
execution remains host-validated regardless of how a candidate path was
proposed.
