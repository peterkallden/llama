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
Workflows may also declare `continuation_workflow_targets`. These are workflow
references the host may consider after a successful tool observation returns
typed dataset references. The host recompiles only declared targets, then
selects among eligible dataset workflows against the original task. It does
not offer `normal-plan` as a competing choice after a workflow explicitly
yielded a materialized dataset for continuation. If no declared target is
eligible, or route selection is inconclusive, the existing route continues.
The host attempts this handoff once per runtime execution. A selected route
transition is host-authorized; the dataset reference remains in plan
observations and is reconstructed into route context when the plan resumes. A
join workflow is eligible only when at least two
distinct datasets are present and the request selects that route.
For a selected graph-less dataset workflow, the host uses the same bounded
workflow-action selector and dataset A* materializer as initial planning. It
adds those verified steps to the existing plan, preserves completed source
steps as history, and makes the existing final-answer step wait for the new
dataset path.

Host-provided read-only memory tools (`memory.search`, `memory.get`,
`memory.inspect`, and `memory.conflict_check`) remain ambient in a scoped route
when they are present in the host's tool catalog. They are not workflow write
authority; memory mutation tools remain subject to the route and memory policy.
This tool availability is separate from memory hits already attached to the
request, which research treats as materialized context rather than work that
must be reacquired.

Conversation compaction does not replace the workflow plan or research
workspace. The session host persists turn/checkpoint revisions via scoped
resource records; plans continue to own workflow state, and research continues
to own claims, sources, and chunk coverage. Context assembly carries stable
references and bounded summaries across turns. A summary never promotes model
text into a user instruction or makes an incomplete evidence set appear fully
reviewed.

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

Model choices may instead declare a bounded, host-owned `choices` array. Each
choice has an `id`, a short description, a destination state, and optional
`requires` JSON pointers that must exist in the preceding observation. Tool
transitions may bind scalar strings from the current or a named completed
transition observation into argument JSON pointers. A missing required source
fails closed; it never becomes a model-invented value. Bindings can name the
source transition with `from` and mark an unavailable value optional with
`required: false`.

The OpenAlex full-text evidence workflow uses this seam after `getWork`: the
model chooses between answering from the abstract and fetching full text. The
fetch option is offered only when the returned work record contains an
open-access PDF URL. Fetch, inspect and read are then host-materialized steps;
the model does not repeat or construct the URL. `resource.read(text)` tries the
PDF text layer first; where the host explicitly configures both the page-image
renderer and Tesseract plus a nonzero page cap, it falls back to sequential OCR
only when direct text is unavailable. Partial page coverage is reported as a
limitation on the derived resource; if no text is recovered, the read remains
unavailable rather than fabricating evidence. The workflow opts into
`resource_read_policy: chunked`: after a successful text read whose derived
resource exceeds the chunk budget, the runtime yields before final synthesis.
The host reads each byte range and gives its actual text to a separate,
tool-free extraction generation with a bounded token budget. The model returns
flat `status: no_findings` or one exact, task-relevant `finding` quote (up to 20
words). The host verifies that quote against the active slice and stores it as
both the finding and its evidence; later synthesis makes the interpretation.
The host parser enforces the DSL and word limit; only a verified quote can be
recorded or advance the cursor. Generic process chatter, invalid DSL, and a
quote absent from the active slice do not count as processed. Final synthesis uses
the ordered validated findings and source provenance instead of reattaching
the complete text. The typed resource reference comes from the read adapter's
actual text descriptor (including PDF/OCR-derived text), not by parsing the
tool's JSON output. Small resources retain the ordinary single-read path.

If reflection subsequently requests research, the runtime can seed its
research workspace with those validated finding/quote pairs and their source
byte ranges. Before any new acquisition is scheduled, the bounded research
assessor evaluates this materialized evidence against the open gaps and the
normal source-count/comparison requirements. Unaccepted findings leave the gap
open for research; ambiguous multi-gap findings are not assigned automatically.

Chunking applicability is intentionally about text consumption, not every
workflow that happens to produce a resource:

- OpenAlex full-text evidence and resource/document analysis explicitly opt in
  to `chunked` text reads. Binary inputs such as PDF or DOCX first pass through
  their configured text processor; the resulting text descriptor is chunked.
- Web research optionally resolves `resource.read`; if the workflow reads a
  fetched resource and is configured for chunked reads, the same host-read
  extraction path applies. A bounded `web.fetch` excerpt alone does not start
  chunk continuation.
- Dataset analysis, OpenAlex record/search workflows, and OpenAPI onboarding
  remain structured: use dataset operations, bounded API projections/pages,
  and the OpenAPI catalog builder rather than splitting their JSON or tabular
  contracts into text chunks.

Workflow definitions default to `resource_read_policy: single_range`. Only a
workflow that explicitly declares `chunked` can trigger full-text continuation
from `resource.read`; each plan is limited to one chunked parent resource. If a
second distinct resource is requested for chunking, the runtime fails with a
clear new-plan requirement rather than mixing evidence or checkpoint lineage.

Workflow-produced text resources are handed to the same host-owned chunk
controller as attached text resources. Multiple chunk parents within one plan
remain unsupported because checkpoint/provenance currently tracks one parent
at a time.

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
- Eurostat statistics through a bounded chain: catalogue TOC XML discovery,
  SDMX dataflow/DSD/code-list/constraint resolution, then Statistics API
  retrieval. The JSON-stat response becomes an ordinary provenance-bearing
  dataset and continues through the generic dataset analysis/quality workflows.
  These starter procedures and blueprints are imported from the JSON package;
  the host supplies generic safe XML/JSON normalization and query serialization.

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

The starter package supplies five configured-OpenAlex workflows and matching
blueprints:

- `workflow://openalex/work-search` for bounded scholarly-work discovery.
- `workflow://openalex/work-lookup` for an OpenAlex ID or DOI-backed metadata
  check.
- `workflow://openalex/literature-review` for a transparent candidate set that
  remains available as a host resource and can optionally be exported.
- `workflow://openalex/entity-filtered-work-search` for title/text discovery
  and bounded works search by a resolved author, institution or research topic,
  optionally combined with publication-year filters.
- `workflow://openalex/full-text-evidence` for a known work where metadata and
  abstract are interpreted first, followed only when needed by a fetch of an
  OpenAlex-returned open-access location into a host resource.

The entity workflow resolves a human name through the corresponding OpenAlex
collection and uses only an ID returned by that operation. For research areas,
it searches the Topics collection, then filters works with `topics.id:T…`.
Work records expose up to three assigned `topics` plus `primary_topic`; use
`primary_topic.id` only when the request specifically asks about the top-ranked
topic. The works `search` parameter searches title, abstract and other text,
and `display_name` is the returned work title. The host bounds each request and
the workflow reports that the resulting candidate set is not exhaustive.
The `max_pages` transition directive is generic: any imported workflow can
enable bounded cursor collection on a read-only OpenAPI tool transition. The
OpenAPI catalog supplies the declared item/cursor response paths, optionally
through the operation's `x-agent-pagination` metadata. No OpenAlex-specific
workflow name or blueprint binding is required. Other pagination methods are
reported as unsupported for automatic collection and remain single-request.

The broad work-search and literature-review workflows may follow the OpenAlex
cursor for at most five pages. This host-only limit is attached to those
workflow transitions; direct `openalex.listWorks` calls and narrower lookups
remain single-request. The host uses the configured page size, records page and
HTTP provenance, materializes the combined raw JSON as a resource, and sends it
through the existing chunked-read path. A page cap, missing/repeated cursor, or
later-page failure is marked `complete: false` with a stop reason. No request is
retried automatically, and a bounded result is never described as exhaustive.

The full-text workflow is deliberately layered. `openalex.getWork` supplies
identity, bibliographic metadata and declared open-access locations. The host
projects OpenAlex's inverted abstract index into bounded `abstract_text`; the
stored provider response remains unchanged. A bounded host choice lets the model decide
whether the abstract answers the request or whether the question requires
methods, results, conclusions or limitations. When a returned record exposes an
open-access file, the host projects `full_text_url`, preferring
`best_oa_location.pdf_url` and falling back to `content_url`. It binds that
exact returned value to `web.fetch`; the model does not repeat or invent a URL,
and a `landing_page_url` is not treated as the content URL. Fetch, inspect and
read are then workflow-materialized steps.
`web.fetch` stores the authoritative fetched body as a typed host resource,
rather than a text-preview wrapper. Text/HTML/XML can be read directly; a PDF
can use the existing host text-representation processor through
`resource.read`. A truncated body or unavailable text representation remains a
reported limitation, never evidence of an unobserved section.

The model-backed continuation smoke can be run in either runtime mode:

```sh
LLAMA_AGENT_THINKING_MODE=deliberate ./scripts/test-agent-openalex-fulltext-model-smoke.sh
LLAMA_AGENT_THINKING_MODE=research ./scripts/test-agent-openalex-fulltext-model-smoke.sh
```

Both keep full agent/generation traces. The research run additionally verifies
that resource-backed observations from the selected workflow are seeded into
the existing research workspace and become recorded evidence. It requires an
extractable source representation to meet research evidence coverage; if a
fetched PDF has no bounded text representation, the run must fail closed rather
than report research evidence. With OCR configured, its bounded derived text
can enter the same evidence path; capped or failed OCR remains explicitly
partial and must not be described as complete coverage. The model smoke
configures the bounded local PDF/OCR processors so it exercises the production
`resource.read` chain rather than manually invoking renderer and OCR as separate
smoke-only steps.

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
