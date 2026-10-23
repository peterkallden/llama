# Agent workflows

## Uploaded-resource orientation

An uploaded file without an accompanying question receives a separate,
read-only initial orientation turn. This reuses the preferred document-analysis
blueprint and its host-owned resource resolution and chunking; it does not add
a separate orchestration graph. The orientation describes the resource and
extraction limits, then asks for the user's next goal. It does not edit the
document, search externally, or automatically continue into dataset analysis.
After the user states the goal, ordinary route selection can choose a dataset,
research, editing, or other applicable workflow. An explicit question sent
with an attachment skips the duplicate orientation, keeps the exact question,
and requires the answer to use the attached resource through the normal tools.
The host requires a successful content read of a bound resource; this question
path may use bounded relevant reads. The initial orientation path is stricter:
it requires complete validated coverage, including all host-generated text
chunks, before the initial summary can be accepted.

Coverage is tracked independently for each required source and tied to the
turn's task fingerprint, source-content fingerprint, representation and
parent-linked resource. A partial read of one attachment cannot satisfy a
second attachment. Unresolved chunks, missing or duplicate chunk indexes,
gapped byte ranges and processor failures remain explicit coverage gaps. A
later question can reuse findings from a complete orientation only when the
same authorized source fingerprint is still current; task-specific reads can
still be requested as needed.

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
Legacy `continuation_workflow_targets` remain supported for dataset handoffs.
New cross-workflow handoffs use `continuation_links`, which bind a source
transition and terminal state to a target workflow revision and may carry
JSON-pointer bindings from successful structured observations, including a
named completed transition. A model choice over host-returned candidates is
persisted as a host-validated structured observation after its following tool
action succeeds. Model prose cannot satisfy a binding or create a verified
resource/dataset reference. The host recompiles only declared targets against
current applicability and execution-envelope policy. A unique eligible route
is deterministic; multiple routes use the bounded route selector. If a
required source value is missing, the host records the gap and leaves the
source route active instead of inventing a value. Typed links can carry
structured resource/API metadata even when the source result is not a dataset;
legacy `continuation_workflow_targets` remain restricted to verified dataset
outputs.

Both within-workflow bindings and cross-workflow handoffs resolve named
transitions only inside the active route's workflow-step segment. A successful
step from an earlier workflow with the same transition ID cannot supply a
current binding. Candidate-choice evidence must match the completed step's
recorded choice alias and reference that step's successful tool observation.
Zero/non-finite confidence and explicit failure envelopes are rejected.
Bindings preserve JSON types (including numbers, lists, and objects); required
null values or empty strings are missing values. The target tool schema still
validates the materialized arguments before execution.

A tool transition that asks the model to supply request-dependent arguments may
include a bounded `model_instruction`. The runtime adds this as method guidance
for that transition only; tool identity, schema, bindings, and execution
authority remain host-owned. Use it to keep discovery queries narrow when later
workflow steps will verify geography, periods, units, or other metadata.
The same field may guide a model-choice transition over host-returned candidates;
it only changes the comparison criteria, not the host-bounded candidate set.

The runtime requests a cross-workflow handoff only after replaying completed,
host-recorded transition IDs reaches the link's declared terminal state. A
successful structured result in the middle of a graph is ordinary workflow
evidence: it does not spend continuation budget or interrupt the next graph
transition. The completed-state replay is shared by graph continuation and
route handoff, and starts at the active route binding's workflow-step boundary
so prior workflow revisions cannot advance the new graph.

The host records each completed link and source-observation fingerprint in the
existing plan. The same link cannot replay against identical evidence, while
new verified output may continue within `max_continuations`. This history
survives plan-store resume. Budget exhaustion is traced as an incomplete,
resumable continuation; it never marks the chain complete. Graph-less targets
use the bounded action selector and the dataset A* materializer where
applicable. Graph targets start at their declared host-approved tool
transition, whose stable transition ID is retained in the plan. In both cases
completed source steps remain in plan history and the final-answer step waits
for the appended work. Join workflows remain eligible only when at least two
distinct datasets are present and the request selects that route.

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

## Thinking modes and factual grounding

A workflow's route and execution envelope determine its available tools.
The runtime thinking policy determines whether the current turn uses
`deliberate` or `research`. Workflow JSON can declare the optional
`preferred_thinking_mode` with either value. With `runtime.thinking_mode: "auto"`
the driver considers it after route selection, on resume and at a declared
workflow handoff. Explicitly selected thinking modes, host review availability,
maximum mode and escalation limits take precedence. A preference never lowers
an already higher mode or expands the caller's tool/revision budgets.
Allowing the research phase in an envelope is still separate from selecting it.

For a specified indicator, geography and period, deliberate can carry out
discovery, metadata checks, retrieval and calculation. Research is useful when
the request requires competing definitions, multiple sources or explicit
evidence cross-checks. In research mode mandatory workflow tool steps execute
before the deferred research pass consumes their evidence. Neither mode proves
that a dataset matches the statistical question.

The source implementation now supports declared `verification_checks` at
`before_tool`, `after_tool` and `before_answer`, and an opt-in
`require_declared_claims` draft contract. Required conditions must pass;
missing or unsupported evidence remains `unknown`. Tool success and semantic
verification are recorded separately. Successful API output survives a failed
after-check, but the guarded transition cannot complete or hand off.

Package v24 applies these controls to Eurostat: requested geography/period,
SDMX constraint tuples, returned dimension coverage and observation status.
Final numeric declarations refer to bounded host evidence handles and reuse
research's claims/evidence contracts in both deliberate and research modes.
Metadata membership does not prove that a model correctly understood the user's
intended indicator. Independent corroboration also requires actual source
comparison; multiple responses from one authority are not independent proof.

See [Workflow grounding](agent-workflow-grounding.md) for JSON, draft DSL,
scope/resume rules and the current validation record. The implementation and
direct model-free regression executables have been built and run; the connected
Codex MCP server has verified selected tools and existing materialized data.
That MCP surface does not expose agent workflow invocation or Eurostat API
operations, so it does not establish live route/gate/handoff behavior. A
workflow-level daemon run and Qwen model smoke remain outstanding. Historical
smoke results do not validate these new paths.

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

For candidate arrays, `candidates` points to the host-returned array and the
optional `candidate_id` points inside each row to its stable string identity
(default `/id`). This supports catalogues whose canonical key is named `code`
without copying or rewriting the source rows.

When a start-state tool transition has a complete, static `arguments` object
that validates against the resolved tool schema and has no observation
bindings, the host materializes that exact call directly. The model is not
asked to select the already-fixed tool or echo its arguments. This lets a
workflow pin representation parameters such as `web.fetch`'s `accept` without
making them model guesses. Invalid fixed arguments fail route materialization
closed. Eurostat's catalogue workflow uses this for its English TOC request
(`accept: text/plain`, bounded `max_bytes`); the next transition binds the
returned resource reference into `dataset.inspect`.

A graph-entry transition may also set `model_supplies_arguments: true`. The
workflow still fixes the tool and validates its host-owned seed arguments, but
the bounded action selector may add request-dependent arguments. This is
intended for safe interpretation such as searching a materialized catalogue;
the selected code still has to come from the query's host-returned rows before
the following model-choice transition accepts it.

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

The `resource-document-analysis` starter is revision `v7` and its path/graph
are revisioned accordingly. For an initial upload orientation, the host now
selects this route directly and materializes a read-only final-response plan;
it does not ask the model to classify tool families, choose a workflow route,
or select a resource tool before reading. The host's typed resource binding
and bounded resource reader own acquisition. The model first participates in
bounded chunk extraction and then synthesizes the validated findings.
`resource.read` is the only mandatory capability. Resource metadata,
document-table inspection and dataset inspection are optional enrichments, so
a missing optional processor does not make ordinary text orientation
ineligible. Binary inputs (including DOCX/PDF) are converted by the configured
resource processor before chunk planning; the model receives only the derived
readable representation. Orientation requires complete validated coverage.
If processing or a chunk fails, the runtime reports the limitation and does
not treat metadata inspection as a content read. Selected document reviews
use a host-generated Markdown projection of the selected blocks plus immediate
neighboring blocks, avoiding a full document dump into the prompt.

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

## Source code workflow starters

The general starter package `docs/examples/agent-bootstrap-workflows-v1.json`
(version `v30`) includes two code-analysis workflows:

| Workflow | Blueprint | Intended result |
| --- | --- | --- |
| `workflow://code/overview` | `code-overview-v1` | Bounded repository orientation, capability availability and explicit semantic/text-fallback limits |
| `workflow://code/entry-point-investigation` | `code-entry-point-investigation-v1` | A selected definition, one-hop semantic call relationships, source inspection and remaining uncertainty |

Both use `code-analysis-evidence-v1`. Code navigation uses the canonical
`code.*` family; compiler/test/crash output remains in `diagnostics.*`. The
host resolves the repository and checks required capabilities and context.
Importing definitions does not create a semantic provider: `code.inspect`
reports actual operation availability, and missing call support is a limitation.
The graph presents static relationships rather than executed behavior.

These definitions become available through the existing configured bootstrap
import. Their presence in the repository does not show that a running daemon
has imported them or that a model has selected/executed them. Provider binding,
projection and current verification limits are described in
[Source Code Analysis](code-analysis.md).

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
- Eurostat statistics through a bounded chain: catalogue TOC TSV discovery,
  SDMX dataflow/DSD/code-list/constraint resolution, then Statistics API
  retrieval. The JSON-stat response becomes an ordinary provenance-bearing
  dataset and continues through the generic dataset analysis/quality workflows.
  These starter procedures and blueprints are imported from the JSON package;
  the host supplies generic safe XML/JSON normalization and query serialization.
  Package v24 declares terminal-state links from catalogue discovery to SDMX
  metadata resolution and from inspected metadata to data retrieval. The
  data-retrieval workflow now uses the same typed continuation-link contract
  for its handoff to dataset analysis, quality, or join; it no longer relies
  on the legacy dataset-only target list. These are host-validated route
  transitions, not proof that a model correctly interpreted every statistical
  definition. In particular, `dataset.inspect` materializes SDMX records but
  does not itself decide that a selected geography, time, unit, or measure is
  semantically compatible with the user's request; that limitation must remain
  visible in the final answer and is a separate verification seam. The
  catalogue workflow now runs `dataset.schema` after materialization and
  before handing the dataset to the metadata workflow, so the model sees the
  actual catalogue columns rather than guessing field names. Its route-bound
  procedure restricts catalogue filtering to one title substring; requested
  geography and period are checked later against SDMX metadata, not as
  guessed catalogue predicates. The metadata graph runs a model-parameterized bounded `data.query` over the
  host-materialized catalogue. Its compact DSL accepts `where: title
  "population density"` as a host-defined case-insensitive `contains`
  predicate; explicit operators remain available for other comparisons. The
  query supports case-insensitive text
  `contains` predicates, so the original request can narrow titles before the
  model chooses only among returned `/code` values; it then retrieves that
  code's SDMX dataflow using Eurostat's fixed dataflow version `1.0` and
  bounded `references=children`. Both SDMX responses are retained as
  provenance-bearing resources; OpenAPI returns a compact resource handle
  rather than duplicating complete XML in the 16 KiB model-facing tool result.
  `dataset.inspect` uses the shared bounded SDMX-XML importer to materialize
  the dataflow/DSD and ContentConstraint records. The graph requests the
  dataflow's version-1.0 ContentConstraint after inspecting the DSD. Versioned
  DSD IDs and codelist references come from the dataflow response; the
  constraint dataset contains dataset-specific allowed values.
  The SDMX endpoint rejects `lang`, so it is not sent. The
  accepted choice is persisted as host-validated structured evidence and is
  bound into the `eurostat.getData` transition; it is not recovered from
  free-form model text. The retrieval graph then runs the fixed API operation,
  binds its returned resource reference into `dataset.inspect`, and only after
  materialization becomes terminal can the ordinary dataset-analysis or
  quality workflow continue. Each handoff revalidates its declared target and
  execution envelope.

  Earlier traces predate the declared verification gates. A failed Qwen run on 2026-10-16 selected the
  ENP-East-only `enpe_demo_r_d3dens` candidate for a Sweden request, then
  emitted metadata URLs as `geo` values. The API returned HTTP 200, but the
  unfiltered JSON-stat was rejected by `dataset.inspect` at the bounded
  Cartesian-cell limit. That exposed the missing candidate/constraint check; see the audit
  trace under `var/codex-mcp-tool-audit/` before treating this model path as a
  successful end-to-end workflow.

  The full 2026-10-16 Qwen Instruct trace completed catalogue materialization
  (12,241 rows, 48 batches), dataflow and ContentConstraint retrieval, JSON-stat
  fetch, and dataset materialization. The selected `demo_r_d3dens` has a regional
  title but contains a valid national aggregate: saved constraints include SE,
  2020, A and PER_KM2, and the JSON-stat response reports Sweden, 2020 and 25.2
  persons/km2. The initial audit's claim that this selection was wrong based on
  its regional title was incorrect. Package v23 replaces the overly restrictive
  title instruction with metadata-based coverage guidance. The metadata graph's
  terminal state is named `metadata_materialized`, not `dimensions_and_codes_verified`,
  because inspection does not implement a semantic verification gate.
  Final answer generation still stopped at context pressure. Package v24 adds
  declared host checks for the constraint and returned codes, but their build,
  model-free and model verification remain pending. The final-context budget
  still needs follow-up. See
  `var/codex-mcp-tool-audit/eurostat-qwen-20261016-v16-run22/qwen-e2e.log`.

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

## Report composition

Report composition is a host-owned workflow transition, not a synchronous
model tool that hides several generations. A bootstrap package may declare
versioned `artifact_profiles` and an `artifact_compose` transition for each
section. The runtime materializes one resumable `artifact_compose` plan step per
transition; section generation therefore has ordinary step identity, tracing,
budget handling and resume behavior. The starter example is
`docs/examples/agent-bootstrap-workflows-v1.json` (`research-report-composition-v1`).

Before the first report section, the host persists one immutable, versioned
`artifact_snapshot` observation. It projects current plan observations,
assumptions, dataset/resource provenance and, when present, the matching
Research Workspace checkpoint: objective, source metadata, evidence statements
and relations, source locations, comparisons, coverage and gaps. The snapshot
retains workspace/checkpoint identity and fingerprints these fields. Later
sections and export reuse it instead of silently reading newer observations.
This is a frozen report input, not a replacement for the research workspace or
source resources.

The model writes each section using the request's selected output format. The
default compact DSL allows repeated blocks and an explicit empty result:

```text
block: observation
text: A concise statement grounded in the supplied source snapshot.
evidence: host-evidence-id
source: resource-uri-or-observation-id
```

An empty section is `status: no_findings`. A JSON-selected model returns
`{"status":"findings","blocks":[...]}` or
`{"status":"no_findings"}` with the same block fields. Each profile section
declares allowed block kinds. The host provides short handles such as `E1` and
`S1` in a budgeted, JSON-escaped evidence view, maps them back to canonical IDs
after generation and records references omitted by the context budget.
That view is intentionally smaller than the frozen snapshot: it carries a
bounded objective and coverage state, then prioritizes compact source/evidence
pairs (`source_handle`, title; `evidence_handle`, linked `source_handle`,
grounded text, support relation and verification state). Long host IDs,
duplicated resource URIs and full snapshot fingerprints stay in the immutable
host snapshot rather than consuming the model's evidence budget. Full gaps and
omitted-reference provenance remain host-side; only records actually present
in the view can be cited by the model.

The host owns section/block IDs and checks reference integrity against the
frozen snapshot. It preserves research evidence relations and marks generated
prose unverified unless the source workspace provides a relation; linked
support is still not proof of entailment. Unknown references, contradictions,
missing required sections, critical source gaps and recorded context omissions
make report coverage incomplete. Sections are stored as host-authorized plan
observations with the snapshot fingerprint, so resume reuses validated output
without generating or appending it twice.

Profiles may set `required` (default `true`) and `evidence_relations` per
section. A required `no_findings` result completes the section but adds an
explicit coverage gap; it does not assert that the phenomenon is absent. An
optional empty section does not itself reduce coverage. Starter research
profiles require `findings` and make `limitations` optional. Contradictions,
critical source gaps and other report-level coverage issues still count even
when the optional limitations section is empty.

Section preparation records relevant, included and omitted references, with a
host-owned omission reason. With no relation filter, all workspace evidence is
relevant. A configured filter selects related claims/evidence and retains
contradicting or qualifying evidence for selected claims. Relevant records
omitted for budget or access make coverage incomplete; records outside the
declared section filter are reported separately and do not. Evidence excerpts
are kept as complete records rather than silently clipped; records that do not
fit become explicit omissions. The same scoped reference index validates both
handles and canonical IDs, so a model cannot cite a snapshot record that was
not included in its section input. Evidence/source pairs must also agree with
the frozen provenance relation. Dataset references are recognized by the same
index and validator.

Only handles whose records fit in the section context are accepted for that
generation. `no_findings` remains available when evidence supports another
section but does not directly address this section's purpose. Generated block
text is constrained to one short, complete sentence. If syntax or a reference is invalid, the runtime makes one bounded,
targeted repair generation; a second invalid response leaves the step
incomplete. Research source coverage and report coverage remain separate:
the export's `coverage_complete` is false when required sections, references,
or relevant context are missing, even if the research workspace's own coverage
was complete.

The final declared `artifact.export` transition builds linked resources for
the frozen source snapshot JSON, structured report JSON and deterministically
rendered Markdown. DOCX is an optional export through the existing
Markdown-to-DOCX resource processor. Resources use project scope when the
active project exists, otherwise session scope, and lineage links snapshot,
report and rendered output. Export is explicit in the workflow; it does not
happen merely because a profile exists.

DOCX export is available only when the host enables `resources.processor_policies.docx.text`,
Pandoc is installed at the configured executable, and `sandbox.workspace.root`
points to a writable host workspace. The processor runs as the bounded
`resource.processor.pandoc` class with artifact-only filesystem access and no
network; an explicit sandbox class can further restrict its limits. Without
this host setup, Markdown and structured JSON publication remain available,
but a requested DOCX export fails as unavailable rather than silently
publishing Markdown in its place.

Profiles define presentation and coverage, not tool authority. Workflows
orchestrate and resume sections; blueprints/routes select the method;
`artifact.export` publishes the composed result. A published report revision is
immutable; a deliberate recomposition creates the next revision with a link to
the previous one. The workflow may provide a stable `report_id` when a new
composition continues an existing report; a changed frozen snapshot increments
the content revision. Re-exporting the same snapshot, including in another
format, keeps the revision. Export uses deterministic cache keys for the
snapshot, document and Markdown, verifies exact bytes before reuse, and writes
a publication manifest last. Retries therefore reuse verified resources and
cannot advertise a partial publication as complete. The manifest and successful
tool observation retain the revision and resource references.
The first-party Document Editor MCP App can create user-authored documents and
new immutable report revisions while the original research evidence remains
authoritative. Changed text is unreviewed even if old references remain
attached. Review happens in a normal daemon turn; a proposed patch is never
applied until the user approves it against the still-current revision.
Simultaneous multi-user collaboration is not included. See
[Document editor MCP App](document-editor.md).

When a completed `artifact.export` returns the structured report and Markdown
resource references plus `coverage_complete`, the host also constructs the
conversation's completion message from those verified fields. It does not run
a second free-form draft that could replace explicit coverage gaps with a new
unsupported report. If those fields are absent, normal response generation is
left unchanged.

The full-trace Qwen smoke for this route is
`scripts/test-agent-report-composition-model-smoke.sh`. It runs with GPU
auto-fit and writes its log under `$HOME/llama-agent-smokes/` by default (or
under `LLAMA_AGENT_SMOKE_WORKDIR` when explicitly set). Its empty-source case
checks that the host freezes the snapshot, executes every section transition,
publishes through `artifact.export`, and retains incomplete coverage rather
than inventing citations. It verifies the publication manifest, persisted
report/Markdown resources and host-built completion summary. This remains a
grounding-boundary check, not evidence of report quality on populated research
data.

`scripts/test-agent-report-composition-sourced-model-smoke.sh` exercises the
populated path with two synthetic source files. It uses persistent Cozo plan,
memory and resource-metadata databases plus filesystem-backed resource blobs,
then starts the MCP stdio server as a separate process against those same stores
and issues the standard `resources/read` requests for both the structured
report and Markdown. The smoke checks the returned URIs, the fixture date, and
the report's structured source bibliography / Markdown `Sources` section. This
readback uses the MCP resource endpoint directly rather than asking a second
model turn to plan a read, so it tests persistence and authorization without
adding model-planner variability. The generation trace and MCP wire readback
are saved separately in the smoke work directory. The synthetic fixtures are
not historical authorities.

Publication rechecks stored blocks against each section's included references
and allowed block types. A canonical reference in the wider snapshot cannot
bypass that selection. Stored `no_findings` results cannot also contain blocks.
The bibliography reuses a research source when a plan observation refers to the
same resource with a compatible fingerprint, retaining the research metadata
instead of emitting a duplicate resource citation. Empty optional sections state
that no findings were produced; they do not assert that a phenomenon is absent.
Documents record `markdown_renderer_version` (currently 3). Documents without
that field retain version 1 rendering, so existing publication manifests remain
verifiable. Format changes reuse the published document, including its renderer
version and bibliography, rather than reconstructing an existing revision.

New source bibliographies retain separate `original_uri`, `resolved_uri`,
`resource_uri`, `file_name`, `source_provider`, `source_operation` and
`request_summary` fields. Markdown and its derived DOCX show separate Original,
Retrieved from (when redirected), and Saved source links. Titles fall back to
the available filename, provider/operation, original URI, then source identity.
No external address is inferred from a provider name. An API endpoint is shown
when retained by the host; otherwise the saved result and available request
arguments remain the reference. File names do not imply an external URI.

The bounded request summary comes from host-retained executed arguments,
including nested filters, search terms and periods. Pagination cursors, transport
limits, headers and credentials are excluded. Omitted oversized arguments are
identified; the full provenance remains in the source snapshot. Existing
resource parent links connect PDF/OCR/chunk representations to captured original
URLs without fetching them again. Missing ancestry remains unknown.
Both persistent plan backends preserve resource request provenance and parent
lineage across resume; Cozo also preserves the full dataset request provenance.

Snapshot V4 includes the existing resource provider/operation/request metadata
for research sources, so it survives export and readback. V1–V3 retain their
original canonical forms and fingerprints. Markdown renderer versions 1 and 2
also retain their earlier output when verifying published resources.

The populated smoke also checks that the model context contains paired `S`/`E`
handles and requires both fixture evidence records to be cited with their
corresponding host handles (`E1`/`S1` and `E2`/`S3` in the current fixture
ordering), alongside the shared date in the exported report. Handle names are
assigned by the host; the smoke assertions pin them only for these fixtures.
This validates the acquisition-to-citation seam, not external factual accuracy.
Web search results alone are discovery candidates and do not close research
gaps or become supporting evidence before a source is read. Contradiction
behavior is covered by the model-free snapshot/report tests.

Validation on 2026-10-16 with Qwen2.5-1.5B-Instruct Q4_K_M on the NVIDIA
GeForce 920MX passed both report scenarios: the empty-source run returned an
incomplete-coverage completion without a free-form second report; the populated
run produced `20 July 1969` with both evidence/source pairs, then the separate
MCP `resources/read` process retrieved the persisted report JSON and Markdown
with the date and bibliography intact. Full traces are retained under
`$HOME/llama-agent-smokes/report-composition-*` on the test host.

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
