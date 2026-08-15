# Agent Runtime Architecture

This note describes the current resident-agent runtime direction. The main completed step is the inference/runtime abstraction that now sits between agent behavior and the old CLI-local generation path, with a small foreground daemon layered on top as an admin/test transport.

The goal is to make `llama-agent` able to run the same agent turn from different hosts. The CLI is the first host adapter. A future resident process, service, or MCP-facing application should build the same runtime contracts directly instead of pretending to be CLI arguments.

## Agent runtime modes

The agent runtime deliberately has a smaller mode vocabulary than ordinary
chat. `direct` remains a chat/runtime behavior; it is not an agent thinking
mode. Agent turns start at `reflective` and may escalate to `deliberate` or
`research`.

```text
runtime mode:  chat | agent
thinking mode: reflective | deliberate | research
```

`reflective` is the minimum assurance loop:

```text
understand -> execute/reason -> draft -> bounded review -> optional revision -> answer
```

It does not require a persistent plan, plan revision, evidence extraction, or
source cross-checking. Its normal policy is one bounded reflection round and
no research iterations.

`deliberate` is the structured multi-step loop:

```text
frame problem -> establish goal/constraints -> plan -> execute -> step review
    -> revise plan when needed -> review draft -> answer
```

Its policy requires a plan, allows bounded plan revisions, and requires review
of relevant work before final answer acceptance. The runtime now emits one
typed step-review event for each completed plan step during the deliberate
reflection pass, plus a typed answer-review event. Plan revisions remain
bounded by the resolved deliberate policy.

Deliberate alternatives and working decisions use the existing plan contract:
constraints and assumptions are proposed through typed plan operations and
remain attached to the plan history. This keeps them inspectable and
revisable without introducing a second hypothesis or decision store.
The bounded runtime plan context now renders their current valid/invalid state
and confidence into the model-facing runtime context, so a later deliberate
revision can explicitly preserve, challenge, or replace an assumption while
remaining inside the active constraints.

`research` is a deliberate runtime with an evidence-acquisition loop:

```text
knowledge gap -> query -> source fetch -> evidence extraction
    -> source comparison -> synthesis -> verification -> stop/iterate
```

Research is not a special MCP tool. A controller uses host-approved tools such
as web search/fetch, repository search, resource read, memory search, and
remote MCP tools. Its temporary workspace composes existing plans,
observations, resource references, trace entries, and turn/session state. It
does not create a parallel persistence model.

The first research contract slice now exists in `common/agent/research`. It is
an ephemeral, scope-bound workspace model for objectives, gaps, tasks, sources,
evidence, budgets, coverage, and a normalized result. Sources may point at the
existing resource store through `common_runtime_resource_ref`; the workspace
does not duplicate raw material or create a second durable memory store. The
bounded controller now selects the highest-priority open gap, schedules one
host-approved research task, consumes a typed task event, records returned
evidence IDs, updates coverage, and stops on coverage, budget, no-progress,
policy, or cancellation conditions. It deliberately does not execute tools,
own resources, perform synthesis, or write memory.

The first execution slice now exists beside the controller. A research runner
drives the controller sequentially through the existing
`common_agent_tool_runtime`; its adapter selects a host-approved research tool,
normalizes the successful result into a workspace source and evidence item,
and feeds the typed event back to the controller. User-supplied resources and
memory hits are represented as sources and can be selected through bounded
`resource_read` and `memory_get` tasks. Multiple objective success criteria
become separate research gaps. The normalized result preserves source,
evidence, comparison and provenance data for later verification.

The bounded research slice now also includes source comparison, a bounded
synthesis context, and a separate answer-verification contract. Tool success
is distinct from gap completion: a typed assessment can be sufficient,
insufficient, inconclusive or contradicted. The current default assessor uses
bounded completion-criterion matching; a host can later provide a
model-backed assessor through the same seam. The answer verifier checks draft
coverage, evidence, plan observations, memory references and user resource
references without creating another persistence layer. If verification asks
for more evidence, the runtime can reopen the same turn-local workspace once,
carry forward the prior evidence, run one additional bounded research task,
update the synthesis context, regenerate the draft and verify it again. The
reopen path has an explicit `research_reopened` event and does not recurse.

Research now starts after the initial plan has been materialized. The runtime
passes that plan identity into the research workspace and emits the
`research_started` event with the associated `plan_id`; research remains a
workspace/controller concern rather than expanding the plan into individual
research tasks.

The ownership boundary is deliberately narrow. The outer plan owns the
research objective, the required contribution and the eventual interpretation
or synthesis. The research controller owns source acquisition, research task
ordering, acquisition-tool selection, retries, source comparison and evidence
coverage. A model-generated research plan must therefore describe what must be
known, not independently create a second acquisition workflow. The controller
uses the same host-resolved tool authority as the rest of the runtime; it may
select only an approved acquisition capability and may not expand the active
tool profile.

The first deterministic acquisition path is source-driven. When an unused
user-supplied resource is available, the controller schedules `resource_read`
with host-owned resource identity and bounds. It does not fall through to
`web_search` merely because the model mentioned the web. Source comparison and
coverage transitions remain controller state transitions, not additional model
tools or research tasks. The model receives the bounded evidence and synthesis
context after acquisition has completed.

This boundary is being tightened incrementally. The current model-free CSV
research smoke verifies that two local resources produce two `resource_read`
calls and no alternative acquisition tool. Completed research is now also
bridged into the outer plan as one bounded `research_workspace` observation
carrying evidence IDs and resource references.

The runtime also applies the first acquisition ownership guard: in research
mode, model-planned acquisition calls such as `resource_read`, `web_search`,
`web_fetch`, repository search, memory retrieval and dataset acquisition are
degraded to bounded reasoning steps before the normal plan scheduler can
execute them. The controller retains the approved acquisition path. This is a
runtime ownership guard, not a second tool profile; both paths remain subject
to the same host-resolved authority.

Inspection is a separate responsibility. After acquisition, the model may
choose an applicable host-approved inspection or analysis operation, such as
`resource_inspect`, `dataset.schema`, `dataset.sample`, `statistics.describe`
or `statistics.value_counts`, when the corresponding representation or dataset
reference is available. These operations are not acquisition substitutes: a
successful `resource_read` establishes bounded source evidence but does not
imply that a tabular schema, document outline or dataset has already been
materialized. The model-facing resource catalog advertises possible bounded
inspection seams by MIME type; the host still validates the final reference,
scope and representation.

`llama-agent-research-runtime-smoke` covers two gaps, tool execution,
source/evidence creation, provenance, answer verification and cancellation
without network access. The common runtime emits structured research events
and a `research` trace stage. The initial and reopened research trace entries
carry the associated `plan_id` and turn-local workspace id, so the existing
session/daemon host can carry the research lifecycle without a separate
progress channel.

`llama-agent-research-csv-ctest` is the model-free resource-backed research
smoke. It reads bounded UTF-8 CSV fixtures through a host-owned
`resource_read` seam, retains both resource references in research evidence,
creates a deterministic source comparison, checkpoints and reloads the same
workspace, and verifies that incomplete coverage is deferred rather than
reported as complete. This complements the deliberate tool smoke: it tests
research coverage, provenance and continuation without depending on a model's
query decomposition or formatting.

When agent tracing is enabled, the research stage also records bounded
lifecycle entries for gap/task/iteration transitions, source and evidence
registration, source comparisons, checkpoints, completion and failure. Trace
entries carry IDs and counts only; raw resource contents and CSV rows remain
outside tracing. This is intended to make model-based Qwen/Phi runs diagnosable
without changing the research contracts or adding a second event channel.

The following research capabilities remain intentionally incomplete:

- model-driven gap decomposition and query reformulation;
- model-backed semantic gap assessment;
- structured claim extraction from the draft;

All mode budgets are host-resolved. A user or caller may request a mode and
bounded limits, but host policy remains the upper bound for reflection rounds,
plan revisions, tool rounds, research iterations, tokens, and time.

## Host-owned tool profiles and capabilities

Tool authority is resolved by the host before a turn starts. The host selects
one profile through configuration or the startup CLI option, resolves that
profile against configured capabilities and available providers, and exposes
an immutable tool snapshot to the runtime:

```text
host config / CLI
    -> active profile
    -> capability map and profile definition
    -> resolved native/provider tools
    -> immutable runtime tool view
```

Capabilities are configuration mappings from stable capability IDs to tool
names. Profiles include or exclude those capabilities and may carry effective
network and policy-gated-write flags. The built-in profiles remain available,
but custom profiles do not require a code change. Caller policy may narrow the
resolved view; it cannot select a profile or expand it.

The daemon `ready` response includes a `tooling` diagnostic with the active
profile, configured capabilities, resolved tool names and effective policy.
Public turn requests that try to provide `tool_profile`, `allowed_tools`,
`allow_writes` or `enable_shell` are rejected at the protocol boundary.

The catalog also defines the first semantic developer execution contracts:
`development.build` and `development.test`. They accept a target-oriented request and carry
host-owned sandbox policy metadata; they do not accept Docker, Kubernetes or
shell details. Their executors are intentionally kept behind the host-owned
sandbox binding while the model-facing schema remains stable. Docker is the
first concrete backend and Kubernetes now uses the same seam through
ephemeral Jobs.

Kubernetes deployment settings belong to host configuration, not to agent
requests. This includes the `kubectl` executable, namespace, service account,
runtime class, storage class, PVC sizes, staging image, kubeconfig/context and
TLS verification and cleanup/retention policy. The current defaults are `4Gi` for workspace data
and `1Gi` for artifacts; an empty storage class uses the cluster default. PVC identity is derived from
`project_id` when present and otherwise from the session workspace identity.
Operation directories are created below that PVC. Clients cannot choose PVC
names, mount paths or Kubernetes Job details.

## Tool execution boundaries

Tools are classified by where their work is performed and which host-owned
authority they use. The classification is a runtime design constraint, not a
choice exposed to the client:

| Tool class | Responsibility | Typical examples |
| --- | --- | --- |
| Host-native | Reads or changes the host's controlled workspace or repository | `workspace.read`, `workspace.search`, `workspace.patch`, `repository.diff` |
| Sandbox-backed | Runs build, test, analysis or transformation in an isolated executor | `development.build`, `development.test`, `python_execute`, `data_compute` |
| Store-backed | Queries an authorized internal store or catalog | `memory_search`, `resource_read`, `plan_lookup` |
| Artifact | Publishes sandbox/store results as named resources with provenance | `artifact_export`, `report_create`, `chart_export` |
| Provider-backed | Uses a host-configured external provider or MCP server | web search, repository search, remote resource tools |
| Orchestration-backed | Schedules, delegates or observes bounded operations without performing the work itself | `delegate_task`, operation status and scheduler tools |

The classes describe the primary execution boundary. They are not exclusive
effect labels: a sandbox-backed tool may produce artifacts, and a
provider-backed tool may register resources in the store. Tool metadata must
therefore keep execution class separate from capability, effect, risk and
artifact production.

### Tool execution hardening

The resolved native and MCP views reserve the per-turn tool-call budget after
validation and under the same mutex as the counter update. Validation may run
concurrently, but two calls cannot reserve the final available slot.

Native results and MCP results are normalized through the same host-owned
result-size boundary. MCP structured content and text content are counted
together, using the lower of the runtime default and the MCP definition limit.
MCP arguments are normalized and validated before the normalized JSON is sent
to the client; the provider does not validate one representation and execute a
different one.

Tool repair context now contains only schema-required properties, using schema
defaults or numeric minimums when available. Optional properties are omitted
so that a repair suggestion cannot look like a complete call while silently
inventing unrelated arguments.

Memory reference arguments have one additional compatibility normalization
before schema validation. `memory_get`, `memory_propose_update` and
`memory_propose_forget` accept the canonical `{"id":"..."}` form, the
equivalent `{"memory_id":"..."}` form, and (for ID-only calls) a JSON string
such as `"memory-1"`. All three forms become `{"id":"..."}` internally.
This normalization changes only the representation; the ID must still come
from a prior `memory_search` result or an explicit recorded memory reference.
It never selects a memory implicitly, which is important when asynchronous
plan steps or multiple searches are in flight.

Async native and MCP tool calls now carry an operation-owned cancellation state
and an effective deadline. The host can cancel a pending call explicitly, while
the operation manager can mark it timed out and invoke the cancellation callback
outside its mutex. MCP stdio enforces the deadline by terminating the bounded
request subprocess when necessary; MCP HTTP bounds connect/read timeouts by the
remaining deadline. Native handlers receive the same cancellation/deadline state
and are checked before and after the synchronous handler call, so cancellation
is cooperative until native handlers gain an interruptible execution seam.

The following items remain deliberately separate backlog work: a bounded native
worker/executor with hard interruption, a distinct per-call confirmation protocol
for write tools, and stricter fuzzy-name policy for policy-gated tools. The
current `requires_confirmation` flag remains an exposure/policy gate; it is not
presented as an interactive approval handshake.

## Tool catalog

The current foundation catalog is intentionally small and semantic. Host
configuration selects which definitions are exposed through a profile; clients
do not select implementations or host paths.

| Tool | Boundary | Current foundation behavior | Maturity | Missing / next step |
| --- | --- | --- | --- | --- |
| `workspace.list` | Host-native | Bounded directory listing inside the controlled repository root | Foundation | Ignore rules, richer entry metadata and pagination |
| `workspace.read` | Host-native | Bounded text-file line reads | Foundation | Binary/resource reads, encoding metadata and larger resource references |
| `workspace.search` | Host-native | Bounded text search over readable files | Foundation | Index-backed search, better language awareness and stable pagination |
| `workspace.patch` | Host-native mutation | Hash-token checked line operations; confirmation-gated | Limited | Atomic multi-file patches, rollback and structured diff artifacts |
| `repository.diff` | Host-native | Bounded Git working-tree diff summary | Limited | Explicit base/head/range selection and larger patch artifacts |
| `development.build` | Sandbox-backed | Declarative target request through the configured sandbox backend | Contract-level | Full backend execution, diagnostics parsing and artifact import per tool |
| `development.test` | Sandbox-backed | Declarative test request through the configured sandbox backend | Contract-level | CTest/JUnit result parsing, cancellation and persistent test history |
| `diagnostics.compile` | Host-native analysis | Parses bounded GCC/Clang- and MSVC-style compiler output | Limited | Compiler invocation, richer formats and source-linked diagnostics |
| `diagnostics.symbol` | Host-native analysis | Host provider seam with bounded text fallback | Experimental | clangd/LSP or project-index backend and definition-kind ranking |
| `diagnostics.references` | Host-native analysis | Host provider seam with bounded text fallback | Experimental | Semantic references, reference kinds and project-index persistence |
| `diagnostics.call_hierarchy` | Host-native analysis | Semantic-provider contract; unavailable without a provider | Experimental | clangd/LSP callers/callees, depth bounds and project-index persistence |
| `dataset.list` | Host-native | Lists bounded host-approved dataset references and legacy files | Limited | Dataset registry, permissions/provenance and non-file sources |
| `dataset.inspect` | Host-native | Returns bounded dataset identity, source and shape metadata | Limited | Workbook/sheet metadata and richer lineage |
| `dataset.schema` | Host-native | Returns a bounded typed column schema | Limited | Type confidence, constraints and backend parity |
| `dataset.sample` | Host-native | Returns a bounded sample from a dataset reference | Limited | Random/stratified sampling, seed handling and non-Cozo backends |
| `dataset.validate` | Host-native | Validates CSV `not_null` and `unique` rules | Limited | Range/type/regex rules, validation profiles and artifact reports |
| `data.query` | Store-backed | Executes a bounded backend-neutral structured query | Limited | Broader query language, materialized inputs and backend parity |
| `data.filter` | Store-backed | Applies declarative predicates through the configured data store | Limited | More operators, typed values and derived dataset artifacts |
| `data.aggregate` | Store-backed | Applies bounded grouping and aggregate operations | Limited | More aggregate functions, null semantics and provenance |
| `data.join` | Store-backed | Joins two host-approved datasets | Limited | Join validation, larger results and artifact-backed output datasets |
| `data.transform` | Store-backed | Applies bounded column transformations | Limited | Safer expression language, type checking and artifact persistence |
| `statistics.describe` | Store-backed | Produces bounded descriptive statistics | Limited | Confidence intervals, distributions and assumption reporting |
| `statistics.outliers` | Store-backed | Identifies bounded numeric outliers with IQR thresholds | Limited | Z-score, MAD and richer anomaly models |
| `statistics.value_counts` | Store-backed | Returns bounded frequency counts for one column, including nulls | Limited | Grouped frequencies, percentages and cumulative distributions |
| `diagnostics.test_failures` | Host-native analysis | Groups normalized failures and classifies common causes | Limited | CTest/JUnit/JSON parsers, stack traces and cross-run grouping |
| `diagnostics.format` | Host-native analysis | Interprets bounded formatter output and reports files needing formatting | Limited | Real formatter backend, patch artifact generation and format profiles |
| `diagnostics.include_graph` | Host-native analysis | Parses bounded `source -> include` dependency output | Limited | Compiler database extraction, cycle analysis and persisted graph queries |
| `artifact.export` | Store-backed artifact | Publishes bounded text or dataset results through the host resource store | Limited | Binary formats and richer provenance/retention policy |

Maturity is a capability status, not a quality rating:

* `Foundation` means the bounded host-native behavior is implemented and directly
  testable.
* `Limited` means a usable first implementation exists, but the current
  schema or backend intentionally covers only a narrow subset.
* `Contract-level` means the semantic tool contract and policy seam exist, while
  useful execution still depends on a configured sandbox/backend integration.
* `Experimental` means the tool is available for controlled use, but its
  result is explicitly provisional, for example the text fallback for symbol
  and reference lookup.

The `Missing / next step` column is deliberately maintained as an architectural
backlog, not as a promise that every tool must grow into a general-purpose
shell, compiler or data-science environment.

The first data foundation deliberately supports safe discovery and bounded
materialization. Cozo-backed queries, filtering, aggregation, joins,
transformations and descriptive statistics now belong behind the data-store
seam. Charts and richer artifact conversion remain follow-up work. Semantic
symbol analysis now has an explicit host-owned provider seam, while the default
build uses a bounded text fallback when no semantic index is available. No data backend is implicitly selected when the
host has not configured one.

#### Resource, dataset, tool and artifact boundaries

These concepts intentionally remain separate:

* A **resource** owns authoritative bytes, media type, scope and source
  provenance. A resource processor may create a derived representation, but it
  does not replace the original resource.
* A **dataset** is a structured analytical handle. It owns tabular schema,
  rows, dataset-level lineage and backend identity; it is not a specialized
  resource and should not be serialized into the model context as a large JSON
  value.
* A **data tool** is a model-visible semantic operation over authorized dataset
  references. Spreadsheet parsing and backend-specific query details remain
  host-owned infrastructure.
* An **artifact** is a user-facing exported resource derived from a dataset or
  result. Export must preserve the dataset/source provenance without mutating
  the input dataset.

The first spreadsheet direction follows the existing processor seam:

```text
authoritative XLSX resource
    -> Pandoc-derived structured representation
    -> bounded worksheet selection/import
    -> dataset reference in the configured data store
    -> dataset tools and immutable derived datasets
    -> optional artifact export
```

Pandoc is an implementation of the resource-processing step, not the dataset
abstraction. Its structured output is an intermediate representation that a
dataset importer validates and materializes. Each worksheet remains a separate
dataset; a workbook is not silently flattened into one CSV. The initial
direction is intentionally bounded and does not promise preservation of every
spreadsheet feature, such as VBA, visual layout, named ranges, hidden-sheet
semantics or formula evaluation. Formula metadata and richer workbook
inspection can be added without changing the resource/dataset boundary.

Dataset contracts should retain typed columns, nullability, stable dataset
identity, source workbook/sheet/range provenance and immutable operation
lineage. This is also the foundation for later statistical analysis: future
descriptive or inferential tools can consume typed dataset references and
report their sampling, missing-value and assumption policies without moving
all rows through the model. The first implementation remains limited to
bounded inspection, schema, samples and deterministic data operations; advanced
statistics, distributions, confidence intervals and charting are later layers.

Document table extraction uses the same dataset contract. Simple rectangular
tables from Pandoc/HTML-derived structure may carry a generic origin projection
with the source representation URI, document node ID, table index, caption,
header mode and bounded classification reason. The document representation
keeps the table; the dataset is an additional analytical handle. Header
classification is host-owned and bounded: explicit headers win, otherwise a
small sample may suggest first-row, first-column, both or ambiguous. An
ambiguous table is not silently materialized with invented column names.

The first table scope deliberately excludes PDF table extraction, merged-cell
normalization, nested or pivot-like tables, automatic unit inference and
LLM-authored authoritative metadata. These remain backlog items.

The Pandoc worksheet normalizer now carries the bounded classification result
into the worksheet envelope. Explicit Pandoc headers are recorded with full
confidence. Headerless tables are sampled; only a clear first-row signal is
eligible for automatic materialization, while first-column, both-direction and
ambiguous results fail closed with a normalization-required diagnostic.

The first dataset artifact-export slice reuses `artifact.export` rather than
adding a spreadsheet-specific tool. The model may provide
`source_dataset`, `format: "csv"`, an optional name and a bounded `max_rows`;
the host reads the dataset through the configured data store, serializes a
bounded CSV, and stores it as a derived `text/csv` resource. Inline text export
continues to use the same tool. The exported resource carries a lineage parent
of the dataset URI, while the dataset remains authoritative. XLSX export,
larger streaming exports and chart artifacts are intentionally later work.

The first common contract slice is implemented in
`common/agent/dataset-contracts.h`. It validates dataset references and
descriptors without selecting a backend. Host limits cover source bytes,
worksheet count, rows, columns, cells and generated datasets; materialization
and query execution remain the responsibility of the configured data-store
seam.

The first importer slice is implemented in
`tools/agent/data/agent-dataset-importer.*`. It normalizes the bounded Pandoc
table AST into a worksheet envelope, creates one immutable dataset descriptor
per worksheet and streams each row through the existing data-store `put_row`
seam. The envelope is deliberately an adapter boundary: raw Pandoc JSON must
not be mistaken for the canonical dataset model. The initial AST adapter
supports table headers, simple text cells and deterministic scalar type
inference; layout, formulas and workbook-only metadata remain explicit
limitations. A backend that does not implement ingestion fails explicitly
rather than falling back to a second store or returning rows through the model
context.

The same adapter is now explicitly available for `document-json` through
`normalize_agent_pandoc_document_json`. It reuses the worksheet envelope and
importer rather than creating a document-specific dataset path. The host marks
the import as `document:table` and supplies the semantic-document resource URI;
the resulting descriptor is therefore classified as `document_table` while
retaining the original document resource and table node provenance. A complete
document analysis can use one Pandoc JSON processing result as the source for
both text projection and simple table materialization; the model does not
select Pandoc or a converter executable.

Document table catalogs are host-owned bounded projections over the same
origin metadata. A model-friendly lookup may use a unique table name or
caption, while `table_index` and `node_id` remain canonical host addresses.
Names are normalized for surrounding whitespace, repeated whitespace and case;
duplicate normalized names fail with an ambiguity error rather than selecting
arbitrarily. The catalog can therefore expose both `Budget summary` and table
index `1` without making a display name part of the canonical dataset URI.

The model-facing surface is intentionally semantic and small:
`document.tables({resource, max_results})` lists bounded table descriptors, and
`document.table({resource, table | table_index | node_id})` resolves exactly one
table and returns its dataset handle. The host validates that exactly one
locator is supplied; processor selection, `document-json` caching and dataset
materialization remain behind host callbacks. These tools are therefore also
model-visible in the `research`, `developer-read` and `all-configured` native
profiles, and are also available through MCP when the active tool profile
includes them, but MCP does
not expose Pandoc, table parsing libraries or filesystem paths.

The importer can materialize all bounded worksheets, or one exact worksheet
selected by the host with `sheet_name` or `sheet_index`; selection is not a
model-selected parser operation. A missing requested worksheet fails closed.

The model-free `llama-agent-data-manipulation-ctest` exercises the existing
Cozo data-store path with two imported-style worksheet datasets. It verifies
bounded filtering, joining, aggregation and materialization of an immutable
derived dataset whose lineage still points to the authoritative workbook
resource. This smoke is intentionally store-level; it does not introduce a
second spreadsheet or query execution path.

Dataset descriptors are persisted through the existing data-store seam as
host-owned metadata. Cozo stores the descriptor separately from row payloads,
typed value indexes and row ordering, and creates that metadata relation when
opening an older database. This keeps dataset identity/provenance available to
inspection tools without making the row store or resource store a second
authority.

The native `dataset.inspect`, `dataset.schema` and `dataset.sample` adapters
accept either a bounded legacy file path or a first-class dataset URI, but not
both in the same call. Dataset-URI inspection reads descriptor metadata from
the configured data store. Dataset-URI samples are delegated to the existing
bounded `data.query` path, so large row sets remain outside model context while
legacy CSV support continues to work during migration.
The first-class dataset URI path only requires the host-owned data-store
binding; it does not require a repository root. Legacy file access remains
repository-root scoped. The same binding rule applies when these tools are
exposed through MCP.

The same inspection tools may also receive an acquired resource handle through
their `resource` field. The CLI host currently supports `text/csv`: it reads
the bounded authoritative resource, normalizes it through the existing
worksheet importer, stores rows and the descriptor in the configured data
store, and returns the resulting dataset reference. The descriptor retains
`source_resource_uri` and `source_representation=csv:dataset`. This is
demand-driven materialization, not a second resource store. Other MIME types
remain unavailable until they provide an equivalent host-owned importer; for
example, document tables use the existing `document.table` path.

Large dataset results stay external. Only bounded schemas, samples, summaries
and dataset references enter the active model context. Resource chunking is
reserved for semantic reading of representations; dataset operations are used
for deterministic tabular computation.

Structured operations retain their existing bounded row-result behavior by
default. A host-approved operation may additionally request `materialize=true`
with an explicit `result_dataset` URI. The Cozo implementation then creates a
new dataset descriptor and rows in the same store, records the parent dataset
URI(s), operation and source-resource provenance, and returns only the new
dataset descriptor. Materialization is fail-closed when scanning or returning
rows was truncated; partial results are never presented as complete derived
datasets. This is the first immutable-derived-dataset seam, not yet a general
transaction or cross-backend materialization guarantee.

The namespaced names are canonical in the catalog, native registry and
resolved profile snapshots. Host configuration, profiles and model-visible
tool views must use these names directly.

The following constraints apply:

* Host-native tools are restricted to host-created workspace and repository
  roots. Mutations use host validation such as expected hashes and explicit
  patch operations.
* Sandbox-backed tools receive semantic requests. They do not expose Docker,
  Kubernetes or shell details to the model, and they do not write arbitrary
  changes directly back to the host workspace.
* Store-backed tools enforce namespace, project, session and turn authority
  at the store boundary. A store result is not automatically evidence until
  the owning runtime registers and scopes it.
* Artifact tools publish explicit, bounded outputs with resource references,
  provenance and size limits. Text in a tool result is not implicitly an
  artifact.

### Tool-call repair

Tool-call repair is a common runtime path shared by `reflective`, `deliberate`
and `research`; it is not a mode-specific tool provider. After a schema or
availability failure, the host records a bounded repair context for the next
reflection/review pass:

* the failed tool and validation diagnostic;
* a host-generated argument skeleton derived from the registered schema, when
  the tool is available;
* the effective model-visible tool names for the resolved profile.

Tool selection also has a host-owned name-normalization step before schema
validation. Exact names are preferred, followed by compatibility aliases and
normalized separators/casing. A unique, high-confidence fuzzy match may be
resolved mechanically; this does not require another model inference. For
example, `join` can resolve to `data.join` when that is the only sufficiently
strong candidate in the effective tool view. The resolver never searches
outside that view.

If more than one candidate is plausible, the runtime does not silently choose
one. It preserves the tool intent as an unresolved repair step and exposes the
bounded candidate list to the common repair pass. Candidates may be grouped by
domain when needed, so a model can select one or more host-provided domains
(for example `data` and `statistics`) without receiving unrelated tools or
being allowed to expand the profile. If a unique fuzzy match has valid tool
name but invalid arguments, the canonical tool step is retained so ordinary
schema repair can repair its arguments rather than reducing the request to
generic reasoning.

Before validation, the runtime also applies the existing compatibility
normalization used by plan and reflection parsing. This unwraps supported
legacy `tool`/`args` shapes and canonicalizes bounded control values; semantic
defaults remain a separate host-owned step. The executor receives the
normalized arguments rather than the pre-normalized model payload.

Planner proposals use the simpler canonical tool shape `{"tool":"domain.operation","args":{...}}`.
Older named-call and nested `tool`/`arguments` wrappers remain accepted at the
compatibility boundary so persisted plans and older small-model output do not
need a parallel execution path. Document table selection also accepts the
model-friendly alias `table_name` and canonicalizes it to `table`; the native
adapter still requires exactly one of `table`, `table_index` or `node_id`.

For an unavailable tool there is no argument skeleton. The repair context only
offers the effective tool view, so reflection can select a valid alternative
without expanding host authority. The context is emitted through the normal
agent event stream as `tool_repair_context_created` and is retained with the
tool-failure observation. A repair pass may correct selection or arguments,
but it cannot change the profile, capability set, policy or backend.

This makes repair a bounded normalization and validation path, not a second
tool-discovery protocol. The model is used only when deterministic matching is
ambiguous or when the arguments need semantic correction. The same path is
available to all thinking modes; the mode changes only the repair budget.

The mode controls only the bounded budget around this common path: reflective
gets the minimum repair/reflection pass, while deliberate and research may
spend their larger review budgets on the same context. Static chat without a
tool call has no repair pass.

### Bounded tool-family navigation

The runtime also has a small operation-local navigation contract for deliberate
and research execution. It describes the current position inside the
host-owned tool tree without creating a second scheduler or plan store:

```text
plan step
  -> tool family
    -> concrete tool
      -> observation
  -> next tool in the family, or return to the plan step
```

The family is derived from the canonical tool name before the first dot. For
example, `statistics.describe` and `statistics.value_counts` belong to the
`statistics` family, while an undotted tool such as `calculator` belongs to the
bounded `utility` family. The current resource tools are intentionally still
in the `utility` family until their names are migrated to a namespaced form.

The navigation context is turn/operation-local and carries operation, plan and
step identities, current family and tool, an optional asynchronous operation
ID, required evidence IDs and completed evidence IDs. It does not own
queueing, persistence or tool execution. The existing session lane and
pending-operation contract remain authoritative for asynchronous work.

After a successful tool result, the host either keeps the context inside the
family when required evidence is incomplete, or marks it ready to return to
the owning plan step. A failed tool blocks the navigation context. Deliberate
and research use the same contract; research normally supplies stricter
evidence requirements, while deliberate may return after a bounded decision
observation. Returning to the plan is explicit and cannot be performed while
required evidence is missing. The existing tool-start trace now records the
resolved family for active plan tool steps, which makes family transitions
observable without exposing host scheduler state to the model. Successful
tool-step completion also forwards the deterministic tool-observation ID to
the existing plan-store completion operation, so `required_evidence` remains
validated by `common_plan_policy` rather than by a parallel navigation rule.

Tool repair is a partial argument patch, not a replacement argument object. If
validation rejects a tool call, a repair that supplies only the corrected field
is merged with the failed call's original arguments before the strict contract
is validated again. For example, a repair containing
`{"max_bytes":8192}` preserves the original resource identifier,
representation, and offset. A tool-backed plan step cannot be completed from
model prose or an arbitrary evidence identifier: completion must reference a
host-recorded observation whose source is the same tool. This keeps resource
research evidence separate from executable plan-step evidence.

## Tool naming convention

New tool names use dotted namespaces so related operations are visible as one
family:

| Canonical name |
| --- |
| `repository.list` |
| `repository.search` |
| `repository.read` |
| `repository.diff` |
| `repository.log` |
| `repository.status` |
| `repository.changed_files` |
| `workspace.list` |
| `workspace.read` |
| `workspace.search` |
| `workspace.patch` |

Repository and workspace tools use only their namespaced model-facing names.
The former underscore names are no longer accepted by the catalog, profiles,
planning or research paths.

Structured data tools use the host-owned `common_agent_data_store` interface. The
tool layer submits semantic JSON operations and does not select CozoDB, DuckDB or
another backend. A backend is supplied by host configuration and can be replaced
without changing tool schemas. The current tool adapter reports the data tools as
unavailable when no backend is bound. CozoDB is the intended first persistent
backend for internal relations, provenance and graph-style queries; file-oriented
CSV/JSON/Parquet analysis can use another backend later.

When `LLAMA_MEMORY_COZO` is enabled, `common_agent_cozo_data_store` provides a
persistent implementation of the same interface. It stores host-materialized
rows in a controlled Cozo relation and currently supports query, filtering,
basic aggregation, descriptive statistics, joins and bounded column
transformations. The host still owns opening the database and binding the store
to the runtime; the model cannot select a database path or backend.

The first bounded statistics contract returns `count`, `null_count`, `min`,
`max`, `mean` and population `stddev` for selected numeric columns, together
with scan metadata. If no columns are supplied, the host uses numeric columns
from the persisted dataset schema. `group_by` returns one bounded description
per group. The result keeps group keys separate from column statistics, so
callers can distinguish a grouped result from an overall description. This
keeps descriptive statistics deterministic and separate from future advanced
statistical analysis such as IQR/MAD outlier detection.

The first outlier contract is `statistics.outliers`. It uses the bounded IQR
rule (`q1 - multiplier * iqr`, `q3 + multiplier * iqr`) and defaults to a
multiplier of `1.5`. With `group_by`, thresholds are computed independently
per group. The result returns the thresholds and bounded source rows that were
flagged. It is intentionally not a claim that a flagged row is erroneous;
the operation identifies statistical candidates for review. Z-score, MAD and
period-change detection remain later extensions.

The dataset contract also has a small model-facing normalization layer. It
accepts a single `group_by` or `column` value for `statistics.describe`, a
single group key and shorthand measures for `data.aggregate`, a shared `on`
column for `data.join`, and `limit` as an alias for `dataset.sample.rows`.
These forms are converted to the same canonical operations before validation;
backends do not implement separate shorthand paths.

Database-backed operations should be expressed as database operations whenever
the selected backend can execute them safely. The common data contract groups
them as follows:

| Operation family | Examples | Backend responsibility |
| --- | --- | --- |
| Projection | `select`, column projection | Return only requested columns |
| Filtering | `where`, `data.filter`, equality and ordered predicates | Apply predicates before result materialization |
| Ordering and paging | `order_by`, `limit`, `offset` | Sort and bound rows in the backend |
| Distinctness | `distinct` | Remove duplicate projected rows in the backend |
| Grouping and aggregation | `group_by`, `count`, `sum`, `avg`, `min`, `max` | Group and aggregate before returning rows |
| Joins | `inner`, `left`, and later other declared join types | Join relations using declared keys |
| Data quality predicates | null, unique and range checks | Prefer indexed or relation-level validation |
| Graph and recursive queries | reachability, ancestry, dependency traversal | Use Cozo/Datalog or another graph-capable backend |

`max_scan_rows` limits backend input scanning and `max_result_rows` limits the
returned result. Results report `scanned_rows`, `row_count`, `scan_truncated`,
`result_truncated` and, where relevant, `scan_mode` so callers can distinguish a
bounded scan from a small result. Cozo keeps a structured field-value relation
alongside the JSON payload relation and compiles typed `data.filter`/
`data.query` predicates into CozoScript before materializing matching payloads.

Dataset predicates have one canonical host/backend form:
`{"field":"country","operator":"=","value":"Sweden"}`. To make
the same tools usable by smaller models, the tool boundary also normalizes two
bounded shorthand forms before validation and backend execution:

```json
{"where":{"country":"Sweden","population":{"gt":100000}}}
```

and:

```text
population -gt 100000 and country in ["Sweden", "Norway"]
```

The object form means an `AND` list of equality or one-operator predicates.
The expression form supports only bounded `AND` predicates with `eq`/`-eq`,
`neq`/`-neq`, `gt`/`-gt`, `gte`/`-gte`, `lt`/`-lt`, `lte`/`-lte` and `in`.
`data.query.where` and `data.filter.conditions` use the same normalizer.
Canonical predicates remain the authoritative contract; tracing, repair and
backend calls use the canonical representation. The normalizer is shared by
native and MCP tool paths and does not turn the expression form into a general
SQL or scripting language.

Ordering, grouping, aggregation, inner joins and left joins use the structured
relation as well. Cozo-backed rows now have a host-owned per-dataset ordinal,
which lets native aggregation and joins apply `max_scan_rows` before grouping or
joining and report `scan_mode: native_bounded`. Existing databases are migrated
when opened. The ordinal is an execution bound, not a user-visible ordering
promise; deletes can leave gaps and newly inserted rows receive the next
sequence number.

This remains an intermediate implementation: a structured Cozo query planner
should own all supported relational operations and preserve the same bound
semantics before the backend is considered production-ready.

Transformations involving arbitrary expressions, file conversion, chart
rendering and advanced statistical methods remain compute- or sandbox-backed
until they have a safe backend-specific query plan. A tool must not accept an
operation in its schema unless its result semantics are implemented; unsupported
operations should be rejected explicitly.

The first diagnostic/data implementations are intentionally bounded. CSV
validation currently supports `not_null` and `unique`, compiler diagnostics parse
existing output, formatter diagnostics inspect formatter output, and include-graph
diagnostics consume normalized `source -> include` lines. Symbol and reference
tools accept a host-owned semantic callback, for example clangd/LSP or a project
index; when absent they return `backend: text-fallback` and `semantic: false`.
`diagnostics.call_hierarchy` is stricter because a text match cannot provide a
meaningful caller/callee relation: it returns an explicit provider-unavailable
result until a semantic provider is bound.
Test-failure analysis groups normalized messages and reports a bounded
classification, count and up to three examples per group. These tools do not
invoke an unbounded compiler, formatter or shell command themselves.

Diagnostics scope is project-oriented. A future semantic index should be keyed
by `project_id`, repository revision, compile-commands hash, toolchain id and
index version. Session and research state may retain query results or evidence
references, but should not become the canonical source for project symbols.
Clang/clangd is an optional host capability: its absence does not disable the
diagnostic contracts, but limits results to the explicit fallback backend until
a semantic provider is configured.

### Optional clang tool support

Clang-backed agent tooling is opt-in at build time:

```text
-DLLAMA_AGENT_RUNTIME=ON
-DLLAMA_AGENT_TOOLS_CLANG=ON
```

When enabled, CMake searches for both `clang` and `clangd` and defines
`LLAMA_AGENT_TOOLS_USE_CLANG` for the agent target. Missing executables produce
configuration warnings rather than breaking the build. Runtime host setup must
still provide the executable paths and a project `compile_commands.json` when a
semantic provider is bound.

The build flag only includes the optional integration. The host/runtime remains
responsible for selecting `auto`, `text` or `clangd` behavior and for exposing
the provider through the existing semantic diagnostics callback. Without the
flag, the current text fallback and all existing tool contracts remain
available. With the flag but without a usable `clangd`, symbol and reference
tools continue to report the explicit non-semantic fallback.

`clang` is suitable for compiler and dependency operations; `clangd` is the
preferred provider for symbol definitions, references and call hierarchy. The
provider integration now has a bounded JSON-RPC framing seam with
`Content-Length` parsing, fragmented-read support and request-id helpers. A
host-owned provider also normalizes `workspace/symbol`, references and call
hierarchy locations into the tool result shape. The daemon/CLI host assembly
now binds the provider when `diagnostics.semantic_backend` is explicitly
`clangd`; default `auto` retains the fallback until executable discovery is
made authoritative. Paths and process settings stay host-owned and never enter
model-facing tool arguments.

The host configuration shape for this selection is:

```json
{
  "diagnostics": {
    "semantic_backend": "auto",
    "clang_executable": "clang",
    "clangd_executable": "clangd",
    "compile_commands": "auto"
  }
}
```

`compile_commands: auto` resolves the database below the host repository root;
an explicit host-owned path may be used when the build directory is elsewhere.
These values are runtime/provider configuration and are never accepted from a
client tool request.
* Provider-backed tools are enabled only through host configuration and
  caller policy. Credentials, endpoints and transport details remain outside
  the model-facing schema.
* Orchestration-backed tools cannot bypass the same profile, capability,
  workspace, budget and confirmation rules that apply to a direct operation.

The runtime should resolve these boundaries before a session starts. A tool
may be absent when its required executor or provider is unavailable; the
client must not be able to add a tool by naming a backend or capability in a
turn request.

The backend-neutral sandbox runtime also has an explicit no-backend state.
When Docker, Kubernetes or another executor is not configured, sandbox-backed
tools are omitted from the effective model-visible tool view during host
startup. A directly validated sandbox request still returns
`sandbox.backend_unavailable`; it never falls back to an unsandboxed host
process. Docker and Kubernetes execution become visible only when selected by
host configuration. Kubernetes uses PVC-backed workspace materialization;
shared-volume materialization for remote clusters remains a later slice.

## Runtime context budgets

Prompt and observation budgets are configured by the host under
`runtime.context_budgets`. Defaults exist in the runtime code and can be
adjusted for the model and `runtime.context_size` used by the instance. Values
are character budgets unless stated otherwise:

```json
{
  "runtime": {
    "context_size": 8192,
    "n_predict": 256,
    "context_budgets": {
      "plan_chars": 4096,
      "step_chars": 2400,
      "tool_observation_chars": 8192,
      "input_resources_chars": 4096,
      "deliberate_input_resources_chars": 2400,
      "resource_chunk_max_bytes": 4096,
      "resource_chunk_overlap_bytes": 256,
      "memory_chars": 4096,
      "memory_per_item_chars": 1000,
      "overlay_chars": 1600,
      "overlay_per_item_chars": 240,
      "deliberate_memory_chars": 1800,
      "deliberate_memory_per_item_chars": 500,
      "deliberate_overlay_chars": 1200,
      "deliberate_overlay_per_item_chars": 300,
      "working_state": {
        "max_total_chars": 8192,
        "max_value_chars": 1024,
        "max_completed_steps": 64,
        "max_remaining_steps": 64,
        "max_constraints": 32,
        "max_open_questions": 32,
        "max_resource_refs": 32,
        "max_chunk_status": 64,
        "max_tool_results": 32
      }
    }
  },
  "limits": {
    "max_continuations": 2
  }
}
```

See also [`docs/examples/agent-runtime-context-budgets.json`](../examples/agent-runtime-context-budgets.json).

The budgets are used as follows:

| Field | Use | Limiting behavior |
| --- | --- | --- |
| `plan_chars`, `step_chars` | Plan and step context for the planner, draft and reflection | Bounded rendering with final truncation |
| `tool_observation_chars` | Tool results retained in plan observations | Inline results are truncated; resource/artifact references are retained separately |
| `input_resources_chars` | Host-approved input resources in the normal prompt | Truncation of the rendered resource catalog |
| `deliberate_input_resources_chars` | Input resources in deliberate reasoning | Truncation of the rendered resource catalog |
| `resource_chunk_max_bytes` | Maximum byte size of one controller-owned text chunk | Range planning uses bounded store reads and rejects zero values |
| `resource_chunk_overlap_bytes` | Byte overlap between adjacent chunks | Preserves boundary context; must be smaller than `resource_chunk_max_bytes` |
| `memory_chars`, `memory_per_item_chars` | Normal memory context | The memory renderer selects entries within total and per-entry budgets |
| `overlay_chars`, `overlay_per_item_chars` | Symbolic memory overlay | Stage-aware selection and compaction before rendering |
| `deliberate_memory_chars`, `deliberate_memory_per_item_chars` | Reasoning context | The memory renderer selects entries within the configured budgets |
| `deliberate_overlay_chars`, `deliberate_overlay_per_item_chars` | Deliberate symbolic overlay | Stage-aware selection and compaction before rendering |
| `context_budgets.working_state.*` | Bounded checkpoint projection used by internal continuation | Limits total/value characters and projected steps, constraints, questions, resources, chunks, and tool-result summaries |

`limits.max_continuations` controls the number of additional inference slices
the existing driver may run after a bounded generation limit. The default is
two; zero disables automatic continuation. It is an operation limit, not a
second queue or session, and the host rejects values above 16.

`context_size` is the model token context and should be configured separately
from the character budgets. The runtime still applies hard host-owned limits
for transport, resources, sandbox requests and result sizes; these budgets do
not allow a client to exceed those limits.

When `runtime.context_size` is available, the runtime also performs a
conservative pre-inference estimate over the active request, plan, observations,
and resource descriptors. Output, tool, and safety reserves are applied before
the model call. `compact_recommended` is recorded as bounded runtime guidance;
`compact_required` and `continuation_required` stop before draft inference and
use the existing continuation/checkpoint path. The estimate is intentionally
not tokenizer-precise and does not claim that full conversation compaction is
implemented.

## Thinking-mode escalation

The requested thinking mode is not necessarily the final mode for a turn. A
host-owned deterministic escalation policy can resolve a bounded upward
transition when the request contains signals such as multiple independent
constraints, external uncertainty, user-resource comparison, or an explicit
verification request:

```text
requested reflective
    -> observed constraints/signals
    -> host escalation policy
    -> resolved deliberate or research
```

Escalation is monotonic and bounded. The policy controls whether escalation is
allowed, the maximum mode and the maximum number of transitions. The runtime
emits `thinking_mode_resolved`, `thinking_escalation_allowed`, and
`thinking_escalation_denied` events with the reason code. The model does not
select the final mode. `direct` remains a chat behavior and is not part of the
agent escalation chain.

Escalation can happen at two points in the same turn. Pre-run inspection is
the primary path for signals visible in the request, resources and objective.
After a draft, reflection may return a typed `assurance_action` requesting
`escalate_deliberate` or `escalate_research`. The runtime submits that request
to the same host-resolved policy; reflection does not change the mode itself.
The provisional draft and reflection issues remain turn-local working context,
not evidence, and the late transition is bounded to one use in the first
version. The existing turn, plan, cancellation identity and event stream are
preserved.

## Event order by thinking mode

The modes share one event model. A mode changes which events are expected and
how many bounded loops may occur; it does not create a separate event channel.
Events below are the logical order. Events in square brackets are conditional,
and `*` means that the event may repeat within its configured budget.

### Reflective

```text
thinking_mode_resolved
  -> [plan_created]
  -> [tool_executed / observation_recorded]*
  -> reflection_completed
  -> [thinking_escalation_requested -> thinking_escalation_allowed
      -> thinking_mode_resolved]*
  -> [response_revised]
  -> terminal response
```

Reflective may use a plan and tools, but it does not require plan revision,
step review, research gaps or source cross-checking. The terminal response is
returned separately from the event stream by daemon/MCP transports.

### Deliberate

```text
thinking_mode_resolved
  -> plan_created
  -> [plan_updated / plan_step_started / plan_step_completed]*
  -> reflection_completed
  -> [thinking_escalation_requested -> thinking_escalation_allowed
      -> thinking_mode_resolved]*
  -> step_reviewed*
  -> answer_reviewed
  -> [plan_revision_requested -> plan_updated]*
  -> [response_revised]
  -> terminal response
```

Deliberate always has a plan and bounded step/answer review. A plan revision
re-enters the same plan and event stream; it does not start a second runtime.

### Research

```text
thinking_mode_resolved
  -> plan_created
  -> research_started (with plan_id)
  -> research_gap_opened*
  -> research_task_scheduled -> research_task_started
  -> research_task_completed
  -> research_source_recorded* / research_evidence_recorded*
  -> [research_sources_compared]
  -> research_completed
  -> answer_reviewed (research verification)
  -> [research_reopened -> research_completed -> answer_reviewed]*
  -> reflection_completed
  -> step_reviewed*
  -> answer_reviewed (deliberate answer review)
  -> [response_revised]
  -> terminal response
```

Research tasks remain in the ephemeral research workspace. The plan owns the
overall research step and identity; the controller owns gap/task progression;
the workspace owns sources and evidence. If answer verification requests more
evidence, the bounded reopen emits `research_reopened` and may run one further
research iteration before draft verification resumes.

The late-escalation event sequence is part of the ordinary mode order, not a
second turn:

```text
reflection_completed
  -> thinking_escalation_requested
  -> thinking_escalation_allowed | thinking_escalation_denied
  -> [thinking_mode_resolved]
  -> continue in the same turn
```

When policy denies the request, the runtime returns the provisional draft with
the bounded limitation when no further revision budget remains. When allowed,
the higher mode receives the remaining turn budget; it does not reset the
turn's deadline, cancellation state or escalation count.

### Continuation slices and checkpoint transport

Completion limits are handled as bounded continuation slices inside one
driver operation. After a slice reaches the model limit, the runtime resumes
the existing plan only when the continuation budget allows it; otherwise it
returns a host-owned checkpoint. The checkpoint is execution state for the
current turn, not a memory candidate or a second plan. It carries the request
and turn identity, plan revision, active step, next action, sequence and
optional resource references so a later resume can validate that it is not
forking stale work.

The host-owned execution-control object is checked before and after every
slice. Cancellation and deadline expiry therefore stop before the next
continuation slice, preserve the already aggregated response, and return a
cancelled result with the original stop reason. The continuation prompt is
also restored on this early path. The same control object is propagated from
the session-host request through the resident runtime and runtime host into
the driver; no parallel cancellation mechanism is introduced.

Session-host results attach the externally accepted request and turn identity
to a checkpoint and validate it before returning it. The daemon response
protocol now serializes the bounded checkpoint projection, including resource
references, and the JSONL client parser preserves it as typed state. Daemon
events remain a separate channel: the checkpoint belongs to the terminal turn
result and is not emitted as an ordinary progress event. A future resume must
match request/turn identity and the current plan revision before execution is
allowed to continue.

Research uses the same ownership boundary. An incomplete research phase can
produce a `common_agent_research_workspace_checkpoint` containing the existing
workspace, its scope/turn/plan identity, sequence, gaps, tasks, sources,
evidence, comparisons and coverage counters. It is validated against the
workspace's existing budgets and identity before it is exposed on the agent
result. This is an active-operation checkpoint only: it is not written to the
memory store, does not create a second research queue, and does not make the
turn-scoped research workspace durable by itself.

## Current Shape

The core runtime remains in-process, but it already has asynchronous internal
seams for inference tasks, selected native/MCP tool calls, and pending
operations. The public `run_turn()` session-host API still blocks until its
lane reaches a terminal result, so the current design is not yet a fully
non-blocking or coroutine-based resident runtime. The daemon can queue and
dispatch commands through a bounded worker pool, while the pending-operation
poll/cancel seam allows selected work to wait, resume, time out, or cancel
without changing the synchronous result contract.

```text
llama-agent CLI / llama-memory chat compatibility
        |
        v
CLI argument parsing and validation
        |
        v
CLI host adapter
        |
        v
runtime host
        |
        +--> inference session backend
        |       +--> local CLI llama generation
        |       +--> in-process server_context smoke backend
        |
        +--> chat runtime driver
        |
        +--> agent runtime driver
                +--> planner
                +--> scheduler
                +--> registered tools
                +--> reasoning / draft / reflection
                +--> memory learning
```

What exists today is a narrow foreground daemon, not a production service lifecycle. It speaks a minimal JSONL protocol over stdin/stdout and is intentionally narrow: one foreground process, keyed session routing for admin/test turns, a bounded configurable dispatcher worker pool (default one worker), explicit shutdown, and no detached lifetime management. The daemon suppresses routine info-level model logs in this admin/test path so stdout stays protocol-oriented, while stderr remains available for warnings and errors.

That foreground daemon now also has a first explicit lifecycle-state contract above the worker/queue slice. The current state model is still intentionally small, but `starting`, `ready`, `draining`, `stopping`, `stopped`, and `failed` are now named service states instead of being inferred only from scattered booleans and transport-local status shaping.

The JSONL administration path now has a first configuration-reload contract.
`reload_config` validates a complete host-config candidate and either applies
the bounded mutable fields for new operations or returns a structured
`config.reload.rejected` result with `restart_required` field paths. Model and
backend resources, stores, worker/queue sizing, runtime assembly, the active
tool profile, capability map and profile definitions remain restart-required.
MCP providers are diffed by stable ID and added, removed, or replaced for new
operations. Repository-root and provider changes do not alter the immutable
profile snapshot of an already-running operation. The current implementation
keeps reload local to JSONL administration and does not rebuild listeners or
already-running provider clients. When the daemon hosts inbound MCP HTTP, the
resolved provider catalog is replaced atomically for new HTTP requests as part
of the same provider reload.

Daemon configuration now travels through a shared immutable snapshot store.
Reload publishes a new snapshot instead of mutating the options object that
worker, session and HTTP paths may already be reading. Requests and resolved
tool views retain the snapshot they materialized for their own lifetime.

The same daemon path is now also a little less transport-shaped around status reporting. Readiness/liveness, queue state, active request identity, and session descriptors now sit behind one daemon-status object first, and the current JSONL protocol mainly serializes that host-owned status surface rather than inventing it inline.

That status surface now also preserves one small but important bit of lane-owned waiting state: when an active turn is parked behind a manager-owned pending operation, daemon status can report not only the active phase/disposition but also the pending operation kind/detail on both the top-level active turn and the keyed session descriptor. In practice that means admin/test callers can now distinguish "turn is awaiting inference" from "turn is awaiting inference because a specific pending lane operation is still unresolved" without scraping internal debug logs.

The response side is now slightly less string-shaped too. The daemon command result carries an explicit response kind for `turn`, `status`, or lifecycle/session actions, so the current JSONL adapter no longer has to infer the response shape only from ad hoc event-string conventions before serializing it.

The command side is now also a little less ad hoc. The transport still speaks the same JSONL command fields, but the host-owned daemon command contract now carries small typed payloads for turn execution, session actions and queued-turn cancellation instead of relying only on a flat bag of optional top-level fields.

That command contract now also carries one shared execution-control seam end to end. Host config and daemon defaults still define the baseline timeout policy, but a JSONL/admin caller can now override turn timeout, inference-step timeout, native-tool timeout, and MCP connect/request/shutdown timeouts per turn without introducing a second cancellation/deadline mechanism. The same host-owned execution-control object continues to carry both the cancellation token and the resolved deadline.

The JSONL client/transport seam is now moving in the same direction. It no longer only has a special turn-request helper plus an ad hoc shutdown helper; it now has named request builders for ordinary turn execution, `status`, queued-turn cancellation, session actions and shutdown, so the current stdio/JSONL adapter is a little less likely to grow one-off inline command JSON as the service surface expands.

That same client/admin layer now also renders failed daemon turns through a small typed summary rather than only echoing raw error text. When a turn is rejected, cancelled, or times out, the current `daemon-chat` and `daemon-session` path can now surface the daemon `event` together with failure class, generation status, stop reason, and cancel state in one stable line.

The daemon ready event now advertises a small protocol version plus capability
list. That initial `ready` event confirms JSONL protocol availability; the
operational readiness gate is exposed through the subsequent `status` response
and its `readiness` object. Turn results also expose a few host-relevant
runtime signals such as runtime reuse, reflection/revision flags, event count
and memory-learning summary. That keeps admin/test clients from having to
infer runtime behavior from stderr.

That turn surface now also carries a slightly clearer failure/result contract. A failed or cancelled turn no longer only has `error` plus a boolean cancel flag; the session/daemon path now preserves a stable `failure_class` together with generation `status` and `stop_reason`, so timeout-versus-cancel-versus-generic execution failure is visible at the host boundary before any later async worker or richer event streaming work lands. The same shaping now applies to early daemon-side rejections too, so a missing turn payload, lifecycle rejection, or pre-expired deadline does not fall back to a thinner one-off error shape.

Turn results now also carry a first structured trace history. The current slice is intentionally modest: the trace is still a bounded execution summary rather than a streamed event protocol, but it already records host-safe facts such as plan creation/resume, step activation/completion, observation recording, tool success/failure, reflection decisions, memory-learning outcomes, and final response completion.

Trace and events deliberately have different responsibilities. The trace is
the ordered turn-local summary returned with the terminal result. The event
stream is the detailed lifecycle channel used by daemon, JSONL and MCP/SSE
clients. Research task, source and evidence milestones, deliberate step/answer
reviews, and thinking-mode escalation are therefore primarily events; the trace
keeps the bounded stage summary and its plan/step/tool/observation/workspace
references. Neither surface contains raw chain-of-thought.

Daemon command results still carry an internal daemon event list plus `daemon_event_count` inside the service/dispatcher result contract. The JSONL adapter now projects those events onto separate `message_type: "event"` lines and keeps the terminal `message_type: "response"` result free of event arrays and event counters.

That event list is now also a little less ad hoc internally. The daemon has started to grow a typed internal event stream of its own, with explicit event kinds such as `command.queued`, `command.started`, `command.rejected`, `turn.accepted`, `turn.started`, `turn.rejected`, `turn.cancel_requested`, `turn.cancel_rejected`, `turn.waiting_for_tool`, `turn.waiting_for_inference`, `turn.completed`, `turn.failed`, `turn.cancelled`, `tool.queued`, `tool.started`, `tool.progress`, `tool.output`, `tool.artifact_created`, `tool.completed`, `tool.failed`, `tool.cancelled`, `tool.timed_out`, `memory.learned`, `plan.created`, `plan.updated`, `plan.step_started`, `plan.step_completed`, `observation.recorded`, `resource.created`, `resource.attached`, `session.reset_requested`, `session.reset`, `session.close_requested`, `session.closed`, `lane.drained`, `resources.listed`, `memories.listed`, `plans.listed`, `resource.read`, `daemon.drain_requested`, and `daemon.shutdown_requested`. Direct daemon tool commands now emit `tool.queued` and `tool.started`, followed by safe output/artifact notifications and exactly one terminal event. Manager-owned asynchronous tool operations use the same event family and retain their `operation_id`; cancellation and deadline expiry are represented as `tool.cancelled` and `tool.timed_out`. The current JSONL/admin path still returns these as part of the ordinary command result, but they are no longer only invented at the transport edge: session lanes can now emit host-owned internal events into a daemon sink, and the dispatcher merges that stream back into the final response as one projection of daemon activity rather than treating the response object itself as the only event source.

The newest part of that projection is intentionally pragmatic rather than fully live-streamed from every runtime subsystem. Tool and memory events are still largely projected from the bounded turn trace/result surface, but planning/resource evidence has taken one step closer to the runtime core: the session host now preserves runtime-side agent events, and the daemon prefers those direct `plan.*`, `observation.recorded`, and `resource.*` signals before falling back to trace-only projection. Tool terminal outcomes are projected according to their trace kind, including failure. Step start/completion remains trace-driven for now. JSONL clients receive these as separate event messages while the terminal response remains result-only.

The next small cleanup inside that seam is now in place too: daemon execution now has an internal split between command outcome and event log before anything is projected back to the current wire/result shape. In practice that means the service layer can produce a host-owned outcome payload without also being the long-term owner of the projected JSONL response envelope, while the dispatcher remains the place where queued/started/internal daemon events are merged and then projected to today's `common_agent_daemon_command_result`. The transport still sees the same response surface, but the core no longer has to treat "response object with embedded event vector" as its native execution model.

That event projection is now also a little more internally consistent for fast service-owned replies. Status, drain, shutdown, resource/listing results, early turn-failure paths, and dispatcher-owned `command.queued` / `command.started` markers now preserve typed daemon event metadata directly instead of mixing the typed path with one-off raw string events for the same already-named event families. The same cleanup now also covers dispatcher-owned rejection and cancellation paths, so `command.rejected`, `turn.rejected`, `turn.cancel_requested`, `turn.cancel_rejected`, and queued-turn `turn.cancelled` no longer depend on raw string-only shaping at the final result seam. Service-owned admin/listing/resource failure paths have now moved the same way too, so `sessions.listed`, `session.found`, `session.not_found`, `session.lookup_failed`, `resources.list_failed`, `memories.list_failed`, `plans.list_failed`, `resource.not_found`, `resource.read_failed`, `session.reset_failed`, `session.close_failed`, and the fallback `command.failed` path all preserve typed event metadata before JSONL/client projection.

That seam now also has a small event-emitter shape rather than only ad hoc `push_back(...)` or four-argument sink calls. The daemon event sink now takes a full event object, and session/service code can stamp request, turn, session, namespace, project, and optional operation context through a small `common_agent_event_emitter` before emitting typed events. The JSONL adapter now keeps this event channel separate from the terminal response, giving the host one clearer append path for lane/session/service events and a better place to centralize sequencing and later fan-out without forcing transport logic back into runtime code.

That event seam now has its first explicit collector as well. The daemon service no longer owns raw `pending_events`, `next_event_sequence`, and event-buffer locking directly; that buffering/sequencing slice has been extracted into a small injected `common_agent_daemon_event_collector`. This is still intentionally a modest step: the service API continues to expose `emit_internal_event(...)` and `take_internal_events()` so the rest of the daemon does not need a wider rewrite just to gain a cleaner event seam.

The intended follow-up is to move ownership of that collector upward, closer to dispatcher/service coordination rather than keeping it permanently inside the service executor. The reason is architectural rather than stylistic: sequencing, buffering, and projection are daemon concerns, while `common_agent_daemon_service` should trend toward "execute one command against runtime/session state" rather than also being an agent event store. Keeping Step A small lets the code move in that direction without forcing transport, session-manager, or JSONL protocol changes first.

The JSONL/client side now mirrors that a little better as well. The parser no longer treats `events` as opaque leftover payload inside only a few response types; turn, status, lifecycle, listing, resource and event responses now all expose parsed `daemon_event_count` plus a small typed event-entry list with `type`, `event_type`, `sequence`, request/turn ids and detail when present. That keeps future client/admin work from having to re-open raw JSON just to inspect the daemon event surface.

The daemon can now open the same store backends as the CLI path. In addition to the default in-memory stores, a build with Cozo support can use `--backend cozo --memory-db PATH`, `--plan-backend cozo --plan-db PATH` and `--data-backend cozo --data-db PATH`. The data store is a separate host-owned Cozo database for structured rows and data-tool queries.

There is now also a first shared host-config slice above those flags. The foreground daemon and the real MCP stdio server can both accept `--config PATH` and load one small JSON host-owned configuration model for model/backend settings, runtime defaults, stores, resources, tool profile, MCP subprocess providers, and a few coarse limits. CLI flags still exist and still matter, but this is the first path where daemon/runtime construction does not have to start from a full CLI-style `args` object.

The resource-store slice now follows the same host-owned backend pattern. The CLI and daemon argument surfaces accept:

- `--resource-blob-backend auto|in-memory|fs|s3`
- `--resource-blob-root PATH`
- `--resource-metadata-backend auto|in-memory|cozo`
- `--resource-metadata-db PATH`
- `--resource PATH` (repeatable for bounded reference resources in an agent run)
- `--resource-mime-type MIME` (optional explicit media type for all supplied resources)

The current implementation supports `fs` and `in-memory` for blob storage, and `in-memory` and `cozo` for metadata. `s3` remains deferred. In the current default shape, blob storage resolves to `fs` and derives a default root if one is not supplied, while metadata resolves to `cozo` when a metadata DB path is present and otherwise stays `in-memory`.

The CLI can now import multiple bounded binary or text resources with repeated `--resource PATH` arguments. The host validates all requested files before writing any of them, limits each file to 1 MiB, stores them as turn-scoped resources, and attaches them to `common_agent_request::input_resources` as optional read-only references. File-extension matching is case-insensitive; PDF, PNG, JPEG, Markdown, JSON, CSV, HTML, and XML have explicit defaults. `--resource-mime-type MIME` overrides the inferred type for every supplied file, for example `--resource report.bin --resource-mime-type application/pdf`. A reference is not a primary source merely because it is required; primary-source authority is explicit in the input-resource role. The host does not expose the original local path to the agent. This requires the agent runtime and is consumed by the research workspace as `user_supplied` sources. Resource URIs are identifiers, not content hashes; the research source leaves its content-hash field empty until a real digest is available in the resource contract. Imported bytes now flow through the same authoritative resource store used by processors, so a supplied PDF can subsequently be read as text through the host-owned PDF processor without a parallel upload path.

Daemon CLI imports use the same text-only policy through the JSONL `put_resource` command. The returned resource URIs are forwarded in `resource_refs` on `run_turn`; daemon-session keeps session-scoped references for later prompts. JSONL/admin clients can also call `put_resource`, list scoped descriptors, and read bounded text with `read_resource`. These operations expose resource metadata and content, not arbitrary workspace or sandbox files.

### Resource lineage and bounded ranges

Large working material remains owned by the resource store. The original
resource is the source of truth; a derived chunk is only a bounded view with
explicit lineage. The shared resource contract therefore allows a chunk to
carry its parent URI, zero-based chunk index/count, byte offset/length,
overlap, and a host-owned derivation label. Lineage is descriptive metadata,
not a second content store and not a memory record.

`resource_read` accepts an optional non-negative `offset` together with the
existing bounded `max_bytes` limit. The host validates both values, enforces
resource authority before reading, and never expands the caller's profile or
scope. A range read may return fewer bytes at the end of a resource. An
out-of-bounds offset is rejected. Backends that do not implement non-zero
ranges fail explicitly rather than silently loading the whole payload.

The read contract is representation-aware without exposing converters to the
model. `resource_read` accepts an optional bounded `representation` field. When
it is omitted, the host defaults to and prefers `text`;
the current host implementation supports `text` only when the resource media
type is text-like, such as `text/*`, JSON, XML, YAML, or structured `+json`
and `+xml` types. Opaque binary resources can still be persisted by the store,
but they do not automatically expose a text representation. When the runtime
binds the host-owned processing provider, a text request for a supported
non-text resource can materialize a derived text resource through the existing
processing service and cache, then read that derived resource with the same
bounded offset and `max_bytes` rules. The returned resource reference and
provenance identify the derived representation; the original resource remains
authoritative. If no provider is bound, or no processor can satisfy the
request, `resource_read(representation="text")` still fails closed.
`resource_inspect` provides the descriptor and the host-resolved
`available_representations` list before a read is chosen. This integration is
synchronous within the existing tool-call/session lane; it does not create a
second queue or scheduler.
These are resource-domain operations in the existing tool catalog: processor
selection, MIME conversion, and execution isolation remain host-owned
infrastructure and are not model-selected tools. Binary and multimodal
representations can be added later without changing the chunking contract.

### Model-facing resource handles

The model-facing resource contract deliberately uses a short current-turn
handle instead of the authoritative resource URI. The rendered catalog exposes
bounded descriptors such as id=r1, name, and mime_type; it does not expose
local paths, blob keys, or storage URIs as the identifier the model must copy.
For example, the model-facing resource_read input is:

```json
{
  "type": "object",
  "required": ["id"],
  "properties": {
    "id": { "type": "string", "minLength": 1, "maxLength": 64 },
    "representation": { "type": "string", "enum": ["text", "bytes"] },
    "offset": { "type": "integer", "minimum": 0 },
    "max_bytes": { "type": "integer", "minimum": 1, "maximum": 32768 }
  }
}
```

The host maps r1 to the corresponding caller-owned resource before native
schema validation and execution. The internal execution contract continues to
use the canonical uri, so resource storage and provenance remain unchanged.
Unknown handles, handles from another turn, and malformed handles fail closed.
This is a bounded projection, not a second resource store or a persistent
identifier namespace. The same model-facing convention is used by
resource_inspect; other resource-domain tools may adopt it through the same
normalization seam when their contracts are updated.

### Compact tool notation

The runtime generates a compact, line-oriented model description from the
strict input and result schemas. This is a presentation projection only. It is
not a second tool contract and the model still returns JSON tool arguments.
The strict schema remains the source of truth for validation, bounds, enum
values, defaults, and host dispatch.

For example, the strict resource_read input schema is projected to:

    resource_read
    Read a bounded host-owned resource.
    args: id:string; representation?:text|bytes="text"; offset?:integer[0..1073741824]; max_bytes?:integer[1..32768]
    returns: resource, representation, content, content_encoding

The mapping is deterministic:

    strict schema property       compact notation
    required string id           id:string
    optional enum                representation?:text|bytes
    bounded integer              max_bytes?:integer[1..32768]
    default value                representation?:text|bytes="text"
    x-agent-type                 uses the semantic type name
    result object properties     returns: property1, property2

For a structured data tool the same renderer produces a compact description
such as:

    data.aggregate
    Aggregate dataset values.
    args: dataset:dataset_ref; measures:measure[]; group_by?:column[]
    returns: rows, dataset

The renderer currently handles bounded object schemas, required and optional
properties, scalar types, arrays, enums, defaults, numeric limits, semantic
x-agent-type labels, and result-property summaries. Unsupported complex schema
constructs use a bounded generic description rather than an invented
interpretation.

The model-facing and host-facing paths are deliberately separate:

    strict JSON schema
        -> compact model description
        -> model emits JSON arguments
        -> safe alias/default normalization
        -> strict schema validation
        -> host canonicalization and execution

For example:

    model JSON:       {"id":"r1","representation":"text"}
    normalized JSON:  {"uri":"agent-resource://resource/resource-1","representation":"text"}

The URI is introduced only after the current-turn handle has been resolved by
the host. MCP and native tool views use the same compact description renderer;
their original JSON schemas remain available as the parameters contract.
This keeps the presentation small for compact models without weakening the
runtime or MCP validation boundary.

The shared resource contract now also defines the first generic processing
boundary for future non-native representations. A processor receives a source
resource reference, the host-resolved media type, the requested representation,
optional page/range selection, and host-owned limits; it returns derived
staged outputs plus a bounded status or typed failure. The registry selects
processors deterministically by MIME type and representation. The concrete
host service then persists each staged output through the existing resource
store, applies the source scope/authority, records `resource_processor`
provenance, and returns ordinary resource references for later bounded reads.
This is deliberately infrastructure beneath the agent tools: a future PDF text
extractor, page renderer, OCR provider, or Office converter can sit behind the
same contract without exposing Ghostscript, MuPDF, OCR, Docker, or Kubernetes
as arbitrary model-selected tools. Derived outputs must retain lineage to the
authoritative original resource and then enter the existing bounded read and
chunking path.

The first `resource_read` integration uses a narrow common
`agent_resource_processing_provider` seam. The common tool adapter knows only
how to request a semantic representation; the concrete processing service in
`tools/agent/resource/` owns registry selection, MIME resolution, limits,
events, persistence, and cache reuse. This keeps the reusable tool contract
independent of the host implementation and leaves operation-bound processors
such as page rendering or OCR for the runtime assembly that can supply their
execution context.

The CLI host assembly installs local processor instances into the same
selection lifetime as the resource store and native tool view. The current
set includes PDF text and, when configured, Pandoc DOCX text processing.
The selection retains the processor instances, registry, and processing service
together, so `resource_read` receives a valid host provider for the whole
operation. Adding operation-bound processors later should extend this
host-owned assembly seam rather than create a second provider or queue.

The CLI also retains the imported resource store through the complete runtime
operation. The resolved native tool bindings keep non-owning store references,
while `common_agent_runtime_tooling::owned_resources` owns the operation-scoped
store after CLI resource import. This mirrors the daemon assembly and prevents
turn-scoped resources from disappearing between import and a later model tool
call. The document-table model-free smoke covers this ownership transfer.

Before processor selection, the host may resolve the resource media type from
declared metadata and a bounded content sample. The current deterministic
resolver recognizes strong PDF, PNG, JPEG, GIF, and ZIP signatures. Declared
text-like types are considered verified only when the sampled bytes contain no
NUL; other declared types remain available as unverified metadata unless a
signature resolves them more strongly. A pre-verified resolved type may be
reused, otherwise the processing service performs this bounded resolution
before consulting the processor registry. MIME detection is infrastructure,
not inference, and the original declared type remains available for audit and
provenance.

Concrete processor implementations belong under
`tools/agent/resource/processors/`. The parent `resource/` area owns the
format-independent resource store, MIME resolution, processing service,
lineage, limits, and chunking seams. The PDF text processor is the first local
implementation in that subdirectory. Future OCR, page-image, Office, HTML,
archive, or audio processors should use the same processor contract there;
they must not add format-specific branches to the planner, resource store, or
generic chunker.

Local document-processing backends are runtime capabilities, not build
dependencies. The agent binary must remain usable when MuPDF or Ghostscript
is absent. A host may provide an explicit executable path or allow the
operating system to resolve the executable through `PATH`; runtime capability
resolution then verifies that the selected process is usable. No CMake
discovery or link-time dependency is required. MuPDF and Ghostscript remain
processor implementation details and are not model-selected tools.

The host configuration can assign an execution policy per processor id below
`resources.processor_policies`. Supported modes are `local_preferred`,
`sandbox_preferred`, `local_required`, and `sandbox_required`; the backend is
`auto`, `local`, `docker`, or `kubernetes`. `executable` is an optional local
process name/path, `image` is an optional sandbox image, and
`expected_version` is an operator-visible compatibility expectation. These
values constrain host execution only. A runtime version mismatch must produce
a bounded warning/status event, and a `*_required` policy must fail closed
when its requirement cannot be met.

The host-side backend resolution seam applies an already host-ordered list of
representation-specific candidates to the verified capability snapshot. It
does not select MIME semantics, expose executables to the model, or create a
second queue. A typical page-image order is local MuPDF, local Ghostscript,
approved Docker, and approved Kubernetes. If none is available, processing
fails with a typed unavailable result; it is not silently converted into an
unbounded local command.

Local execution now reuses the existing `common_agent_sandbox_runtime` seam
through `common_agent_sandbox_local_runtime`, which delegates process creation
to the repository's `common_subproc` wrapper. The provider maps virtual
`/workspace/source`, `/workspace/writable` and `/workspace/artifacts` paths to
the already-created host workspace operation, applies bounded timeout and
combined-output limits, and returns the normal sandbox result status. This is
an execution foundation, not yet proof that a PDF renderer is installed or
that processor artifacts have been registered as derived resources. The
existing sandbox helper now also exposes this bounded raw result to host-owned
resource-processing adapters, without converting it into a model-facing tool
result or introducing another execution queue. The generic
`agent_resource_processing_host` seam now packages that adapter explicitly;
processor families can use it with their own typed command construction while
sharing workspace setup, policy validation, execution placement and bounded
result handling. Its artifact boundary accepts bounded local files from the
operation artifact directory and normalizes their MIME type; unsafe paths,
missing outputs and size-limit violations fail closed. Remote providers can
continue returning host-authorized resource references through the same result
contract. The first concrete hosted processor is the operation-bound
`pdf-page-image-v1`: it converts one completed page-image artifact into the
existing `agent_resource_processing_output`, after which the existing
processing service persists it with the normal source lineage and provenance.
The operation-bound construction is intentional for now; a host factory must
create the processor with the active operation context rather than sharing it
across concurrent session lanes.

The local MuPDF E2E smoke has now exercised the complete bounded path with the
repository PDF fixture: resource store, binary workspace staging, local
`common_subproc`, `E:\tools\mutool.exe`, PNG artifact collection, and derived
resource persistence. The renderer path is selected at runtime; MuPDF remains
an external executable and is not a build or link-time dependency.

The same E2E executable can run with the locally built
`llama-agent-pdf-ocr-worker:local` image using Docker or the Kubernetes Job
runtime. The image contains MuPDF, Pandoc and Tesseract so the E2E can exercise
PDF rendering, DOCX processing and OCR without changing the processor contract.
It is defined by `docker/agent/pdf-ocr-worker.Dockerfile`; it is a
test/development worker image, not a production registry reference.
Kubernetes E2E uses the active host policy image and keeps TLS verification
disabled only for the local Docker Desktop test context.

The current Kubernetes execution path is intentionally deployment-light: it
creates an ephemeral Job per operation and therefore does not require Helm.
The configured processor `image` is the image that contains the PDF worker and
its external processor, whereas Kubernetes `staging_image` is only the
host-owned helper used to materialize workspace and artifact data. For a
remote cluster, the processor image must be published to a registry visible to
that cluster and should normally be pinned by digest. The local
`llama-agent-pdf-worker:local` image is only suitable where the cluster shares
the local Docker image store, such as the Docker Desktop development setup.

A future Helm chart may package stable cluster installation concerns—RBAC,
service accounts, namespaces, storage defaults, NetworkPolicies, image
references, or a resident worker/controller—but it is not part of the current
processor contract or scheduling model. Adding Helm must not introduce a
second queue, scheduler, or processor lifecycle beside the existing host-owned
ephemeral Job path.

### Resource processor catalog and configuration

Resource processors are host-owned infrastructure. They transform an
authoritative resource into a bounded derived representation; they are not
entries in the model-visible tool catalog. The processor registry selects by
resolved MIME type and target representation, while the execution policy
selects where an external implementation may run.

The current processor catalog is intentionally small:

| Processor | Input | Representation | Status | Configurable processor arguments |
| --- | --- | --- | --- | --- |
| `pdf-text-local-v1` | `application/pdf` with a direct text layer | `text` | Implemented | No external command arguments; existing host limits apply |
| `pdf.page_image` | `application/pdf` | `page-image` | Command-contract foundation | Page, DPI, image format, colorspace and pixel/output limits |
| `tesseract-ocr-v1` | `image/*` | `text`, `hocr` or `tsv` | Implemented; local, Docker and Kubernetes E2E verified | Explicit language or `auto`, fallback language, OCR engine mode, page segmentation mode and output limits |
| `pandoc-docx-text-v1` | `application/vnd.openxmlformats-officedocument.wordprocessingml.document` | `text` | Implemented; local E2E verified when Pandoc is installed | Pandoc executable, plain-text output and bounded output bytes |
| `pandoc-docx-document-json-v1` | `application/vnd.openxmlformats-officedocument.wordprocessingml.document` | `document-json` with `application/json` target | Contract implemented; model-free command smoke covered | Pandoc executable, structured document JSON and bounded output bytes |
| `pandoc-markdown-docx-v1` | `text/markdown` | `docx` | Implemented; local E2E verified when Pandoc is installed | Pandoc executable, DOCX output and bounded output bytes |
| `pandoc-odt-markdown-v1` | `application/vnd.oasis.opendocument.text` | `text` with `text/markdown` target | Implemented; model-free command smoke covered | Pandoc executable, Markdown output and bounded output bytes |
| `pandoc-html-markdown-v1` | `text/html` | `text` with `text/markdown` target | Implemented; model-free command smoke covered | Pandoc executable, Markdown output and bounded output bytes |
| `pandoc-html-document-json-v1` | `text/html` | `document-json` with `application/json` target | Contract implemented; model-free command smoke covered | Pandoc executable, structured document JSON and bounded output bytes |
| `xlsx-workbook-json-v1` | `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet` | `workbook-json` with `application/json` target | Generic processor contract and bounded normalizer provided | Configured external XLSX normalizer, worksheet selection and bounded dataset materialization |

The first processor is a bounded local implementation used for the current
contract smoke. It is not a complete PDF parser: it does not render pages,
run OCR, select individual pages, or invoke an external executable. Its
processing request is bounded by `max_source_bytes`, `max_output_bytes`,
`max_generated_resources`, `max_duration_ms`, `max_pages` and
`max_page_bytes`; the resource service and store remain authoritative for
scope, reads and lineage.

External processor arguments must be introduced as typed processor options,
not as arbitrary command-line text. The `pdf.page_image` command-contract
foundation now validates bounded values such as `page`, `dpi`, `format`,
`colorspace`, `width`, `height` and `max_output_bytes`, and builds a
host-owned `common_agent_sandbox_request` for a selected local renderer. The
command smoke verifies MuPDF and Ghostscript argument construction, safe source
filenames, output limits and fail-closed validation without requiring either
executable to be installed. This is not yet actual page rendering: execution,
binary resource staging and sandbox-provider integration remain open. The
`tesseract-ocr-v1` maps bounded values such as `language`,
`fallback_language`, `oem`, `psm` and output format to an external Tesseract
executable. `language` may be explicit or `auto`. In auto mode, accepted
resource metadata is preferred (`resolved_language`, then
`declared_language`), followed by an explicit processor fallback. No language
is silently guessed when none is available. The derived resource records the
selected language, confidence and source; changing language metadata or typed
OCR options changes the cache identity and creates a new derived resource.
Tesseract is resolved at runtime and is not a build or link-time dependency.

The generic Pandoc processor currently has seven registered directions:
`pandoc-docx-text-v1` converts a validated Office Open XML document (`.docx`)
to a derived `text/plain` resource, while `pandoc-markdown-docx-v1` converts a
`text/markdown` resource to a derived DOCX artifact. `pandoc-odt-markdown-v1`
and `pandoc-html-markdown-v1` normalize OpenDocument Text and HTML resources to
derived `text/markdown` resources. `xlsx-workbook-json-v1` produces a
bounded structured JSON intermediate representation for a later host-owned
worksheet dataset importer; it is not itself a dataset. XLSX is deliberately not
routed through Pandoc: the installed Pandoc release supports XLSX output but not
XLSX input. The reference normalizer is `scripts/agent-xlsx-to-json.py`, invoked
through the generic external-process resource host. All directions use the same
processor contract, registry, processing service, resource store and lineage
rules. Pandoc is a runtime dependency: it is not discovered by CMake and is
not linked into the agent. The host may provide an executable name resolved
through `PATH`, or an explicit executable path.

XLSX normalizer implementation backlog
----------------------------------------

The current reference normalizer is `scripts/agent-xlsx-to-json.py`, which
requires Python in the selected processor runtime. This is intentionally an
implementation detail behind the generic processor contract, but it should
not become an implicit product-runtime requirement. Evaluate a compiled
worker or an already-supported host utility (for example a small Rust
`calamine` worker, a bounded C++ ZIP/XML implementation, or an approved
LibreOffice worker) before packaging XLSX ingestion for production. Any
replacement must emit the same bounded worksheet envelope and retain the
existing sandbox, resource-lineage and dataset-import seams.

`pandoc-docx-document-json-v1` and `pandoc-html-document-json-v1` produce a
structured Pandoc JSON representation as a derived `application/json` resource.
This is a document representation, not a dataset and not a model-visible
converter operation. It preserves a structural boundary for a later
document/table adapter: the document remains available for ordinary reading,
while simple table nodes may be projected into the existing dataset contract
with document origin metadata. The current scope does not infer complex table
semantics, extract tables from PDF, or materialize datasets directly during
Pandoc processing.

Processor selection now evaluates the existing boolean `supports` contract
through a structured support result containing eligibility, priority, lossiness
and sandbox requirements. The processing request also carries target MIME type
and purpose (`normalization`, `artifact_generation` or `preview`). This keeps
`DOCX -> text`, `ODT/HTML -> Markdown` and `Markdown -> DOCX` semantically distinct without creating a
second artifact registry. The processor invokes Pandoc with typed, fixed
arguments such as `--from`, `--to`, `--wrap=none` and a host-controlled output
path; arbitrary model-provided command-line arguments are never accepted.
Source and generated resources retain their parent lineage and processor
provenance.

The current DOCX ingestion representation is intentionally text-oriented. It is useful
for feeding the existing bounded reads and MIME-independent chunking path, but
plain text does not preserve all page layout, visual positioning, or complex
table structure. Legacy `.doc` and macro-enabled `.docm` inputs are not part
of this processor's current contract. Visual inspection and richer Office
representations remain separate future processors. The reverse Markdown-to-DOCX
direction is an artifact-generation path: its result is a bounded derived
resource suitable for download/export, not an authoritative replacement for
the Markdown source.

ODT and HTML normalization is also intentionally semantic-text oriented. The
derived Markdown is suitable for bounded reads, chunking and synthesis, but it
does not preserve every layout detail, stylesheet rule, embedded object or
complex table. HTML is treated as untrusted input: processor execution has no
network access by default, and scripts or external fetches are not model-visible
operations. The authoritative ODT or HTML resource remains unchanged.

The active execution-policy fields are representation-independent:

```json
{
  "resources": {
    "processor_policies": {
      "pdf.text": {
        "execution": "local_preferred",
        "backend": "auto",
        "executable": "mutool",
        "expected_version": ""
      },
      "pdf.page_image": {
        "execution": "sandbox_required",
        "backend": "kubernetes",
        "image": "registry.example/pdf-worker@sha256:replace-me",
        "expected_version": "mupdf-1.26"
      },
      "docx.text": {
        "execution": "local_preferred",
        "backend": "auto",
        "executable": "E:\\tools\\pandoc-3.10.1\\pandoc.exe",
        "expected_version": "pandoc 3.10.1"
      },
      "odt.text": {
        "execution": "local_preferred",
        "backend": "auto",
        "executable": "E:\\tools\\pandoc-3.10.1\\pandoc.exe",
        "expected_version": "pandoc 3.10.1"
      },
      "html.text": {
        "execution": "sandbox_preferred",
        "backend": "docker",
        "image": "registry.example/document-worker@sha256:replace-me",
        "expected_version": "pandoc 3.10.1"
      },
      "xlsx.workbook": {
        "execution": "local_preferred",
        "backend": "auto",
        "executable": "python",
        "expected_version": "Python 3.x; agent-xlsx-to-json.py"
      }
    }
  }
}
```

`execution` accepts `local_preferred`, `sandbox_preferred`,
`local_required` and `sandbox_required`. `backend` accepts `auto`, `local`,
`docker` and `kubernetes`. `executable` is an optional runtime executable
name or path; an empty value permits host `PATH` resolution. `image` selects a
host-approved sandbox image, and `expected_version` is an operator-visible
compatibility expectation. Version mismatches must produce bounded warning
and status information. Required policies fail closed when the selected
execution requirement is unavailable.

`resource_read` remains a semantic tool. For a configured sandbox-backed
representation, host assembly may install an operation-scoped processing
factory. Each read then creates the existing registry/service with the current
resource authority, operation id, workspace and sandbox host context. The
factory is short-lived for that read; it does not introduce a second queue,
store or scheduler, and the model still cannot select a processor executable.
The static provider path remains the fallback when no compatible explicit
processor policy and sandbox execution class are available. The current
operation-scoped assembly covers the CLI, daemon and MCP host seams for the
configured PDF page-image and Tesseract processor families; general automatic
backend resolution for all future processors remains open.

When `docx.text`, `odt.text`, `html.text` or `xlsx.workbook` is configured, the same operation-scoped host assembly can
install the local Pandoc processor without adding a model-visible `pandoc`
tool. `local_preferred` with `backend=auto` uses the configured executable or
host `PATH`; `local_required` fails closed if Pandoc cannot be started. Docker
and Kubernetes execution for these Pandoc directions use the same isolated worker image and
existing execution-provider contracts. The worker command remains the
host-typed `pandoc` invocation; the model does not select an executable. The
Pandoc sandbox paths are architecturally implemented, while live Docker and
Kubernetes execution remain environment-dependent assurance runs. Adding a
different worker image should reuse the same processor and execution-provider
contracts rather than add a DOCX-specific queue or scheduler.

The parameterized model-free E2E can be invoked as follows:

```text
llama-agent-docx-text-local-e2e-smoke.exe <pandoc-path> <docx-fixture> local
llama-agent-docx-text-local-e2e-smoke.exe pandoc <docx-fixture> docker llama-agent-pdf-ocr-worker:local
llama-agent-docx-text-local-e2e-smoke.exe pandoc <docx-fixture> kubernetes llama-agent-pdf-ocr-worker:local
```

The first form uses the local runtime; the latter two select the existing
Docker or Kubernetes sandbox runtime and run the configured Pandoc directions
through the same registry and resource service. The current E2E fixture covers
DOCX-to-text and Markdown-to-DOCX; ODT/HTML normalization is covered by the
deterministic processor contract smoke and can use the same parameterized
runner after adding representative fixtures.

The processor-specific options are host-owned typed options and must not become
model-controlled raw flags. Resource language metadata may be supplied
at ingestion or accepted through an existing host/policy path; model
suggestions are not authoritative writes. The original resource remains
authoritative, and the same processor contract is used whether the execution
provider is local, Docker or Kubernetes.

The processing service loads only a host-bounded source slice for a local
processor and enforces source, output-count, output-size, page-count, and
optional duration limits before returning derived resource references. The
generic resource tools expose `bytes` as a bounded base64 representation for
opaque or multimodal-friendly transport; `text` remains fail-closed for
binary media. Processing emits host-owned start, derived-resource, completed,
and processor-failed events through the existing agent event sink. Processing
is demand-driven at the representation boundary: the first PDF processor
extracts only its requested text representation and does not eagerly render
pages or invoke OCR. Completed derived representations are reused through the
existing resource store/catalog. The host records a deterministic processing
cache key in resource metadata containing source identity, resolved MIME type,
processor cache identity, representation, page/range and bounded limits. A
cache hit is accepted only for the same authority, processor provenance and
lineage parent; invalid entries are ignored and processing runs normally. This
is a completed-result cache, not a second store or queue. Concurrent
single-flight coordination remains a later activity.

The design policies for later chunk analysis are:

- Chunk creation is host/controller work in the existing session lane. It does
  not create a second agent, queue, or cancellation mechanism.
- Per-chunk observations retain resource references and bounded summaries.
  They are working evidence, not automatic long-term memory.
- Only reusable facts, decisions, constraints, or procedures may be proposed
  to memory through the existing memory policy.
- Chunk progress belongs in the existing plan/checkpoint state so cancellation,
  deadlines, continuation sequence, and plan revision checks remain unified.
- Full payloads remain recoverable from the original resource; summaries must
  retain references to that source instead of becoming the only authority.

Chunk observations and continuation state now share one small contract. A
`resource_chunk` observation must reference exactly one parent-linked resource;
native validation rejects malformed lineage, mixed parents, and duplicate
chunk indexes. When a completion checkpoint is created, the existing runtime
driver records the parent URI, total chunk count, and completed indexes in the
same checkpoint that already carries plan and resource state. Daemon JSON and
JSONL transport preserve those fields, so a resumed turn can continue the
same session-lane work without inventing a second chunk queue. The checkpoint
is progress state only: it does not promote chunk observations to memory and
does not make derived chunks a second source of truth.

The shared text chunker uses a deterministic boundary preference: paragraph,
table row, sentence, line, and finally a UTF-8-safe hard byte limit. It keeps
bounded byte offsets and optional overlap in each derived chunk. Table-aware
boundaries are intentionally row-based; a later format-specific adapter can
repeat table headers without changing the resource lineage contract. Fenced
code and richer document extraction remain follow-up adapters, while the
current text-only resource path never assumes that a newline alone is a
semantic boundary.

The resource-store integration plans chunk ranges with bounded
`read_text_range` calls. It retains only the parent descriptor and the range
metadata during planning; a later step reads one planned range at a time and
can render it as a parent-linked resource view. This deliberately avoids
creating persistent derived blobs or loading the complete source into the
session lane. The current implementation rejects non-text-oriented resources
and treats the original resource as the only source of truth.

After plan and blueprint selection, the agent runtime may perform this
preflight for oversized host-imported input resources. The resolved host
tooling supplies the resource-store pointer and scope; the runtime records a
bounded `resource_chunk_planned` observation in the existing plan and replaces
the model-facing input view with the first chunk. The original descriptor stays
authoritative in the resource store and plan evidence; it is not rendered a
second time beside the active chunk.

The existing session-lane driver then advances one chunk at a time. After a
slice returns, the driver records a `resource_chunk` observation containing a
bounded result summary and the parent-linked chunk reference. If another range
exists, it replaces the active input view and continues the same plan and lane.
There is no chunk-specific queue, worker, or cancellation path. Cancellation,
deadlines, continuation limits, and checkpoint creation therefore remain the
responsibility of the normal turn driver. The input-resource renderer exposes
the parent URI and byte range so the model can use the existing `resource_read`
tool contract to retrieve the active slice.

The runtime event stream distinguishes `resource_chunk_planned` from
`resource_chunk_processed`. The daemon projects these into typed
`resource.chunk_planned` and `resource.chunk_processed` events while the plan
observation remains the durable progress record. This keeps live tracing useful
without making the event stream a second progress store.

When the final bounded chunk has been processed, the controller restores the
original resource input and schedules one bounded synthesis slice in the same
session lane. An active parent-linked chunk is therefore never allowed to
complete the final-response plan step by itself. The synthesis slice is gated
by the ordered plan observations and retains the original resource as the
authoritative source. This is an internal continuation of the existing turn,
not a second daemon command or chunk queue.

If native validation finds conflicting parent-linked chunk observations, the
active synthesis step is blocked through the existing plan `block_step`
operation and the conflicting evidence remains in the plan for diagnosis. An
incomplete set is not presented as complete synthesis; it remains unfinished
until the required observations are available.

The current range implementation covers the in-memory and filesystem blob
backends. Persistent catalog lineage migration and controller-owned automatic
chunk scheduling remain follow-up slices; the current contract is designed so
those slices extend the same resource/store/session seams.

### Current resource chunking boundary

The current chunking path is deliberately bounded and text-oriented:

1. After plan or blueprint selection, the host-owned resource store reads
   bounded text windows using the configured `resource_chunk_max_bytes` and
   `resource_chunk_overlap_bytes` budgets.
2. The deterministic chunker prefers paragraph, table-row, sentence, and line
   boundaries before using a UTF-8-safe hard byte limit.
3. The original resource descriptor remains the source of truth. Each chunk is
   only a parent-linked range view carrying its byte offset, length, overlap,
   and chunk index.
4. The existing session lane exposes one active chunk at a time to the model.
   After the slice returns, the runtime records a bounded `resource_chunk`
   observation and advances the same plan to the next range.
5. Resume uses the plan's completed chunk observations and starts at the first
   missing index. Typed daemon events expose planning and processing progress,
   while the plan remains the durable progress record.

This is not full context management. General compaction of conversation
history, arbitrary tool output, and model responses remains a separate future
activity. The current resource path only ensures that large text resources can
be processed in bounded, resumable slices without creating a second scheduler
or source of truth.

The first context-pressure contract is also deliberately separate from the
resource chunker. A host-owned budget evaluator can classify a measured input
as `normal`, `compact_recommended`, `compact_required`, or
`continuation_required` after reserving output, tool, and safety space. It is a
deterministic measurement helper, not a new context store; runtime integration
must continue to use the existing plan, session lane, resource references, and
continuation checkpoint.

The checkpoint now also carries a bounded working-state projection derived
from the existing plan. It preserves the goal, current phase, completed and
pending step identifiers, constraints, unresolved assumptions, tool-result
summaries, chunk status, and opaque resource references. This projection is
working evidence for a later compaction/continuation turn; the plan and
resource stores remain authoritative, and no long-term memory write is made.
The projection has independent count and value-size limits for steps,
constraints, unresolved questions, resource references, chunk entries, and
tool-result summaries in addition to its total character budget. Identifiers
and references are therefore bounded by omission/count limits rather than by
allowing an unbounded list to escape the compact-state budget.

During context-pressure continuation, the driver stores this projection on
the request and the normal plan-context renderer emits the
`<compact_working_state>` block exactly once. The continuation instruction
does not inline a second copy. This keeps prompt ownership in the existing
CLI/runtime rendering seam and avoids spending context budget on duplicate
working state.

Context pressure uses the existing host/runtime boundary for token accounting.
The runtime keeps its conservative character-based estimate as a fallback, but
an assembled host may provide a `common_agent_context_token_estimator` through
the runtime configuration. When present, that estimator supplies the
template-aware or tokenizer-aware input count for the current request and
plan; an empty result falls back to the bounded estimate. This keeps exact
tokenizer ownership with the inference adapter without adding a second context
store or moving model-specific accounting into planning code.

Chunk entries in the compact projection retain the parent URI, position,
`status=completed`, and source observation ID. They are still bounded working
evidence; the original resource and plan observations remain authoritative.

One Windows-specific detail is now explicit in the build path as well. The local Cozo artifact used by this branch is currently a release-built MSVC library under `work/cozo-release`. When a Debug build enables Cozo-backed memory, plan, and resource support, the build now detects that release Cozo input and switches the current MSVC build tree to release-compatible CRT / iterator settings for that configuration. The scope is intentionally narrow: keep the resident agent, daemon, and MCP-host-facing targets buildable on this machine without requiring a separate locally-built debug Cozo package first.

That compatibility slice was re-verified on July 16, 2026 with a narrow serial build in `build-plan-resident-cozo-debug-3`. Instead of treating the whole workspace tree as the verification unit, the current practical bar on this laptop is the agent chain that actually exercises the resident/daemon/MCP path. The following targets built successfully after the Cozo/MSVC compatibility fix:

- `llama-agent.exe`
- `llama-agent-daemon.exe`
- `llama-agent-cli-mcp-selection-smoke.exe`
- `llama-agent-cli-run-mcp-smoke.exe`

The two CLI/MCP smoke binaries also ran successfully after rebuild. A full-tree debug build can still surface broader workspace concerns such as UI asset provisioning or unrelated test churn, so the current documented verification strategy for this branch is "narrow serial target build plus targeted smoke execution" unless a wider sweep is the explicit goal.

The resource path is also now split more cleanly internally:

- blob storage owns raw bytes and content-addressed persistence
- resource catalog owns descriptor/authority metadata and lookup
- the composed resource store binds those two responsibilities together for runtime and tool callers
- text operations remain compatibility wrappers over the byte-oriented store boundary

## Target Layout

The medium-term target is to separate durable domain contracts from host/runtime implementation more explicitly.

```text
common/
  memory/      what the agent can remember
  plan/        how the agent structures work
  resource/    how larger working material is referenced
  runtime/     host-neutral runtime DTOs and envelopes
  agent/       agent orchestration contracts above memory/plan/resource/tools

tools/
  agent/
    cli/       command-line entrypoints and CLI-only host adapters
    daemon/    daemon admin/client/jsonl transport and related host-side protocol code
    host/      host configuration and shared host-side policy/provider config
    mcp/       MCP client/server transport, stdio server, and shared MCP protocol helpers
    runtime/   resident runtime/session/inference assembly and host runtime plumbing
    tooling/   tool-provider, tool-runtime adapter, and host-owned tool-selection/result contracts
    resource/  resource-store implementation and host-owned resource plumbing
    ...        future CLI, daemon, MCP host/server implementation modules

pocs/
  archive/     older experiments and superseded slices
```

Short responsibility summary:

- `common/memory`: durable memory records, scopes, retrieval, and store contracts.
- `common/plan`: plan structures, state transitions, evidence links, and plan stores.
- `common/resource`: host-neutral resource references, authority descriptors, and later broader resource contracts for larger working material.
- `common/runtime`: neutral runtime-facing envelopes such as traces, resource refs, turn/result DTOs, and other contracts that should not be owned by one PoC host adapter.
- `common/agent`: agent orchestration contracts and logic that explain how memory, plan, resources, tools, and reasoning fit together.
- `tools/agent`: operational host code for running the agent as CLI, daemon, MCP host, or MCP server.
- `tools/agent/cli`: command-line entrypoints, selection/config parsing, and CLI-specific host adapters.
- `tools/agent/daemon`: daemon-facing transport, JSONL protocol shaping, lifecycle/event/dispatcher/service code, and daemon entrypoints.
- `tools/agent/host`: shared host configuration and provider-selection contracts used by daemon and MCP-facing host entrypoints.
- `tools/agent/mcp`: MCP client/server protocol code, stdio transport, and MCP-facing host/server entrypoints.
- `tools/agent/runtime`: resident runtime/session assembly, runtime host/session contracts, and in-process inference/runtime plumbing shared by CLI/daemon/MCP-facing hosts.
- `tools/agent/tooling`: host-owned tool provider/view code, tool-runtime adapters, and tool-related selection/result contracts shared by CLI/runtime/daemon/MCP hosts.
- `tools/agent/resource`: concrete resource-store implementations and resource runtime plumbing used by agent hosts.
- `pocs/archive`: retired or superseded experiments that are still worth keeping as reference.

The practical rule is simple: reusable contracts move downward; executable host assembly moves upward; old experiments move aside.

## Refactor Status and Migration Notes

The repository split described here was implemented in small, buildable
slices. The following is now the status of that migration rather than a
future work plan:

1. Target directories and their responsibilities are established.
2. Neutral runtime contracts live under `common/runtime`.
3. Host-owned resource-store implementations live under
   `tools/agent/resource`.
4. Active CLI, daemon, MCP, runtime and tooling hosts live under the
   corresponding `tools/agent/*` areas.
5. `common/agent` is kept for agent orchestration contracts and logic, while
   neutral DTOs remain in lower common layers.
6. `pocs/agent` is now primarily smoke assembly, helper binaries and
   migration-era build glue; it is no longer the active owner of the runtime.

The migration history is retained below because it explains why the current
layout looks the way it does. These are completed transitions, not parallel
implementation tracks:

- neutral runtime headers were moved under `common/runtime`
- resource-store implementations were moved under `tools/agent/resource`
- the temporary compatibility-header bridge was retired for the active path

The first correction kept `resource` itself as a domain contract. Traces and
generic runtime envelopes belong in `common/runtime`, while resource
references, authority and host-owned resource-store contracts fit under
`common/resource`.

The bounded daemon host slice is also complete: daemon client/admin helpers,
JSONL protocol shaping, lifecycle/event/collector code, dispatcher/service/
runtime assembly and the daemon entrypoint live under
`tools/agent/daemon`.

Host configuration and MCP-provider configuration were then moved toward
`tools/agent/host` so daemon and MCP-facing entrypoints share the same
host-owned configuration contracts.

The MCP host-facing cleanup is complete as well. Client/server protocol
helpers, stdio client/server support and the stdio MCP server entrypoint live
under `tools/agent/mcp`.

Resident runtime/session assembly, runtime host/session contracts and
server-context runtime plumbing now live under `tools/agent/runtime`.

The command-line entrypoints and adapters now live under
`tools/agent/cli`.

Tool-provider/view logic, tool-runtime adapters and host-owned tool
selection/result contracts now live under `tools/agent/tooling`.

That compatibility-header bridge has now effectively been retired for the active agent path. The branch no longer keeps a forwarder header layer under `pocs/agent`; active code, tests, and smoke binaries now include the concrete `tools/agent/...` locations directly. In practice that means `pocs/agent` is now much closer to its intended role in this phase: smoke harnesses, a few helper binaries, and migration-era build glue, rather than a second include tree pretending to own the runtime.

## Design Constraints

The runtime direction depends on keeping the layer boundaries boring and explicit.

- `common/memory` should not depend on agent/runtime host code.
- `common/plan` may depend on memory contracts and stores, but not on agent PoC host flow.
- `common/agent` may depend on plan and memory, but should still prefer neutral runtime-facing contracts when a type does not need full agent semantics.
- `common/runtime` should own host-neutral execution-control, pending-operation, trace, and other runtime DTO-style contracts that do not need full agent semantics.
- `tools/agent/*` is now the owner of active agent host/runtime behavior: resident runtime assembly, daemon/service code, MCP transport, tool providers, resource stores, and CLI adapters live there.
- `pocs/agent` may still depend on all lower layers, but it should now stay focused on smoke harnesses, fake backends, helper binaries, and migration-era build glue rather than regaining ownership of core runtime behavior.
- `pocs/memory` should not become the owner of agent orchestration or resident host flow.

In practice that means "almost production" shared types such as lightweight runtime DTOs, resource references, execution-control contracts, pending-operation descriptors, trace envelopes, or host/service contracts should move toward neutral common headers instead of being trapped inside one PoC adapter. The goal is to keep reusable contracts below the operational host layer, keep `tools/agent/*` responsible for concrete runtime behavior, and keep the PoC layer focused on smoke assembly rather than ownership of core abstractions.

The practical JSON rule is now the same: if JSON crosses a subsystem boundary and is not just a short-lived local implementation detail, it should move behind a named parse/serialize/validate helper. JSON as a wire or storage format is fine; raw `ordered_json` plus string `.dump()` should not silently become the contract.

The first concrete example of that constraint is now in place for tracing: the structured trace envelope lives in a neutral `common/runtime-trace.h` header, while `common/agent` populates it and `pocs/agent` only adapts or serializes it for CLI/daemon surfaces.

The same cleanup has now started for tool execution contracts. The lightweight tool-call and execution-result DTOs used by runtime-side tool validation/execution no longer need to live in the heavier native registry header; they now have their own smaller neutral contract header. The native registry still exists and still owns handlers, but the runtime-facing contract is starting to separate from the older registry-era shape.

The same direction has now been reinforced around the biggest JSON-heavy runtime edges:

- native tool payloads now serialize through named result-contract helpers instead of ad hoc JSON literals at each return site
- the older native chat-bridge path now also serializes tool success/failure payloads through named common-agent helpers instead of inlined JSON snippets
- the runtime core now also routes its remaining bounded observation/default JSON seams through named runtime-contract helpers instead of open-coding request-tool default stamping and observation payload `.dump()` calls in `agent-runtime.cpp`
- those runtime helpers now also expose one JSON-level safe-default shaping seam for tool arguments, plus a named reasoning-observation serializer, so tests and future adapters can validate the contract without going through a full runtime turn
- daemon JSONL request/response shaping now goes through explicit daemon protocol helpers
- the foreground daemon and its child-process client now also share named JSONL ready/event parsers instead of validating those protocol messages inline
- MCP stdio transport still owns framing, but JSON-RPC request/notification construction and tool result parsing now live behind extracted protocol helpers
- plan-step tool arguments now have a small named contract wrapper even though stored compatibility still remains `arguments_json`
- host config now has an explicit `schema_version`, a validator, and a roundtrip JSON helper instead of only a one-way parse path

That does not mean every JSON surface is now formalized. It means the highest-value runtime seams now have a named contract boundary, so later daemon/host/MCP work is less likely to hard-code behavior into scattered `.dump()` or `parse()` sites.

## Remote MCP and global inference scheduling

The resident-runtime foundation is followed by a separately bounded
remote-MCP/scheduler phase. Operation management, MCP schema validation, stdio
hard-timeout handling, dispatcher cancellation and the beta smoke pack are
already part of the current implementation.

The next branch adds the remote-provider configuration contract and documents
the requirements for Streamable HTTP, authentication, inbound MCP, global
session-lane scheduling, and inference microbatching. The detailed design and
acceptance criteria live in [agent-remote-mcp-scheduler-plan.md](agent-remote-mcp-scheduler-plan.md).

The key boundary is intentional: authenticated HTTP requests enter the existing
daemon dispatcher and session lanes; a future global scheduler controls only
inference capacity; batching remains an optimization behind the inference
backend. No transport or scheduler is allowed to become the owner of
conversation state, tool policy, resource authority, or operation lifecycle.

### Verification activity: beta smoke pack

This is the current verification activity after the transport/auth hardening.
It proves the existing seams end to end before a scheduler expansion.

Smoke scope:

- hanging MCP stdio server: deadline, subprocess termination, reader cleanup,
  and daemon survival;
- remote HTTP MCP: initialize, tools/list, tools/call, timeout and bounded
  result handling;
- inbound HTTP/TCP/Unix auth: valid/invalid credentials, scope binding,
  allowlist, read-only writes, admin commands and no credential leakage;
- session lanes: same-session ordering, multiple sessions, waiters, queued and
  active cancellation, reload and graceful shutdown;
- operation manager: deadline, poll, cancel, terminal state, events and reap.

### Backlog activity: lane-aware worker pool

The worker-pool design remains backlog work and must extend the existing
dispatcher without introducing a second runtime path. Its target shape is:

```text
request -> session lane mailbox -> ready-lane queue -> bounded worker pool
                                      ^                    |
                                      |                    v
                              pending completion <- one lane step
```

The first implementation must preserve these invariants: one active turn per
session lane, parallelism across different lanes, no worker held while waiting
for external I/O, fairness between ready lanes, prioritized control commands,
and explicit queue/capacity metrics. `worker_count` is scheduler concurrency,
not automatically GPU inference concurrency; inference capacity should remain a
separate backend/resource limit.

Not part of this activity: distributed scheduling, public network exposure,
full inference batching, or a second async runtime. Batching follows only after
the non-batched scheduler passes the concurrency and cancellation smokes.

The same direction has now started for host-owned resource references. The neutral `common/runtime-resource.h` contract no longer stops at lightweight resource refs; it also carries the first blob/resource store interfaces plus authority/descriptor DTOs. The current implementation now lives under `tools/agent/resource`, and it is no longer only an in-memory proof: resource blobs can now be stored on the filesystem through a content-addressed `fs` blob backend, and resource metadata can now be persisted through a first Cozo-backed metadata store. In the current default shape, resource blob storage prefers `fs`, while metadata remains `in-memory` unless a Cozo metadata database is selected explicitly or implied by `--resource-metadata-db`.

That contract is now also a little less ad hoc for tools. Native tool bindings no longer thread a raw `resource_store` plus separate namespace/session/project/turn fields through the host path. Instead they carry one scoped `agent_resource_runtime`, and helper functions derive read authority or stamp put-requests from that host-owned runtime scope.

That resource metadata is also starting to become more than a storage note. The current descriptor shape now has room for host-authored purpose, short content summary, usage hint, limitations, and lightweight semantic tags such as keywords or entity names. The intended meaning is practical rather than decorative: a resource row should answer what it is, why it was created, what it contains in short, how a later step can use it, and what its limits are.

The same contract-cleanup pattern now also covers two older JSON-heavy seams that were still carrying more history than necessary:

- plan tool-argument parsing/materialization now goes through the named `common_plan_tool_arguments_contract` path instead of reparsing and redumping ad hoc JSON inside `plan-bindings`
- the plan argument serializer now re-applies the safe integer normalization pass, so wrapped or legacy small-model shapes cannot leak `"limit":"2"` style control fields back out after normalization
- memory tool `search` and `remember` execution now parse JSON through named argument-contract helpers before business logic runs, instead of letting the service implementation read raw `ordered_json` directly
- those memory tool contracts can now also be parsed from an already-owned JSON value, so host/runtime adapters do not need to round-trip through a string just to validate or normalize bounded memory-tool arguments

That is still intentionally modest. The stored compatibility format remains `arguments_json` where older plan/runtime paths expect it, but the contract boundary is now explicit at parse/materialize/serialize time.

The same "almost boring" rule now applies to symbolic memory. The clean first fit is not a second symbolic-memory subsystem beside the current memory path; it is the existing host-owned memory model with a few more durable kinds and a later typed overlay above it. `procedure` already fills one symbolic role today, so the next incremental expansion is to let `constraint` and `decision` travel through the same scope, retrieval, proposal, provenance, MCP, and store contracts. Overlay compaction for planning, reflection, and reasoning should sit above that durable store rather than replace it.

That first overlay slice now exists in a deliberately narrow form. `common/memory` can render a compact symbolic overlay from already retrieved memories, grouping bounded `constraint`, `decision`, `procedure`, and a small amount of supporting `fact` context. The current planner, reasoning, draft, reflection, and memory-learning prompts still keep the older full memory context available, but they can now prepend this typed overlay so the model gets a cleaner symbolic summary without changing persistence, authority, or retrieval ownership.

The next small refinement is now also in place: overlay selection is stage-aware before rendering. The current selector is still deterministic and intentionally simple, but it no longer treats every prompt phase the same. Planning prefers `constraint` and `decision`; reasoning still favors `procedure` while allowing relevant symbolic guidance to follow; reflection emphasizes `constraint` and `decision`; and memory-learning can bias back toward `decision` plus reusable `procedure` context. That keeps retrieval and persistence unchanged while giving each runtime phase a cleaner symbolic slice from the same authorized memory set.

There is now also a first compacting pass inside that same host-owned rendering layer. It stays deliberately local to prompt shaping: before symbolic overlay or policy-pack text is rendered, repeated normalized items are deduplicated, oversized entries are trimmed to bounded per-item limits, and the retained set is capped by the existing section budgets. This is not a new store or a second compaction database. It is prompt-budget hygiene that fits the current model: keep durable memory unchanged, compact the rendered overlay/policy view when repetition or prompt pressure would otherwise waste context.

There is now also a narrow post-reflection hook for session-owned policy packs. It is intentionally conservative: if a turn actually ran reflection and the active session policy-pack still shows obvious duplication or section overflow according to the existing host-owned compaction heuristic, the session host compacts that pack and re-seeds the resident runtime with the compacted version for later turns. If there is no sign of duplication or pressure, nothing happens.

The next adjacent layer is now also explicit: a small host-owned policy-pack contract can be rendered ahead of the retrieval overlay. This follows the same broad shape as bootstrap blueprints and plan templates, but it is intentionally not a plan and not a second memory store. It is a compact declarative pack for purpose, goal, success criteria, constraints, decisions, and preferred procedures that a host or session can supply directly. The current prompt builders can derive a lightweight pack from caller-owned objective data or from the active plan's blueprint-like constraints, then prepend it before stage-selected symbolic memory.

That policy-pack seam is now present in the runtime/session contracts as well, not only in prompt rendering. Resident/session configuration can carry a stable pack across turns, the agent runtime path now preserves that host-owned pack instead of dropping it while rebuilding `common_agent_request`, and the daemon resident-request builder can seed one small session-level pack from host configuration. Request/objective and active-plan derivation still remain as fallbacks, but the host path no longer depends on those fallbacks alone.

The session layer now owns that seam a little more explicitly too. A session-host turn can provide a policy-pack override once, the host retains it as session state, later turns can omit it without losing the active pack, and daemon/session status can report the current `policy_pack_id` for diagnostics. The intent is still conservative: policy-pack identity is session state, not part of the resident model/inference reuse key.

## Layer Responsibilities

### CLI Adapter

The CLI remains responsible for local command-line concerns:

- Parse and validate `args`.
- Resolve profiles and defaults that are meaningful only to CLI users.
- Translate CLI backend flags and store paths into host-owned runtime/store configuration.
- Bootstrap, import, export, and blueprint package setup.
- Build the tool context for the selected profile and host-owned scope/policy.
- Retrieve memory context and render any CLI debug output.

The CLI should not own the agent loop. It should build runtime inputs and call the runtime host.

### CLI Host Adapter

`agent-cli-host-adapter` is the bridge between CLI state and the runtime host contract.

It currently owns argument-derived wiring that is still local to CLI behavior:

- Build chat and agent host inputs from CLI-owned state.
- Attach the post-run episode-recording hook.
- Resolve provider-backed tool exposure and execution for the selected tool profile.
- Print the final response and decoded-token summary.

This adapter is allowed to know about CLI `args`. The runtime/session host below it should not need to. The current daemon path now follows the same rule for policy/config assembly, even though its own option parsing is still local.

The older explicit memory-tool flags have now been removed from `llama-agent`. Agent-side memory tool exposure goes through the catalog/provider path only: the CLI chooses a tool profile, the host resolves a scoped `agent_tool_view`, and execution stays behind the same provider boundary used by the rest of the runtime.

`agent-cli-run` now also has a small adapter helper beside it. That helper owns CLI-only validation, default stamping, and agent/bootstrap/export setup so the top-level run function can stay focused on retrieval, tool wiring, and dispatch into the runtime host.

The CLI runtime and CLI selection paths now also share one small generation-helper utility for trace IDs, request envelopes, generation options, and failure formatting. That keeps the resident/runtime contract shaping in one place instead of duplicating it across two CLI-facing files.

The CLI tool path is now also shaped the same way. `agent-cli-run.cpp` no longer carries separate inlined assembly blocks for different tool wiring paths. A small CLI resolver now returns one `tools + tool_view + profile_tools_active` bundle, which lets the top-level run function stay focused on retrieval and runtime dispatch while the CLI adapter owns host-specific tool wiring.

### Runtime Host

The runtime host owns one prepared agent turn.

It is responsible for:

- Receiving already-built host inputs.
- Selecting and initializing the inference backend.
- Owning the runtime session lifecycle for the turn, including whether a prepared session is reused or reset after completion.
- Dispatching to chat mode or agent-planning mode.
- Running completion hooks and session reset policy.

The runtime host does not parse CLI arguments. It should remain small enough that a future resident host can provide equivalent inputs directly.

The host inputs now carry a CLI-free runtime turn request: request payload, scope, inference options, runtime policy, runtime config, orchestration config, generation options, and memory authority. The CLI adapter translates `args` into that request at the edge.

The host/runtime path now also carries one small tooling contract instead of threading separate `tools + profile_tools_active + tool_view` fields through each layer. That keeps the provider-facing shape more explicit: one host-owned tooling bundle contains the model-visible `common_chat_tool` list plus the resolved `agent_tool_view` used for execution.

That provider assembly is now a little less CLI-shaped internally as well. The CLI-facing resolver still exists, but it now sits on top of a smaller host-owned tool-selection request: resolved `agent_tool_context`, repository root, resource-store config, and optional MCP stdio provider command. The foreground daemon now uses that host-owned request directly instead of first synthesizing a temporary CLI `args` object just to reach the tool provider seam.

The provider/view seam has also grown a first narrowly scoped async capability. A resolved `agent_tool_view` can still be used exactly as before through ordinary synchronous `call(...)`, but it can now also advertise per-tool async support and expose a small `begin_call_async(...)` / `poll_call_async(...)` contract for tools that a host marks as async-capable in the resolved tool context. The current slice is intentionally host-owned and opt-in per tool rather than a global mode switch: one tool can remain synchronous while another is allowed to run behind a pending operation, and the existing model-visible tool schema does not need to change first.

Pending work now also has a neutral central registry in `common/runtime/runtime-operation.h`. The registry owns operation identity, deadlines, poll/cancel transitions, terminal state and cleanup; session lanes still own ordering and turn-phase decisions. This is deliberately an intermediate seam, not yet a global inference scheduler or GPU batcher.

The new host-config slice is intentionally modest. It currently models:

- model backend/path and optional embedding model
- runtime defaults such as context size, `n_predict`, planning/reflection toggles, and trace/learning flags
- memory/plan/data store backend and path
- resource blob/metadata backend and path
- tool profile, repository root, and a list of configured MCP providers
- a few daemon-style limits such as queue capacity and max turn seconds

In the current slice, the daemon and the real MCP stdio server can both carry a list of enabled stdio MCP subprocess providers from that host-config path into the provider/view seam. The daemon also supports outbound Streamable HTTP providers and inbound Streamable HTTP hosting. Outbound stdio and HTTP tool calls consume the host-owned cancellation/deadline state; broader streaming hardening remains follow-up work.

A thin resident-host wrapper now exists above this layer. It owns a runtime session and can run multiple turns against the same host contract without forcing session reset after each turn. That keeps the resident path small: it reuses the same runtime host and turn request instead of introducing a second agent loop.

There is now a small resident runtime layer on top of that wrapper. It owns the reusable resident host session plus the base runtime turn contract, and it can run either ordinary chat turns or agent planning turns against the same keepalive-backed model session. The thinner resident chat and agent helpers now delegate to that layer. Their job remains deliberately narrow: stamp per-turn prompt and turn identity onto the base request, run the turn, and in agent mode keep track of the active plan identity after completion.

The resident path also now has small builder contracts above the raw runtime types: one for constructing a base resident turn request from host-owned model/session/scope settings, one for constructing the resident runtime config itself, and one lightweight daemon-facing turn request/result shape. That keeps the first daemon step focused on process and transport concerns instead of rediscovering how to assemble runtime state.

There is now also a small generic resident session host above that builder layer. It owns the reusable resident runtime plus one small host-owned runtime-reuse key for session/scope matching, and can execute repeated chat or agent turns from a prepared host contract without carrying several parallel `active_*` identity fields.

The `server-context` resident backend now also has its own extracted host layer instead of being assembled inline inside the generic runtime assembly file. That layer still uses the current coarse `server_context` API, but it now owns the backend-specific load key, context key, host config, derived load params, loop lifetime and inference-session construction in one place. It also now distinguishes host configuration from the active running instance that owns the live `server_context`, derived params and loop thread. The active host now materializes the inference session directly from that running instance instead of rebuilding backend-specific details out in the generic assembly layer. Runtime session ownership has also been tightened a little further: for the `server-context` backend, the long-lived resident host now sits in the session's loaded-model state, while the active inference-context state only owns the currently built inference session that uses that host. That gives the next model-versus-context lifetime split a more concrete home instead of leaving it buried inside generic assembly code.

On top of that sits the first explicit session manager for the daemon/admin path. It now keys resident session hosts by namespace and session, while the currently bound project and scope remain part of the runtime state inside that session host. That means the foreground daemon can now treat the resident lane as `session A`, `session B`, `session A again` without baking project ownership directly into the manager key, while still rebuilding the resident runtime if a session changes project or scope.

That manager has now also taken one small step toward an actor-like session shape. Each keyed session lane now owns its resident host plus a small turn-state record: queued-turn count, any currently active request/turn id plus phase/disposition, and the most recent completed/failed/cancelled turn phase plus disposition for diagnostics. The lane now also has a real internal mailbox plus a named turn-disposition seam, and lane draining advances the active turn through explicit internal phases rather than treating the whole lane step as one opaque call. A small but important follow-up in this slice is that the active-turn record now stays alive across those intermediate phase advances and is only released on a terminal disposition, so active-turn introspection and cancellation target the real in-flight lane instead of a dispatcher-local shadow. Processing still drains synchronously today, but the shape is now explicit enough that later async scheduling can introduce waiting states, multi-step advancement, and mailbox ownership without moving session bookkeeping back up into the daemon transport or dispatcher.

The lane state model has also become a little more specific. Instead of reporting only `running` for every non-idle active lane, the manager now distinguishes a plain active lane from `running_with_waiters`, where one turn is already in flight and at least one later turn is queued behind it in the same session mailbox. That is still a small first slice, but it gives the host one explicit “mailbox has pressure behind the active turn” state before any real async worker split exists.

That same slice also begins to separate command handling from event production a bit more cleanly. Queueing and command execution still end in one request/result response today, but the lane/session layer can now emit internal events without directly mutating the final daemon result object. The dispatcher collects those emitted events and attaches them when the command completes. That is still not a fully streamed event bus yet, but it is an important structural step: later async transports should be able to observe the same internal event flow without depending on a specific JSONL result envelope.

The mailbox itself is now a little more explicit too. Session-lane messages carry a stable message id plus completion/result state as their own object rather than being only a transient deque entry, and lane draining now targets a specific enqueued message instead of relying on a more implicit "push then empty the whole deque" shape. That seam is now also waitable: when a second turn arrives for a lane that is already running, it can wait on its own mailbox message completion instead of failing immediately with a lane-local bookkeeping error. The implementation is still intentionally modest and still drains synchronously, but it now looks more like an actor mailbox with one active drainer plus queued waiters than like a plain recursive helper call.

One small but important technical-debt cleanup landed in the same area: the lane message now owns its own completion result and error state instead of keeping raw pointers to caller-owned stack storage. The public `run_turn(...)` API still looks synchronous to current callers, but the internal mailbox object is now much safer to carry forward into later async scheduling or pending-operation completion without tying session state to the caller's stack lifetime.

The lane also now owns one explicit current-message pointer alongside the active turn record, and its own processing state is now a named lane-state rather than a bare boolean. That means active request/turn identity, queued-vs-running observation, and host-side cancellation all have one clearer owner inside the lane itself instead of reconstructing the same identity from several partially overlapping fields. It is still the same single-lane synchronous flow, but the ownership model is now closer to "mailbox message plus lane execution state" and less like "deque entry plus a separately inferred active turn".

The turn-state seam itself is also a little more future-proof now. The active turn can carry an explicit pending-operation descriptor in addition to phase/disposition, and the disposition enum has reserved room for `wait_for_inference` and `wait_for_tool` rather than only terminal outcomes plus `continue_immediately`. That does not yet mean the full resident agent loop pauses and resumes around tool calls in production; it means the lane/runtime seam now has a neutral place to describe those waits without redesigning the state model later.

That seam now also emits an explicit "wait entered" internal event when a lane parks a turn behind either a manager-owned tool wait or the host-owned inference wait. The current slice keeps this intentionally narrow: it does not add a new external protocol surface, but it does give the host one clear event boundary between "turn started" and "turn resumed/completed" when a parked wait begins.

That lane-state is now visible in the session diagnostics path as well. Session descriptors and daemon status snapshots can report the lane's processing state directly, so operators and later scheduler code do not have to infer "idle versus running" only from a mix of queue length and active-turn fields.

The current lane-state list is still intentionally small, but it is no longer only a passive idle-versus-running flag:

- `idle`
  The lane has no current message in flight. It may still retain session/runtime state and last-turn diagnostics.
- `running`
  The lane currently owns one active mailbox message and is advancing its turn state machine. There may also be queued waiters behind it.
- `resetting`
  A host-side reset has taken ownership of the lane. New turns are rejected, queued waiters are completed with a reset error, and the current message is allowed to drain before the host reset clears lane-local runtime state.
- `closing`
  A host-side close has taken ownership of the lane. New turns are rejected, queued waiters are completed with a close error, and the current message is allowed to drain before the resident host is released and the lane is erased.

That means `reset_session` and `close_session` now go through the same lane-owned lifecycle surface as ordinary turns instead of directly clearing or erasing session state from the side. The implementation is still synchronous and intentionally narrow, but the ownership boundary is now much cleaner: lifecycle actions first claim the lane state, then deal with queued work, then wait for any current message to finish, and only then mutate or remove the resident session.

There are still a few obvious future candidates, but they are being deferred on purpose for now:

- `stopping`
  Still worth adding once daemon-wide drain/stop behavior starts flowing through the same lane/session model instead of staying mostly above it at service level.
- `failed`
  Deferred because terminal failure is currently better modeled as last-turn diagnostics plus lane return to `idle`. Making failure a lane-state now would blur "lane health" with "most recent turn outcome".
- `waiting_for_tool` / `waiting_for_inference` / `cancelling`
  These are better treated as turn phases or turn dispositions, not lane-states. The lane can remain `running` while the active turn moves through those finer-grained sub-states.

The foreground daemon entrypoint is now also split a little more cleanly. `agent-daemon.cpp` is mostly the process loop, while a small daemon adapter layer owns daemon-only argument parsing, store and host assembly, and JSONL request/response translation.

The practical daemon startup and JSONL examples now live in [agent-daemon-usage.md](agent-daemon-usage.md), with copyable host-config and request files under `docs/examples/`. This documents the current foreground lifecycle and flags without implying that the daemon is already a production service. Inbound MCP HTTP is now an available host transport; detached supervision, TLS termination and production operations remain separate concerns.

The daemon now also routes requests through explicit daemon commands plus a small daemon service layer. On top of that sits a bounded configurable dispatcher worker pool: stdin/JSON parsing still happens on the transport thread, while command execution runs through the queue before reaching the runtime service. The shape is intentionally modest: it separates transport from execution without yet introducing a richer async protocol or full end-to-end cancellation. Multiple workers can process different session lanes in parallel; one lane remains ordered. The important cleanup in the latest slice is ownership: the dispatcher no longer keeps its own parallel active-turn identity and cancellation handle. It now asks the service/session layer for the active request/turn descriptor and forwards active-turn cancellation back through that same session-owned seam.

The foreground JSONL transport loop is now also explicitly owned by the daemon adapter rather than living inline inside `agent-daemon.cpp`. That is still a small step, but it matters: `main` is now closer to pure process bootstrap plus environment wiring, while the JSONL request/response loop has a named adapter seam that later transports can mirror without reintroducing daemon lifecycle logic into the entrypoint.

That seam now has a concrete stream boundary as well. The stdin/stdout adapter
and the optional TCP adapter both use the same JSONL read/write loop and the
same foreground request parsing plus dispatcher execution path. TCP accepts
multiple authenticated connections, binds caller namespace/project fields to
the connection policy, and shares the configured worker pool. TCP framing
limits and listener binding are restart-required; TLS is intentionally left to
a sidecar or service mesh for this first container-oriented slice.

The same host can now use a POSIX Unix domain socket for local background
operation. This remains a foreground process from the operating-system point
of view: systemd or another supervisor owns restart and logging, while the
daemon owns its dispatcher, workers and graceful shutdown. Socket-file mode
provides a local owner/group boundary and the existing caller policy remains
the application-level authorization layer.

That adapter loop is now a little thinner too. The outer loop still owns stream framing and lifetime, but one small helper now owns the "parse one JSONL request, run one daemon command, serialize one response" path. It is still synchronous and intentionally modest, but later foreground/socket/pipe adapters now have a cleaner seam above raw stdio framing.

There is now also a first explicit foreground request/response contract above that helper. The adapter no longer treats "one foreground admin request" as only a transient local combination of parsed JSON plus immediate writeback. It now has a named host-owned foreground request/result seam that can later be reused by a socket/pipe/HTTP adapter without first inheriting the stdio loop structure itself.

The daemon adapter is now also slightly less CLI-shaped in its host construction path. It still uses the existing store-opening helpers, but it no longer has to synthesize a temporary full CLI `args` object just to build runtime policy, runtime config, orchestration config, or the resident session-host contract.

That cleanup now extends one step further down the daemon path. The daemon runtime no longer synthesizes temporary CLI-style `args` just to open stores, build resource-store config, or resolve provider-backed tooling. Memory/plan store selection now follows the same host-owned backend/path values directly, including the old `auto` resolution rules, while tooling resolution receives a host-built tool-selection request instead of a CLI object.

That service layer now understands a slightly broader host-oriented command surface:

- `run_turn`
- `cancel_turn`
- `status`
- `reset_session`
- `close_session`
- `shutdown`

`status` reports a narrow readiness/liveness snapshot plus the currently tracked session keys and queued-command count. It now also exposes a few small lifecycle signals from the dispatcher itself, such as whether the worker thread is running, whether the daemon is still accepting new commands, whether shutdown has been requested, and the current queue capacity. The top-level status snapshot now also carries active turn id plus phase/disposition, and the daemon-status object has started to keep that as one small transport-neutral active-turn snapshot internally instead of only as loose protocol fields. `reset_session` and `close_session` go through the same keyed session manager as ordinary turns, which gives the admin/test path an explicit place to manage resident session state before a fuller queued daemon lifecycle exists.

Session status is now slightly richer too. In addition to key, project, scope, and policy-pack identity, the daemon-side session descriptor can report queued-turn count plus active request/turn phase/disposition and last-turn phase/disposition details. The current JSONL admin path still uses this only as bounded diagnostics, but it gives later actor/scheduler work a stable place to expose per-session execution state without inventing a second status model beside the session manager itself.

That lifecycle surface is now also enforced inside the service itself. Once shutdown or draining has been requested, `run_turn` is rejected with a host-owned lifecycle error instead of relying only on the dispatcher's outer acceptance window. That closes the small gap where a late turn could otherwise slip in after shutdown had conceptually started but before the queue had fully stopped accepting work.

The dispatcher/protocol path now also carries a small status snapshot on non-status responses, including lifecycle replies such as `shutdown`. That means the current foreground/admin path can observe `draining` directly from the shutdown response instead of only inferring it later from booleans or stderr timing.

`cancel_turn` now exists as a first dispatcher-level contract, and the daemon result contract distinguishes two cases explicitly: queued-turn cancellation succeeds and emits a `turn.cancelled` daemon event, while cancellation against the currently active turn now records a host-owned cancel request against that turn's execution-control state and returns `turn_cancel_requested` plus the active request/turn identity. In the current slice that active-turn identity is resolved from the session manager rather than from dispatcher-local side state, which is a small but important cleanup for later async/session-mailbox work. The follow-up in this sweep is that cancel/lifecycle replies now go back through the same dispatcher status-snapshot filling path as ordinary status responses, so queue state and active-turn metadata are no longer partly ad hoc on the cancellation path. The CLI/admin status renderer now also shows active turn id plus phase/disposition directly in its one-line summary. This is still intentionally narrow. The current runtime/inference/tool stack does not yet have a full end-to-end safe abort path, but the daemon/session seam now has one shared place for cancellation identity, turn deadlines, and later timeout propagation.

One concrete "full current functionality" foreground run looks like this on Windows/PowerShell:

```powershell
@'
{
  "schema_version": 1,
  "model": {
    "backend": "server-context",
    "path": "C:\\Users\\kalld\\models\\Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
    "embedding_model": "C:\\Users\\kalld\\models\\nomic-embed-text-v1.5.Q4_K_M.gguf"
  },
  "runtime": {
    "default_mode": "agent",
    "thinking_mode": "auto",
    "max_reflection_rounds": 2,
    "max_plan_revisions": 3,
    "max_research_iterations": 4,
    "memory_learn": "post-turn",
    "agent_plan": "auto",
    "n_predict": 96
  },
  "stores": {
    "memory": {
      "backend": "cozo",
      "path": ".\\work\\agent-memory.cozo"
    },
    "plan": {
      "backend": "cozo",
      "path": ".\\work\\agent-plan.cozo"
    },
    "data": {
      "backend": "cozo",
      "path": ".\\work\\agent-data.cozo"
    }
  },
  "resources": {
    "blob_backend": "fs",
    "blob_root": ".\\work\\agent-resources",
    "metadata_backend": "cozo",
    "metadata_db": ".\\work\\agent-resources.cozo"
  },
  "tools": {
    "profile": "minimal",
    "repository_root": "C:\\Users\\kalld\\Documents\\Codex\\llama-dyn",
    "providers": [
      {
        "type": "mcp",
        "id": "local-mcp",
        "enabled": true,
        "transport": "stdio",
        "command": [
          ".\\build-agent-current-ninja-debug\\bin\\llama-agent-mcp-stdio-fake-server.exe"
        ],
        "prefix": "local",
        "server_name": "local"
      }
    ]
  },
  "limits": {
    "queue_capacity": 8,
    "max_tool_rounds": 2,
    "max_turn_seconds": 120,
    "turn_timeout_ms": 120000,
    "inference_step_timeout_ms": 30000,
    "tool_timeout_ms": 5000,
    "mcp_connect_timeout_ms": 3000,
    "mcp_request_timeout_ms": 10000,
    "mcp_shutdown_timeout_ms": 1000
  }
}
'@ | Set-Content .\work\agent-host.json -Encoding utf8

@(
  '{"request_id":"status-1","command":"status"}',
  '{"request_id":"turn-1","mode":"agent","prompt":"Plan how to inspect the repository tooling path.","session_id":"demo-session","namespace_id":"local","project_id":"llama-dyn","memory_scope":"project","plan_scope":"project"}',
  '{"request_id":"shutdown-1","command":"shutdown"}'
) | Set-Content .\work\agent-requests.jsonl -Encoding ascii

Get-Content .\work\agent-requests.jsonl |
  .\build-agent-current-ninja-debug\bin\llama-agent-daemon.exe --config .\work\agent-host.json
```

That example exercises the current end-to-end foreground daemon shape: resident `server-context` inference, Cozo-backed memory/plan/data stores, filesystem+Cozo resource storage, bounded research iterations with reflection and plan revisions, post-turn memory learning, repository/native tools, and one MCP stdio provider under the same host-owned config. The packaged full configuration uses these enabled settings so longer tests exercise the orchestration paths; the bounds keep a test turn finite.

The packaged full configuration selects the built-in `all-configured` tool profile because it is a test fixture rather than a production default. This makes the complete catalog available to model-free and integration tests, including repository, workspace, memory, plan, diagnostics, data, sandbox, artifact, resource and web tools. It should be reviewed or replaced with a narrower host-owned profile before production use. The file remains strict JSON; the rationale is documented here instead of using non-standard comments in the configuration file.

On top of that, the CLI now has two thin child-process adapters. `daemon-chat` starts the foreground daemon, sends one turn, reads one response, and shuts the child down. `daemon-session` keeps the same foreground child alive across multiple prompts in the same admin/test session. Both paths still go through the same runtime request/result contracts rather than delegating multi-turn state to a backend conversation loop, and the CLI reads protocol from stdout while relaying daemon diagnostics from stderr separately.

That CLI session path is now a little less turn-only as well. The child-process adapter has explicit request helpers for daemon `status`, `list_sessions`, `get_session`, `list_resources`, `read_resource`, `put_resource`, `list_memories`, `list_plans`, `drain`, `reset_session`, `close_session`, and `shutdown`, and `daemon-session` exposes a small admin/test command set over stdin: `/help`, `/status`, `/status --verbose`, `/sessions`, `/session`, `/resources`, `/resource-put <path>`, `/resource <uri>`, `/memories`, `/plans`, `/reset`, `/close`, `/drain`, and `/quit`. The implementation also normalizes Windows-style stdin a bit more carefully, including a first-line UTF-8 BOM edge that showed up in PowerShell piping during smoke verification.

The `/resources` command now renders one bounded descriptor line per registered resource, including its resource id, URI, display name, media type, byte size, and scope. The `/resource <uri>` command remains the explicit read/fetch operation and returns the bounded text content for an authorized URI. This distinction is intentional: resource-store entries are discoverable and fetchable through the resource contract, while ordinary repository/workspace files remain source-tree data accessed through workspace or repository tools. Sandbox-produced files only appear in the resource listing when the selected sandbox backend imports them as resource references; a file written directly into a source tree is not implicitly published as a downloadable artifact.

The JSONL/admin path still has an explicit `put_resource` command, but that surface remains text-oriented until its binary transport contract is extended. The interactive `/resource-put <path>` command likewise remains a text convenience wrapper. Daemon CLI `--resource PATH` currently uses the text JSONL path; the native agent CLI uses the byte-oriented import described above. Keeping these boundaries explicit avoids silently base64-encoding binary documents into a text-only protocol.

The CLI-side diagnostics seam is a little cleaner now too. The child-process adapter still reads daemon stderr separately from protocol stdout, but it suppresses the high-volume routine model/bootstrap chatter in the ordinary admin/test path and only forwards warnings, errors, and unexpected lines by default. That keeps `daemon-chat` and `daemon-session` usable as foreground integration tools without making the protocol consumer scrape through several hundred lines of model-loader noise. For deeper debugging, the adapter still becomes more permissive when agent tracing is enabled.

That foreground client path is now also slightly less ad hoc internally. It has a tiny JSONL transport wrapper around the child-process stdio pipes, and the admin/test command handlers no longer peel turn/session/status results straight out of raw `ordered_json` with repeated `response.value(...)` calls. Instead they consume a few small protocol-shaped result parsers for turn, status, and lifecycle/session events while still speaking the same external JSONL wire format.

The `daemon-session` admin/test surface is now a little friendlier too. Its
`/status` command renders one compact typed summary of lifecycle state, queue
health, active work, and bound sessions, while `/status --verbose` renders the
complete parsed JSONL status payload including readiness, providers, warnings,
sessions and metrics.

That rendering is now its own small CLI-facing seam rather than another helper hidden in the wire-protocol file. The JSONL parser still owns the transport/status DTOs, while the foreground client owns the tiny status-summary contract and rendering policy that turns those DTOs into a stable human-facing admin/test line.

The same split now exists for service-owned admin commands. The child-process session adapter still owns daemon process lifetime and stdio transport, but `status`, `list_sessions`, `get_session`, `list_resources`, `list_memories`, `list_plans`, `read_resource`, `drain`, `reset_session`, `close_session`, and `shutdown` now go through a small dedicated admin client layer above the JSONL request/response seam instead of being open-coded one by one inside the subprocess/session class.

The same client path now also uses the lifecycle snapshot actively instead of only carrying it through the protocol. `reset`, `close`, and `shutdown` now parse lifecycle responses through the richer DTO and use the embedded state snapshot for rendering and shutdown validation rather than treating those replies as event strings alone.

Those child-process adapters now also pass through the same daemon-owned tool configuration surface as the direct JSONL admin/test path: `--tool-profile`, `--repository-root`, and `--mcp-tool-command` all reach the foreground daemon when present. The chat-oriented daemon client path also now defaults its plan scope more conservatively when planning is off, so a simple session- or project-scoped admin/test chat turn does not accidentally force a synthetic turn-scoped contract.

The daemon-facing request shape now carries host-owned scope data such as namespace, session, project, memory scope and plan scope. The current session manager treats namespace plus session as the live resident lane, while status responses still report the currently bound project/scope for that lane. That is still intentionally modest: it is enough to drive multi-turn resident smoke and integration tests, while keeping the future service-owned session model explicit.

There is now also a focused smoke for that foreground client seam itself: `scripts/test-agent-daemon-client-clean-io-smoke.ps1` checks that `daemon-session` keeps protocol-oriented output on stdout, keeps routine loader/bootstrap chatter off stderr in the ordinary path, preserves the small admin/test command surface (`/help`, `/status`, `/status --verbose`, `/sessions`, `/session`, `/resources`, `/resource-put <path>`, `/memories`, `/plans`, `/resource <uri>`, `/reset`, `/close`, `/drain`, `/quit`), and renders the compact or verbose status form as requested. Separate protocol and client CTest smokes now exercise `put_resource`, resource listing, and resource reading against the JSONL seam without needing a live model.

Each keyed session host still manages one active resident runtime at a time and still matches reuse from the current host-owned session/scope contract. That is sufficient for the current foreground daemon and smoke coverage, but it is still an early manager shape rather than the final host/service session model.

The runtime surface is now split more explicitly in code as well:

- turn contracts
- resident runtime contracts
- session-host contracts
- host input/build contracts

The backend-specific inference-session selection now also lives with runtime-session initialization rather than in the generic runtime-assembly layer. That keeps runtime assembly focused on agent behavior wiring while session initialization owns backend choice and resident inference-session reuse.

That split is still structural rather than behavioral, but it makes the next service-facing step easier because the current single-active session host is no longer buried inside the same header as every other runtime layer.

Inside the resident runtime session, ownership is now also expressed a little more explicitly. The session keeps separate state for:

- loaded model ownership
- active inference-context ownership

In practice that means a resident session no longer presents one flat bag of `model + templates + inference session + reuse flags`. The first step toward a cleaner resident host is now visible in code: model-loading state and active inference-context state are separate sub-objects, runtime execution asks the session for its active inference session instead of reaching straight into one merged structure, and the `server-context` host itself now lives with the loaded-state side of that split rather than being hidden only inside the active session object.

### Runtime Drivers

The runtime drivers contain the agent behavior.

The chat driver handles ordinary chat generation plus bounded synchronous tool follow-up rounds. The agent runtime driver handles planning, step scheduling, registered tool execution, reasoning, draft synthesis, reflection, and memory learning.

These drivers should not know whether the caller was CLI, a resident process, or a future MCP-facing host.

### Inference Backend

`common_agent_inference` is the abstraction for model generation.

The current backends are:

- `cli`: local llama-backed generation using the existing CLI-style path.
- `server-context`: an in-process resident smoke backend using `server_context`.

The generation request/result contract is narrower than top-level CLI state. Requests carry purpose, trace metadata, scope, messages, tools, optional schema, and generation options. Results return content, decoded-token counts, status, stop reason, parser metadata, and errors in one shared envelope.

Today the runtime session can also be reused when the host keeps the same backend and inference options. The current CLI adapter still chooses to reset after each completed turn, but a resident host no longer needs a different core contract to keep the model session alive across turns.

The reuse logic is now split a little more explicitly inside the resident session as well. There is a small model-load key for properties that really affect model loading, and a separate inference-context key for properties that still require rebuilding the active inference session. In the current resident backends that means turn-shaped settings such as `n_predict` no longer look like model or context identity changes.

The current split is still a pragmatic first slice rather than the final shape. It is good enough for resident smoke and the foreground daemon, but the longer-term split should distinguish:

- model-load options
- context/inference-context options
- per-turn generation options

That split now exists structurally for both the CLI-backed and `server-context` resident sessions, and the daemon/admin path now treats `n_predict` as a turn-level override instead of a resident-runtime identity change. The `server-context` path is still coarser in a deeper sense: its current host object still combines model load and inference-context lifetime, so the next cleanup there is to separate those lifetimes more explicitly rather than just removing turn-shaped fields from the reuse key.

The resident `server-context` host no longer bakes `n_predict` into its own resident host config either. The decode limit is now only stamped onto per-turn server tasks, while the long-lived host keeps only load/context identity plus baseline runtime settings. That makes the reuse boundary line up better with the actual runtime behavior exercised by the resident and daemon `n_predict` reuse smokes.

### Stores and Scope

Memory and plan stores are runtime dependencies, not global singletons.

Scope values are caller-provided authority:

- namespace
- session
- project
- turn
- memory scope
- plan scope
- global-memory opt-in

The model cannot choose these values. A future server or MCP host must derive them from authenticated caller/session context before constructing runtime inputs.

For the current daemon/admin path, store location and backend are still chosen by the host process at startup. Clients can provide session/scope identifiers, but they should never be able to choose arbitrary persistence paths at turn time.

## Remaining Informal JSON Surfaces

The recent contract work removed several of the highest-friction JSON seams, but a number of important "still mostly implicit" JSON shapes remain. The most relevant near-term backlog looks roughly like this:

1. `common/agent/agent-runtime.cpp`

   This file is much cleaner now: request-tool safe defaults plus the bounded user-correction, failure-observation, reflection-hint, and reasoning-observation payload seams all go through named runtime JSON helpers. The remaining runtime-core cleanup is more about reducing mixed responsibilities than about raw JSON literals.

2. `tools/agent/cli/agent-cli-selection.cpp`

   This seam is narrower now too: selection schemas are requested through named schema-string helpers, and blueprint-binding tool arguments travel as a named plan tool-arguments contract instead of raw nested JSON until the final plan serializer step. The remaining work there is mostly around separating more assembly responsibility out of the CLI adapter, not about ad hoc nested selection payloads.

3. `tools/agent/daemon/agent-daemon-adapter.cpp` plus `tools/agent/daemon/agent-daemon-client.cpp`

   The daemon wire payload and the JSONL transport framing are now both explicit seams. The daemon protocol still owns the command/response objects, while the transport endpoints now reuse one small JSONL helper for line-oriented parsing and emission instead of open-coding request/shutdown JSON in each caller.

4. `tools/agent/tooling/agent-tool-provider.cpp`

   The provider/view boundary is structurally much better now, and the final success/failure payload shaping has been pulled behind named helper contracts. The remaining work here is mostly around richer typed normalization and broader MCP capability parity, rather than hand-built wrapper JSON at each provider return site. MCP tool input schemas already use the shared bounded validator on native, outbound and inbound paths.

5. `common/memory/memory-tool-service.cpp`

   Memory tool `search` and `remember` now parse through named argument-contract helpers before the service logic runs. The follow-up work here is mostly about extending that pattern to any additional memory tool surfaces rather than first extraction.

6. `common/plan/cozo/plan-cozo.cpp`

   Plan persistence and event rows still serialize and deserialize larger JSON payloads inline for Cozo storage. Some of this is legitimate persistence encoding, but the stored row/document shapes are still only implicit.

7. `common/memory/cozo/memory-cozo.cpp`

   The same pattern exists on the memory side: several metadata/result/row shapes are persisted and reloaded as raw JSON without a named storage contract layer.

8. `common/chat.cpp`

   This file intentionally owns a large amount of OpenAI-compatible chat/tool JSON mapping, so not all JSON use here is debt. Even so, there are still some model-facing parsed/normalized shapes here that could benefit from more explicit contract helpers over time, especially where argument strings are parsed and re-emitted.

9. `tools/agent/runtime/agent-server-inference.cpp`

   A few resident `server_context` result paths still parse response JSON blobs directly to interpret structured output. The scope is smaller than the files above, but it is still a live runtime seam.

If we count "production-relevant, non-test JSON shapes that are still at least partly informal", there are about 8 to 9 meaningful clusters left. If we count every single `.dump()` or `parse()` site in persistence, chat compatibility, diagnostics, and tests, the raw number is much larger. The clustered backlog above is the useful planning unit.

### Tools

Tools currently have three layers:

- Catalog: declares versioned metadata and profile membership.
- Registry: owns executable handlers.
- Adapter bindings: bind catalog definitions to local runtime resources such as memory store, plan id, repository root, and embedding provider.

There is now also a small provider/view boundary above those native pieces:

- `agent_tool_provider`: resolve tools for one host-owned runtime context.
- `agent_tool_view`: expose model-facing `common_chat_tool` values and execute one validated tool call.

The first implementation started native-only, and it still uses the existing catalog, registry, and adapter bindings underneath. The chat runtime no longer dispatches profile tools directly through a runtime-owned registry pointer. Instead, the host resolves a policy-bound, scope-bound `agent_tool_view`, passes `chat_tools()` into generation, and routes parsed assistant tool calls back through that view.

The embedding callback that sits underneath memory-oriented native tools is also one step less CLI-bound now. There is still a CLI wrapper for convenience, but the underlying helper can now work from a host-owned model path plus `n_gpu_layers`, which lets the daemon and real MCP stdio server construct their own embedding providers without carrying full CLI option objects into that path.

That provider result path now also carries host-owned resource refs for native tools, not only MCP-shaped ones. In the first concrete slice, larger `web_search` payloads can now be externalized into the configured resource store and returned as `resources` alongside a shorter inline result. The full search payload remains host-addressable by resource URI, while the inline tool result can stay bounded for the model.

That evidence path now continues one step further into planning state. Tool observations can now retain resolved `resource_refs` alongside their bounded inline summary, and the read-only `resource_read` native tool gives later steps a host-owned way to load the deferred payload back by opaque URI instead of forcing the earlier tool to keep everything inline.

There is now also a first MCP-shaped provider slice beside the native one. `mcp_agent_tool_provider` sits behind a narrow `agent_mcp_tool_client` interface, resolves model-visible `common_chat_tool` values from listed MCP-style tool definitions, namespaces exposed tool names, applies the same host-owned policy gate at exposure time, and normalizes tool-call results back into the shared `agent_tool_result` shape.

That seam now has two concrete test paths:

- an in-process fake MCP client used to exercise provider filtering and result normalization
- a first stdio-based MCP client adapter that speaks JSON-RPC-style `Content-Length` framed messages to a subprocess

The stdio client is still deliberately small. It is enough to prove the provider boundary through a real child process with `initialize`, `tools/list`, `tools/call`, `resources/list`, and `resources/read`. The subprocess is started with the async/no-wait sheredom options, and all stdout response reads use `subprocess_read_stdout()` into an incremental byte buffer. The client must not mix those options with blocking `FILE*` reads such as `fgetc()` or `fread()`; on Windows that can leave a pipe read blocked and can also make timeout cleanup hang while joining a reader thread. A request deadline now terminates the child process on expiry, while the no-deadline configuration uses the same non-blocking subprocess read path without imposing a synthetic deadline. Reconnect logic, approval model, streaming event path, and broader capability surface remain deferred.

The regression contract is covered by `llama-agent-mcp-stdio-client-smoke`, registered as `llama-agent-mcp-stdio-client-ctest` under the `agent` label. The smoke exercises ordinary framed responses, malformed-server diagnostics, and a hanging `tools/list` server that must be terminated within the configured request deadline. This is intentionally a transport-lifecycle check rather than a newline-format check: MCP framing remains `Content-Length` plus CRLF separators, but the transport read API is the portability boundary.

On the server side, the subprocess path is now also a little more explicit. There is a small reusable stdio MCP server core in the PoC layer: JSON-RPC framing helpers, a tiny server-side tool registry, and a stdio server loop that dispatches `initialize`, `tools/list`, `tools/call`, `shutdown`, and `exit`. The older fake subprocess now reuses that same core, and the first real PoC MCP stdio server binary now exports a host-resolved tool surface through the same catalog/provider/bindings path used by the host runtime. That keeps the current subprocess path from drifting into a second ad hoc server shape while still staying much smaller than a full agent daemon or broader MCP host surface.

That real MCP server can now also be bootstrapped through the same `--config PATH` host-config entrypoint as the daemon, and can run either its existing stdio transport or the bounded inbound HTTP transport selected with `--http-listen`. In practice that means the tool export seam is no longer tied only to per-process CLI flags; it can already be described through a shared host-owned config file, including enabled stdio MCP subprocess providers, even though the current server still exports tools rather than a full resident agent runtime.

Its internal tool assembly is now also closer to the rest of the host/runtime stack. The server no longer hand-builds a separate native provider wiring path for export; instead it resolves its tool surface through the same host-owned tool-selection seam used by the CLI/daemon side, while still applying its own export policy to keep the MCP-visible tool surface intentionally narrower than a fully bound local runtime.

That current subprocess story now has two intentionally different shapes:

- fake/smoke MCP server: small hand-authored stub tools used to exercise protocol, namespacing, error mapping and resource-link normalization
- real MCP stdio server: a host-resolved tool surface exported through the same provider/bindings path the host runtime already uses

The important implication is that "available through MCP" now means "available through the real stdio server's resolved tool surface and bindings", not "everything mentioned anywhere in MCP smokes". The fake server remains a protocol/regression harness, not the exported host surface.

### MCP Export Surface Today

The current real MCP stdio server exports a host-resolved tool surface, not the whole agent runtime loop. In practice that means native tools from the selected profile can appear there, and configured external stdio MCP providers can also be forwarded into that same exported surface when the host config enables them.

The table below describes the currently bound/exportable surface, not every
tool proposal or every catalog entry. A tool is visible only when its profile,
provider binding, host scope and backend requirements are satisfied. In
particular, sandbox-dependent tools can be omitted when no usable execution
backend is configured.

| Tool/capability group | Available through real MCP stdio server | What it requires today |
| --- | --- | --- |
| `calculator`, `time_now` | Yes | A profile containing the tools, such as `minimal` |
| Memory reads: `memory_search`, `memory_get`, `memory_inspect`, `memory_conflict_check` | Yes, when bound | A matching profile and a host-owned memory store, in-memory or persistent |
| Memory proposals: `memory_remember`, `memory_propose_update`, `memory_propose_forget`, `memory_link`, `memory_compact_propose` | Yes, when bound and allowed | A write-capable profile, memory store and host policy allowing proposals |
| Planning: `plan_get`, `plan_propose` | Yes, when bound | A matching profile and plan store; `plan_get` is useful with a bound `plan_id` |
| Repository: `repository.list`, `repository.search`, `repository.read`, `repository.diff`, `repository.log`, `repository.status`, `repository.changed_files` | Yes | A matching profile and host-owned `repository_root` |
| Workspace: `workspace.list`, `workspace.read`, `workspace.search`, `workspace.patch` | Yes | A matching developer profile and host-owned repository/workspace root; patch remains policy-gated |
| Development: `development.build`, `development.test` | Yes, when bound | A matching profile and configured sandbox execution backend |
| Diagnostics: `diagnostics.compile`, `diagnostics.symbol`, `diagnostics.references`, `diagnostics.test_failures`, `diagnostics.format`, `diagnostics.include_graph` | Yes | A matching profile and host repository root; symbol/reference results use a semantic provider or explicit text fallback |
| Dataset: `dataset.list`, `dataset.inspect`, `dataset.schema`, `dataset.sample`, `dataset.validate` | Yes | A matching profile and host-owned repository/workspace root; current file support is bounded, with CSV as the primary implementation |
| Structured data: `data.query`, `data.filter`, `data.aggregate`, `data.join`, `data.transform`, `statistics.describe`, `statistics.outliers` | Yes, when bound | A matching profile and configured host-owned data store, such as Cozo |
| `resource_read` and `artifact.export` | Yes, when bound | A matching profile and host-owned resource store; artifact output remains bounded |
| Network: `web_search`, `web_fetch` | Yes, when allowed | A matching profile, network policy and the host's safe network provider |
| `resources/list` and `resources/read` | Yes | The real MCP stdio server owns a scoped host resource store and exposes host-authorized resource descriptors and reads |
| External MCP subprocess tools such as prefixed `github_search_issues` | Yes, when configured | Host configuration enabling an stdio MCP provider with a prefix and command |
| Full agent/planning runtime, reflection loop and memory-learning loop | No | These are runtime behaviors, not MCP tools in the current surface |
| Fake server tools such as `search_issues`, `search_recent_failures`, `create_issue` | No, not from the real server | These remain test-only tools from the fake MCP server path |

This is also the cleanest way to think about memory, planning and resources through MCP right now:

- memory can be exported as native MCP-visible tools when the server process has any host-owned memory store, including the default in-memory store
- planning can expose its native plan tools, but that is not the same thing as exporting the whole agent/planning runtime as an MCP tool surface
- resources are the strongest fit so far because the real server always owns the resource store contract, can now expose `resources/list` and `resources/read`, and native tools can already return opaque resource refs for deferred payloads

So the current MCP stdio server is best understood as "native tool export through an MCP transport seam", not yet as "the agent runtime itself exposed as an MCP server".

Even in this small slice, the stdio client now does a little more than the original smoke seam: it switches Windows stdio pipes to binary framing for `Content-Length` transport, attempts a best-effort `shutdown` plus `exit` sequence before tearing down the child process, and can map structured MCP-side tool error metadata back into the shared failure contract used by native tools.

The stdio client now also keeps a small stderr tail from the MCP child and appends it to transport-level failures when the subprocess exits early or emits malformed JSON-RPC data. That keeps the first real subprocess smoke debuggable without pulling in a larger async lifecycle or logging subsystem yet.

There is now also a first thin integration into the ordinary CLI host-adapter path. When `llama-agent run` is given `--mcp-tool-command`, the CLI tool-selection layer resolves an MCP-backed `agent_tool_view` through the same host/runtime tooling contract used by native tools. This first slice is intentionally narrow: it covers the direct CLI host/runtime path, not the daemon adapters yet.

That CLI path can now also compose native and MCP-backed tools in the same resolved view. If both `--tool-profile` and `--mcp-tool-command` are present, the host resolves them through one small composite provider surface and keeps model-visible tool names unique at the merge boundary.

The subprocess-based MCP smokes now also declare their helper binaries explicitly at the build layer. Targets that launch `llama-agent-mcp-stdio-fake-server` or the real `llama-agent-mcp-stdio-server` now depend on those helper executables directly, which avoids a subtle stale-binary failure mode where a smoke would exercise an older helper build and appear to hang or regress in protocol behavior even though the caller target itself had rebuilt cleanly.

The foreground daemon can now also resolve the same native and MCP-backed tool surface that the CLI host adapter already uses. Session-host execution now has a per-turn tooling resolver hook, and the daemon uses it to rebuild tooling from the current host-owned session/scope turn contract before each turn. In practice that means `--tool-profile`, `--repository-root`, and `--mcp-tool-command` can now participate in the same resolved daemon tool view instead of the daemon having a narrower MCP-only wiring path.

The modern CLI profile-tool path also no longer builds a second parallel native registry just to derive model-facing tools. For profile-driven tools it now resolves one provider view and reuses that single host-owned surface for exposure and execution wiring.

That keeps host authority in one place:

- the model sees only `common_chat_tool`
- the model requests tools only through parsed `tool_calls`
- the runtime owns bindings, scope, repository root, memory authority, and policy
- disallowed native tools are filtered before exposure instead of being shown and rejected later

The first agent/planning migration is now also in place for blueprint binding. Auto-selected blueprint bindings no longer inspect the native registry directly; they validate against the resolved `agent_tool_view`, which means blueprint-binding now follows the same host-owned exposure and read-only policy surface as chat tool dispatch.

The planning/orchestration edge is also a little less ad hoc now. Plan auto-selection and blueprint auto-selection no longer take a longer loose parameter list for inference, generation config, scope, plan state and tool-view details separately. Instead they can be driven from one small orchestration runtime context that carries the current host-owned planning inputs plus the optional tooling bundle used for blueprint binding.

The older common agent runtime is now also structurally decoupled from the concrete registry type. Planned tool-step execution runs through a small `common_agent_tool_runtime` interface in `common/agent`, and the PoC runtime now adapts the resolved `agent_tool_view` into that interface through one shared provider-backed bridge. That means the common planning/runtime core no longer needs to know whether tool execution ultimately comes from native tools, a host-resolved provider view, or a future MCP-backed adapter.

What still remains is mostly coverage and convergence work rather than contract churn. The provider-backed path is now the intended modern shape; the next cleanup is to keep broadening smoke and integration coverage around the same interface as more agent/runtime and daemon flows exercise it.

There is now also a focused smoke for the planned-tool-step path itself. It runs a tiny `common_agent_runtime` scenario where the planner emits a calculator tool step and execution goes through a provider-backed tool-runtime adapter rather than a raw registry pointer. That gives the current refactor one concrete end-to-end proof point before broader resident/daemon agent smokes are added.

The resident/session-host layer has also been trimmed a bit further: it now carries `agent_tool_view` for the modern profile-tool path, but no longer keeps threading a separate `tool_registry` field through its own configs just to pass it onward unused. That makes the resident host/session contracts slightly closer to the intended provider-first shape.

The agent/runtime assembly path has also dropped its registry-backed tool-runtime fallback. Planned tool-step execution in that path now expects the modern provider-backed `agent_tool_view` when profile tools are active, and fails explicitly if a caller tries to run profile-tool planning without that resolved view. That narrows the remaining legacy surface and keeps the planned-tool runtime aligned with the provider-first direction already covered by smoke tests.

The host/chat contracts have now been trimmed in the same direction. The runtime host, chat driver, CLI host adapter and resident host path no longer carry a parallel `tool_registry` field through their modern provider-backed contracts. Chat dispatch itself also no longer carries a separate legacy tool-handler branch. Modern agent-side tool execution now goes through `agent_tool_view`, with the host deciding which scoped provider view to resolve for the turn.

The first shared memory-tool migration now sits underneath that provider path as well. Native `memory_search` and `memory_remember` execution both go through `common_memory_tool_service`, which keeps host-owned scope, embedding, store and policy bindings in one place while preserving the current synchronous behavior and result shapes.

The remaining CLI coupling around embeddings has also been narrowed further. The CLI path still supplies one concrete embedding implementation, but profile-tool bindings and runtime memory-learning hooks now reach it through a small `agent_embedding_provider` seam rather than capturing CLI options directly inside the native tool bindings.

The default runtime tool path remains synchronous once a tool call is entered,
but the provider/view seam now supports selected async-capable tools through
`begin_call_async(...)` and `poll_call_async(...)`. The daemon/session pending
operation path can carry those calls with explicit cancellation, deadlines and
terminal outcomes. A future broader worker model still needs explicit
semantics for cancellation, timeouts, ordering, result delivery and
shared-state access.

The cancellation/timeout seam now also reaches a little deeper into the runtime-owned tool and inference paths. Resolved `agent_tool_context` values now carry the shared execution-control contract, native and MCP-backed tool views fail early when a host cancellation or turn deadline has already fired, and the current runtime checks the same seam again immediately after synchronous tool execution. That is still cooperative rather than preemptive, but it means the provider-backed tool surface now has one host-owned place to report `tool_call_cancelled`, `tool_call_deadline_exceeded`, or timeout-class failures instead of only treating those conditions as outer daemon concerns.

The runtime now also starts mapping inference-step budgets into the existing generation request contract. CLI and resident/session-host assembly set `t_max_predict_ms` from `inference_step_timeout_ms` when configured, so the `server-context` path can begin enforcing bounded generation time through the same generation-options seam it already uses for `n_predict`. This is still only a first slice: active cancellation does not yet safely interrupt every in-flight inference or native tool body, and inference is not yet managed as a globally scheduled/batched operation.

### Continuation checkpoints and context pressure

Long-running work must remain resumable without creating a second plan or a
second daemon queue entry. The first continuation contract therefore lives in
the existing runtime/session architecture. A
`common_agent_continuation_checkpoint` is execution state for the current turn,
not a durable memory record. It carries the accepted request and turn IDs, the
plan ID and version it observed, the active step or next action, a sequence
number, the continuation reason, and references to large external resources.

Native validation requires a non-empty checkpoint identity, the original turn
identity, a plan identity and revision, and an actionable continuation point.
The session lane is the anti-forking boundary: a checkpoint can resume only the
same accepted turn against the same plan revision. A changed plan must produce
a new checkpoint rather than silently continuing from stale state.

The checkpoint is transported by the existing agent result and active-turn
state. This keeps the result observable to the controller and daemon layer
without persisting execution state in memory or introducing a second session
store.

The first bounded context-compaction helper now uses the same path. Before a
context-pressure continuation, the driver projects the authoritative plan into
`common_agent_working_state`, compacts an active policy pack with the existing
memory-policy helper, and deduplicates/caps input-resource references while
preserving their host-owned descriptors. This is deliberately section-level
compaction, not a second context store. The current driver does not own a
general conversation-history ledger, so exact tokenizer accounting and
arbitrary chat/tool-history summarization remain outside this milestone.

The agent driver now performs a bounded automatic continuation for this first
case. When a generation slice ends with `stop_reason=limit`, the driver keeps
the same accepted turn and current plan, constructs a bounded continuation
prompt from the existing plan and recent result, and runs at most
`max_continuations` additional slices. The default is two additional slices;
zero disables this behavior. The slices stay inside the existing driver
operation, so the daemon queue does not receive a second external command and
the session lane remains the anti-forking boundary.

The execution flow is:

```text
same session-lane turn
  -> bounded inference slice reaches a continuation boundary
  -> driver resumes the same plan when the continuation budget allows
  -> otherwise checkpoint records plan revision and resource references
  -> one terminal turn result
```

When the boundary is caused by deterministic context pressure rather than a
model completion limit, the next slice uses the bounded working-state
projection carried by the execution request. It does not re-render the full
verbose plan context. The projection keeps the goal, phase, completed and
pending steps, constraints, open questions, resource references, and chunk
status needed to continue safely; the plan and resource stores remain
authoritative. This is still an internal continuation of the same session-lane
operation, not a new queue item or a second plan.

The daemon dispatcher must not receive the continuation as a new external
command. Pending asynchronous tools remain owned by the operation manager,
and an inference slice may reacquire the existing inference-capacity lease.
This preserves queue capacity accounting, per-session ordering, cancellation,
and the existing terminal-event contract.

This contract does not yet claim full automatic context compaction. Rendered
memory overlays and policy packs already have bounded compaction, and research
workspace state can now be checkpointed without leaving the session lane, but
full conversation/tool-result compaction still needs controller integration
and resource-reference recovery. It also does not structurally repair arbitrary
JSON or apply the continuation loop to chat-driver output. If the bounded
slice budget is exhausted, the result remains incomplete and carries a
validated checkpoint rather than being reported as a successful final answer.

The chat-driver boundary is now explicit as well. If a conversation generation
ends with the backend's length/limit stop reason, the partial text remains in
the native result for diagnostics, but the result is marked `limit_reached` and
the driver returns an incomplete/error outcome. It is never silently treated as
a complete user answer or concatenated with an arbitrary retry. Structured chat
continuation must use a validated checkpoint and a structural regeneration
boundary, especially for JSON or tool-call output.

The guard also runs before chat parsing and tool dispatch. A truncated message
therefore cannot turn a partial tool-call envelope into an executable call; the
native result is rejected at the same boundary and the host retains the
partial payload only for diagnosis.

For tool-free plain text, the chat driver now has a bounded continuation slice
using the same turn/session lane and `runtime_config.max_continuations` budget
as the agent driver. It appends the completed text fragment as an assistant
message, asks for only the next segment, and clears the transient limit state
when a later slice completes. This does not apply to tool-enabled or
structured output; those paths still require structural regeneration rather
than text concatenation.

The model planner now applies that rule to its compact plan JSON: a partial or
schema-invalid planner object receives one bounded regeneration request from
the beginning. The first payload is never concatenated with the retry. If the
regeneration is also invalid, the existing bounded fallback/error behavior
applies. This is intentionally one local repair attempt, not an unbounded
model loop.

Reflection JSON follows the same boundary. A length-stopped or parse-invalid
reflection is regenerated once from the beginning before any reflection
decision, learning hint, or proposed plan operation is accepted. If the retry
does not validate, the existing safe behavior accepts the current draft without
applying untrusted reflection operations.

The post-turn memory candidate extractor uses the same one-retry boundary. A
partial or invalid candidate object is regenerated before it reaches the
existing evidence, provenance, scope, and memory-learning policy. The first
payload is never treated as a candidate and is not concatenated with the retry;
if validation still fails, no candidate is proposed.

The retry mechanics are shared through a small `common/agent` helper. The
planner, reflection engine, and memory candidate extractor still own their
separate prompts, schemas, and parsers; the helper only enforces the common
bounded generate/accept contract. A retry is allowed only after a completed
generation whose structured payload was rejected. Host cancellation and
backend failures are returned immediately and cannot consume another
inference attempt. This keeps structured regeneration in the existing
host/runtime assembly rather than creating a second repair subsystem.

## MCP Direction

An MCP integration should be built on top of the runtime host, not inside the core agent loop.

The natural mapping is:

```text
MCP-facing llama-agent host
        |
        +--> runtime host
        |       +--> inference backend
        |       +--> stores
        |       +--> scope and policy
        |
        +--> tool providers
                +--> native registry provider
                +--> future MCP client provider
```

For this codebase, "MCP support" should mean the application can act as an MCP host: it discovers tools/resources/prompts from MCP servers through MCP clients, applies local policy, exposes allowed capabilities to the model, dispatches model-requested tool calls, and feeds results back into the runtime.

The runtime should first grow an internal tool-provider boundary before adding MCP transport details.

A useful provider shape is:

- List tools available to this runtime scope and policy.
- Return model-visible schemas for allowed tools.
- Call one tool with validated JSON arguments.
- Return structured success/failure results.

The native tool registry is the first concrete provider. There is now also an MCP-shaped provider seam built on top of an abstract MCP client contract, with stdio and Streamable HTTP client adapters covering `initialize`, `tools/list`, `tools/call`, `resources/list`, and `resources/read`. The daemon and MCP server also expose an inbound HTTP adapter above the same host-resolved tool/policy surface. Broader capability coverage and streaming remain separate follow-up work.

That MCP-facing tool surface has now also been tightened slightly around naming and policy. Runtime filters treat the resolved model-visible name as the authority surface for exposed MCP tools, so prefixed MCP names are filtered the same way the model actually sees them rather than by an internal pre-prefix identifier.

MCP tool arguments now use the same bounded JSON Schema subset as native tool calls. Required fields, additional-property policy, scalar types, enum/bounds and array bounds are validated against the advertised `inputSchema` before the provider is invoked. The agent's MCP server registry applies the same validation before invoking an inbound tool handler and passes the normalized arguments onward. Full JSON Schema and transport-specific schema dialects remain outside the current subset.

Resources now follow the same host-owned pattern and are exposed through the MCP client/server seams where configured. Prompts and broader MCP capabilities can follow later; they should not be added directly to the agent loop as transport-specific concepts.

## Service Direction

The next daemon-facing step should be designed as a host/service core that can later be run as a Unix-style service even though current development happens on Windows. That should affect the shape of the code now, but not force immediate platform-specific daemonization work.

The long-term target should look more like this:

```text
process host / transport adapter
        |
        +--> stdio foreground host
        +--> unix-socket host (POSIX)
        +--> future named-pipe host
        +--> inbound MCP HTTP host
                |
                v
agent daemon service
        |
        +--> lifecycle state
        +--> command ingress
        +--> scheduler / worker lanes
        +--> session manager
        +--> event/result sink
                |
                v
agent runtime host
        |
        +--> inference provider
        +--> tool provider
        +--> stores
        +--> policy and scope resolution
                |
                v
resident inference backend
        +--> local CLI generation adapter
        +--> server_context host
        +--> later remote or pooled backends
```

The important rule is that the service core should not know whether it is hosted from foreground stdio, a Unix service wrapper, a Windows process wrapper, or a later HTTP listener. Those are transport/process-host choices around the same command/session/runtime core, not separate daemon implementations.

`llama-server` remains useful inspiration here, but mostly for responsibility boundaries rather than for direct reuse of its public REST surface. In practice the reusable ideas are:

- transport threads should not run agent turns directly
- requests should become commands/tasks that move through a queue or scheduler
- readiness, drain and shutdown should be explicit service concerns
- inference/context ownership should stay inside the inference side of the host rather than leaking into transport code

For this branch that means `server_context` should be treated as an inference/backend building block under `llama-agent`, not as the outer service model. The agent daemon should own routing, session lifetime, tool policy and stores; the inference backend should own model/context execution.

### Service States

The service lifecycle should become explicit before new transports or real background hosting are added. A small target state model is:

- `starting`
- `ready`
- `draining`
- `stopping`
- `stopped`
- `failed`

The corresponding host-facing operations should stay equally small:

- `start()`
- `request_shutdown(drain|cancel)`
- `status()`
- `accepting_commands()`

Foreground stdio can keep using this lifecycle first. A later Unix-service or detached-process wrapper should only host the same service object and react to the same lifecycle/status contracts.

### Scheduler Direction

The daemon is still intentionally single-lane today, but the next design should already make room for asynchronous and multi-session execution.

The intended shape is:

```text
transport adapters
        |
        v
bounded command queue
        |
        v
scheduler / dispatcher
        |
        +--> worker lane 1
        +--> worker lane 2
        +--> ...
```

The dispatcher already supports a configurable worker pool, with one worker as the compatibility default. A global inference admission scheduler now sits above the resident host execution path: it enforces `limits.inference_max_active` independently of worker count, admits named waiters using priority-aware FIFO ordering with bounded aging, and parks excess turns through the existing pending-operation/status/event seam. Backend slot and batch awareness remain deferred to the model-serving adapter.

Admission priority is currently derived from the host turn: chat is
interactive, normal agent work is normal, and research is background. Waiters
remain FIFO within an effective priority; aging raises a waiter's effective
priority after five seconds, capped at the interactive level. This is a
bounded first policy, not yet a configurable QoS system.

That scheduler should eventually be able to enforce rules such as:

- different sessions may run in parallel when capacity exists
- the same session should normally serialize turns
- cancellation and deadlines should target queued or active commands by request/turn identity
- inference-capacity limits should be separate from logical session ownership

For a turn that must wait, the observable event order is:

```text
inference.queued
  -> turn.waiting_for_inference
  -> inference.capacity_granted
  -> turn.waiting_for_inference       # backend execution is now active
  -> turn.completed | turn.failed | turn.cancelled
```

The first waiting event describes admission waiting; the second describes the
resident host execution that follows. A cancelled or expired waiter is removed
from the admission queue before the turn becomes terminal.

### Session and Lifetime Model

The long-term host model should keep three identities distinct:

- `namespace` as tenant/authority boundary
- `project` as longer-lived shared work container
- `session` as live runtime/conversation lane inside that project

That aligns with the current daemon/session direction, but it should become more explicit over time. A future multi-session host should be able to keep several sessions alive inside one project without treating each session as a separate model owner.

The same principle applies to runtime lifetime:

- process lifetime
- model lifetime
- inference-context lifetime
- agent-session lifetime
- turn lifetime

The current runtime/session split already moves in that direction. The
remaining lifetime work is to carry that split through the concrete resident
host so a host can unload one session, reset one context, or expire one lane
without forcing a model reload.

### Beta Readiness Status

The structural preparation described in the original beta plan is now present
in the current daemon/service core. It should be read as an implementation
status, not as a list of unfinished prerequisites:

| Seam | Current status | Remaining boundary |
|---|---|---|
| Service lifecycle | Implemented with explicit `starting`, `ready`, `draining`, `stopping`, `stopped` and `failed` states. | Detached or OS-managed hosting is still deferred. |
| Command/result contract | Implemented as transport-neutral daemon command/result contracts, with JSONL as an adapter. | Additional transports must reuse the same core rather than create a second path. |
| Status/readiness | Implemented through the same host-owned status snapshot used by daemon/admin flows. | Richer observability is deferred. |
| Session ownership | Implemented in the keyed session manager, including resident state, scope and policy-pack identity. | A richer project object is still deferred. |
| Lifetime split | Model, inference-context, session and turn identities are represented separately in the runtime/session contracts. | The resident server-context backend still has a coarser concrete ownership boundary. |
| Bounded queue | Implemented with explicit capacity, worker lifecycle, queue state and shutdown behavior. | The inference gate provides admission control, but fairness and backend slot/batch scheduling remain deferred. |
| Cancellation identity | Implemented for queued cancellation and active-turn cancel requests using request/turn/session identity. | End-to-end interruption of inference, native tools and MCP calls remains incomplete. |
| Event/result seam | Implemented with typed internal daemon events and a separate terminal result projection. | Live cross-process streaming remains a later protocol capability. |

These seams are therefore beta foundations already in use by the current
smokes and daemon flows. The actual beta limitations are the boundaries in the
last column, especially inference fairness and backend slot scheduling,
complete abort propagation, detached hosting and production observability. The intended
architecture remains one daemon/service core with foreground stdio as its
first host.

## What Not To Build Yet

These are intentionally deferred in the current branch:

- Detached or OS-managed daemon/service lifecycle.
- Named pipes and detached/OS-managed service lifecycle.
- A richer project object above the current namespace/session lane model.
- Full `llama-server` integration.
- Broader MCP capabilities beyond the current bounded Streamable HTTP GET/SSE
  event projection and opt-in experimental Tasks polling.
- Full end-to-end cancellation through inference, native tools and MCP transports.
- Global inference scheduling and batching.
- Production observability and a richer cross-process event protocol.

The current code should remain useful without these deferred capabilities. The
transport, auth, worker-pool and bounded schema seams already exist; the next
steps should extend them without creating a second runtime path.

## Next Steps

1. Finish moving the last host callers away from CLI-shaped state.

   The host builders now accept a CLI-free runtime turn request and CLI-free policy/runtime/orchestration contracts. The next cleanup is to move more callers onto those contracts directly, so non-CLI hosts can build prompt/messages, scope, policy, inference options, generation options, plan identity and hooks without routing through CLI-shaped helpers.

2. Complete the remaining concrete lifetime separation in resident hosts.

   `common_agent_runtime_session` already tracks loaded-model state and active
   inference-context state separately. The remaining work is to carry that
   separation through the concrete resident host so keyed agent sessions do
   not implicitly own more model/context lifetime than they need.

3. Converge the remaining CLI-shaped adapters around host construction.

   The host/runtime contracts now carry a shared tooling bundle, and the runtime/orchestration modules no longer expose `args`-shaped builders. The next cleanup is to keep pushing that edge outward so more callers can assemble host turns, plan identity and hooks directly from host-owned contracts.

4. Finish the remaining provider-surface convergence.

   The runtime host now carries one tooling bundle instead of separate `tools + profile_tools_active + tool_view` fields, and chat/runtime dispatch uses that bundle consistently. The remaining cleanup is mostly convergence work: keep trimming older helper signatures and make sure blueprint/planning-related tool decisions continue to depend on the same host-owned provider view rather than drifting back toward registry-era wiring.

5. Extend cancellation, timeout and event contracts before broader async tools.

   Synchronous tools are still acceptable for the current slice. The daemon/session seam now has a shared execution-control contract plus active-turn cancellation requests, configured timeout budgets, and per-turn daemon request overrides, but workers should still wait until those controls propagate all the way into inference, tool execution, MCP requests, retry policy, ordering, and richer failure reporting.

   The session-lane state machine now also has a first concrete pending-operation seam above that control contract. This is intentionally narrow: the keyed session manager can now park a lane behind a manager-owned pending operation, poll that operation, and then either resume the same turn or cancel/fail it through the same host-owned execution control. The same seam now covers both `wait_for_tool` and `wait_for_inference`; the difference is the parked phase/disposition and the point where the turn resumes. The important boundary is that this is still a lane/session seam, not a fully asynchronous resident runtime yet. `common_agent_runtime_session_host` still executes one turn synchronously once it is entered; the new waiting states exist so daemon/session orchestration can start behaving more like an actor without pretending the whole agent loop is already coroutine-shaped.

6. Split model lifetime, inference context lifetime and agent-session lifetime more explicitly.

   The current `server-context` path is a good resident smoke backend, but a real host should be able to keep models loaded while resetting or expiring individual agent sessions.

7. Harden the first stdio MCP client before extending MCP capability breadth.

   The first provider slice now exists behind an abstract MCP client contract and already has a small stdio transport adapter with basic shutdown and structured tool-error mapping. The next step is to harden that path further: clearer diagnostics, less forceful teardown, better request/response correlation under notifications, and then broader MCP capability support. Add resources and prompts only after tool discovery, session ownership, and policy are stable.

## Backlog Notes

- Keep symbolic memory inside the existing memory/store model, then add overlays above it.

  The intended first symbolic slice is deliberately small:

  - continue using the current host-owned memory stores, scopes, provenance, proposal policy, MCP exposure, and retrieval path
  - treat `procedure`, `constraint`, and `decision` as the first symbolic memory kinds
  - keep inferred post-turn learning conservative at first; new symbolic kinds should start as explicit proposal-backed memory rather than immediate auto-learn targets
  - build later project/session overlays by compacting retrieved symbolic memories plus resource/evidence links instead of inventing a parallel persistence model

  The next follow-up after this kind-level slice is a typed overlay builder for planning, reflection, and reasoning. That overlay should assemble a bounded block such as constraints, decisions, procedures, relevant facts, and evidence/resource references from already authorized memory retrieval. In other words: memory remains the durable source of truth, while overlays become a host-owned context-shaping view of that memory.

  The current implementation is still intentionally conservative: the overlay is only a renderer over already retrieved memories, not a new retrieval path, scoring policy, or compaction store. The next useful step is therefore to make symbolic retrieval/selection a little smarter per stage, especially so planning and reflection can emphasize project-scoped constraints and decisions while reasoning can keep favoring procedure memories plus any directly relevant symbolic guidance.

### Blueprint applicability and asynchronous learning

Blueprint selection and post-turn learning share one contract. A blueprint is
not applicable merely because its description is topically similar to the
request: its purpose, goal, success criteria, constraints, assumptions, and
step contributions describe the execution boundary that selection must
respect.

The model-facing selector projection is derived from that same persisted plan:
it includes purpose, goal, success criteria, constraints, assumptions, and
bounded step contributions in addition to the short description. This keeps
semantic ranking informed without giving the selector storage identity or
runtime authority. Native code remains responsible for scope, state, and
later instantiation checks.

Promotion therefore preserves the plan purpose, constraints, and assumptions
alongside the reusable goal, success criteria, and steps. This keeps a learned
blueprint usable by later asynchronous turns: the host can snapshot the
caller-owned objective, resolved capability profile, scope, identity, and
budgets before selection, then evaluate the persisted blueprint against that
same snapshot. Selection may rank eligible candidates, but it must not make a
candidate executable when its assumptions are known false or its hard
constraints cannot be honored.

The async runtime must also keep the ordering explicit:

`resumable task plan -> deterministic blueprint eligibility -> model ranking ->
instantiation -> execution -> verified post-turn learning/promotion`

Resumable task plans are checked before blueprint candidate availability. If a
task plan can be resumed, an empty or stale blueprint list must not prevent the
continuation. For a new task, native code then filters candidates against the
persisted plan kind and the current namespace/scope identity before invoking
the model selector. This keeps model ranking bounded to executable candidates
and reuses the same scope contract used by task-plan resumption.

An assumption marked `valid: false` is treated as a known-false precondition
and makes that candidate ineligible. An assumption that remains valid is not
treated as proven; unknown-but-not-false assumptions remain available for a
future bounded verification step or normal plan handling.

Blueprints may also declare `required_capabilities` on the existing persisted
plan contract. These are semantic host requirements such as
`development.build`, not selected tools or executable bindings. After the
active tool profile has been resolved, native selection removes candidates
whose required capabilities are absent before model ranking. A capability set
is therefore a host-owned snapshot for one turn; it is never expanded by a
blueprint, selector, or model. During earlier bootstrap installation an
unresolved capability remains unknown and is preserved for the later resolved
selection stage.

Events and traces describe this flow; they are not a second decision path.
Learning reads the completed, stable plan result and uses the shared memory
policy with stronger evidence requirements for procedure promotion. Decisions
and constraints remain explicit-first knowledge, while a promoted blueprint is
created only after repeated verified procedural use.

Blueprint selection also returns bounded native diagnostics: candidate count,
eligible count, and one rejection reason per filtered candidate. Automatic
selection projects those diagnostics into the existing agent event and plan
trace vectors, so asynchronous hosts can observe why a candidate was removed
without asking the model to explain native policy decisions.

Explicit capture and inferred post-turn learning use the same memory policy and
store, but are different acquisition paths. An explicit host-supplied
candidate is marked with `explicit_user_statement` provenance and may capture a
decision or constraint after normal validation. Inferred post-turn learning
continues to allow only fact, preference, or evidence-backed procedure
candidates; it cannot promote a model claim into a decision or constraint.
Explicit capture does not create a blueprint directly and does not bypass
scope, conflict, duplicate, or persistence policy. It is also confirmation
gated: the first asynchronous request may return the candidate with
`awaiting_confirmation` and emit `memory_capture_confirmation_required`, but
it does not write to the store. A later host-owned request may replay the same
candidate with `explicit_memory_confirmed`; only then can the normal native
memory policy persist it and emit `memory_capture_confirmed`.

This is the deliberate/research knowledge-gap pattern, not a second learning
store. A deliberate or research turn can inspect the repository and retrieved
memory, identify bounded unanswered questions, ask the maintainer for focused
answers, and summarize those answers as explicit candidates. The summary is
then confirmed as a separate turn before persistence. The runtime keeps the
candidate in the result/request boundary while that exchange is in progress;
the research workspace remains turn-scoped and is not silently promoted to
durable memory. Facts, decisions, and constraints are therefore explicit-first,
while inferred procedures still require completed-plan evidence and later
verification before promotion.

When the model declines or reports low confidence, native fallback ranking uses
the caller prompt together with the policy-pack purpose, goal, success criteria,
constraints, and preferred procedures. Purpose and goal overlap receive the
strongest weight, followed by success criteria and step contributions. This
ranking is only applied after deterministic eligibility filtering; it cannot
select a blueprint that is outside scope, has a known-false assumption, lacks
a resolved required capability, or conflicts with a host-blocked hard
constraint.

The daemon exposes the same path through `--agent-blueprint off|auto|ID` and
the `runtime.agent_blueprint` configuration field. Before automatic ranking,
the daemon resolves the active host tool profile for the turn and passes its
semantic capability set to native selection. A blueprint requirement such as
`development.build` is compared with that set, not with a model-selected tool
name. The `ready` response reports the configured profile, effective
capabilities, and blueprint-selection capability; selection diagnostics remain
in the existing event and trace vectors so asynchronous clients can observe
the same decision path as foreground hosts.

Hard constraint filtering is similarly host-owned. The active tooling context
may provide exact blocked constraint identifiers; native selection rejects a
blueprint with a matching hard constraint before model ranking. Unresolved
free-form constraint text is not interpreted with keyword heuristics and stays
within normal plan and policy validation.

  The current stage-aware selector is still only the first cut. It uses bounded kind-based weighting on the already retrieved hit set rather than a new retrieval pass. A later refinement can add project/session affinity, tool/step metadata boosts, evidence reuse hints, and more explicit cross-links, but that should stay above the same host-owned memory authority and below prompt rendering.

  The same principle applies to session policy. Blueprints are still the right inspiration for the declarative shape, but not the right runtime type to reuse directly. A later host/session layer can own stable policy packs per project or per resident lane, while retrieval overlays stay ephemeral and evidence-driven. That keeps executable plan structure, durable symbolic memory, and host/session policy related but distinct.

- Keep tightening the session-versus-project split above the current manager key.

  The current daemon/session layer now treats `namespace + session` as the resident lane key, while `project` remains part of the host-owned runtime scope bound within that lane. That is closer to the intended model, but it is still only a first cut: project-scoped memory and plans are still assembled through the same session-host contract, and there is not yet a richer host-owned project object above the resident runtime.

  In practice the intended direction is still the same: `namespace` stays the tenant/authority boundary, `project` is the longer-lived shared work container, and `session` is the shorter-lived live runtime/conversation lane inside that project. That becomes more important once MCP-facing host state, external tool providers, and richer multi-session project workflows sit above the current daemon/admin path.

- Revisit activity order after the current runtime/session cleanup wave.

  The branch has now completed several structural extractions in a row: provider-first tools, resident runtime/session layering, daemon service/dispatcher split, CLI-free runtime builders, and removal of older daemon/session aliases. Before taking another deep refactor in the same area, it is reasonable to pause and choose between a new activity track such as event/cancellation contracts, deeper `server-context` lifetime splitting, or MCP/provider-facing host work.

- Add host-owned resource references with turn/session/project lifetime.

  Large tool outputs do not need to stay inline forever. The first host-owned resource-store shape now exists with content-addressed filesystem blobs and a first Cozo-backed metadata option, and `web_search` is now the first native tool that can materialize a full result payload there while returning a shorter inline answer plus `resources`. The next real step is to use that same path more broadly from runtime/tool flows and to deepen the metadata side with TTL cleanup, provenance links, blob reference management and richer lookup operations. The intended lifetime split remains `turn` for short-lived tool artifacts, `session` for live working-set reuse across turns, and `project` for longer-lived shared artifacts tied to the work container rather than one conversation lane. As with memory and plan scope, the model should never choose arbitrary resource URIs or storage locations directly.

- Let planning and tool binding treat resource metadata as first-class evidence shape.

  The next planning-facing slice should not be "blob store first, meaning later". Resource metadata should become one of the ways observations explain what a deferred payload is for: purpose, short content summary, usage hint, limitations, and lightweight semantic tags such as keywords or entity names. That gives later planner/tool-binding steps a host-owned bridge between a compact inline observation and a richer off-context payload, and it sets up later cross-references such as `observation -> resource`, `resource -> derived observation`, or project-scoped semantic lookup without forcing the model to carry the full content inline.

- Maintain and deepen structured execution history without turning it into a second event system.

  The first bounded trace slice is implemented. It makes a turn inspectable
  through structured execution facts such as `plan -> step -> observation ->
  tool result -> reflection decision -> final answer`, while the detailed
  lifecycle remains in the typed event stream. Remaining work is limited to
  deeper trace context, optional timing/sequence metadata, and broader
  assertions in integration tests; raw reasoning must remain excluded.

- Add a file-backed host configuration model above CLI flags.

  CLI flags remain useful for PoC and admin overrides, but a near-production daemon should be constructible from a host config file that describes server settings, model load settings, profiles, storage backends, and safety/runtime limits. The runtime and daemon service layers should consume that host-owned configuration directly rather than depending on CLI `args`.

- Keep the daemon target shape centered on a resident runtime service, not a bigger CLI.

  The intended direction is still a host/service process that owns the loaded-model manager, session manager, store manager, tool execution surface, policy decisions, and trace sink, with the CLI eventually acting as a client/admin tool against that daemon rather than a parallel owner of the agent loop.

### Agent build and library structure

The agent build keeps reusable implementation out of individual smoke
executables. The current first extraction is `llama-agent-runtime-support`, an
internal library containing the shared implementation used by the CLI, daemon,
resource, runtime, and MCP-client targets. Smoke executables should normally
compile only their focused `pocs/agent/smoke/<area>/*.cpp` source and link the
library they exercise.

The dependency direction is intentionally one-way:

```text
common/*
    -> llama-agent-poc
    -> llama-agent-runtime-support
    -> CLI, daemon, MCP, and smoke targets
```

`common/agent` must not depend on implementation under `tools/agent`. Daemon
services, MCP servers, and optional Cozo storage may be split into narrower
libraries when their real dependency boundaries have been measured. Do not
create a new library solely to mirror a directory name.

Two narrow extractions now follow that rule. `llama-agent-daemon-protocol`
owns the reusable JSONL serialization/parsing implementation used by the
daemon protocol smokes and fake-daemon fixture. It is intentionally static so
that a subprocess fixture can be copied to a temporary directory without
requiring a separate DLL staging step. `llama-agent-storage-cozo` owns the
Cozo data-store implementation and is created only when `LLAMA_MEMORY_COZO`
is enabled. The runtime-support library links both narrow libraries where
needed; tests and seed tools link the Cozo library directly rather than
compiling the same data sources again.

Daemon service and host-configuration sources remain in the runtime-support
library. The narrow libraries therefore reduce repeated compilation without
moving orchestration or ownership into a lower-level utility target.

Agent support libraries follow the repository's `BUILD_SHARED_LIBS` policy. They
use target-scoped include directories, compile definitions, and link libraries;
they do not rely on smoke targets carrying the implementation source list. The
Cozo-specific implementation remains optional and must continue to be guarded
by the existing Cozo build options and explicit factory/linking contracts.

CTest labels and smoke groups are independent build and verification concepts:
the category targets (`llama-agent-smoke-runtime`, `llama-agent-smoke-mcp`,
`llama-agent-smoke-cli`, `llama-agent-smoke-daemon`, and
`llama-agent-smoke-resource`) control build grouping, while labels such as
`agent`, `sandbox-docker`, and `sandbox-kubernetes` control test selection.
New tests should preserve both structures and should not add a label merely to
make a target build.

### Adding a new test or smoke

When a new contract or externally visible seam is added, keep its first
regression proof close to the owning layer and add it to the pack deliberately:

1. Put a focused C++ smoke under `pocs/agent/smoke/<area>/` and keep reusable
   contracts/implementation under `common/*` or `tools/agent/*`; `pocs/agent`
   should remain the harness and migration/build-glue layer.
2. Add an executable in `pocs/agent/CMakeLists.txt`, link only the reusable
   support libraries and explicit fixture sources needed by the seam, and add the target to
   `AGENT_RUNTIME_TARGETS` plus its category list (`runtime`, `mcp`, `cli`,
   `daemon`, or `resource`). This makes it available to both the category
   smoke and `llama-agent-smoke-all`/`llama-agent-build-pack`.
3. If the test needs a process, socket, filesystem, timeout, or platform
   setup, add the corresponding step to both
   `scripts/test-agent-daemon-beta-smoke.ps1` and
   `scripts/test-agent-daemon-beta-smoke.sh`. Use the same stable suite name
   in both runners, clean temporary state, avoid real credentials, and return
   non-zero on failure.
4. Run the focused binary first, then the relevant category target, the
   non-smoke CTest label, and finally the beta pack. A smoke should assert
   observable contract behavior and failure cleanup, not implementation-only
   details.

For a deterministic unit-style check, prefer CTest with the `agent` label. For
daemon or transport behavior, prefer the platform runner because it owns child
process cleanup, per-suite timeouts, and diagnostic logs. Update the
verification baseline below when the new smoke has actually passed; do not
claim coverage merely because the target exists in CMake.

## Current Verification Baseline

For the authoritative branch, commit, platform, counts and known failures, see
the separate [Agent Assurance record](agent-assurance.md). This section is
intentionally descriptive rather than a second test inventory. It explains
which verification layers exist and what they are intended to cover; the
assurance record owns the current pass/fail baseline.

The verification layers are:

- **Agent CTests** cover model-free contracts and focused runtime behavior for
  agent lifecycle, thinking modes, memory, plans, reflection, tool adapters,
  tool catalog/profile resolution and sandbox contracts.
- **Runtime smokes** cover provider/tool execution, workspace operations,
  session and operation managers, deliberate/research orchestration, resource
  stores and sandbox backend contracts.
- **MCP smokes** cover stdio and HTTP client/server transport, tool discovery,
  namespacing, error mapping, resource capabilities, inbound dispatch and the
  complete vertical path.
- **CLI smokes** cover native-plus-MCP tool selection and CLI-to-runtime
  argument/host-adapter wiring.
- **Daemon smokes** cover JSONL lifecycle, status, sessions, cancellation,
  event projection, reload/configuration and daemon-side tool resolution.
- **Resource and data checks** cover scoped resource reads/writes, artifact
  references and host-owned structured data seams where the selected build
  enables them.
- **Model-backed checks** cover resident keepalive, multi-turn reuse and local
  model integration. These require an explicit local model and are kept
  separate from model-free assurance.

The following are representative paths and focused regression cases, not a
complete list of every CTest or smoke executable:

- `llama-agent-inference-ctest`, the compatibility smoke for runtime-driver,
  host, resident-session and generation-metadata coverage
- `llama-agent-deliberate-runtime-ctest`, verifying complete, incomplete, and
  conflicting resource-chunk synthesis gating in the runtime driver
- `test-tool-adapters`
- `llama-agent-tool-provider-smoke`
- `llama-agent-mcp-tool-provider-smoke`
- `llama-agent-mcp-agent-tools-smoke`, verifying the explicit opt-in `delegate_task`, `summarize`, and `review_plan` MCP tool profile, including tool filtering, write-policy rejection, delegation-depth bounds, and dispatcher callback failure mapping
- `llama-agent-mcp-http-vertical-smoke`, verifying the first complete inbound vertical path: MCP `delegate_task`, deliberate request with host escalation to research, plan creation, research gap/evidence acquisition through `memory_get` and `resource_read`, answer verification, a separate SSE event stream, and a result-only terminal response
- `llama-agent-mcp-stdio-client-smoke`, verifying the client/provider path against the reusable fake stdio MCP server core, including normal framed reads, malformed `tools/list` diagnostics, and bounded termination of a hanging server
- `llama-agent-mcp-stdio-server-smoke`, verifying the same client/provider path against the first real PoC stdio MCP server binary while exporting host-resolved tool surfaces such as `minimal`, `research`, and `research` plus configured external MCP subprocess tools
- MCP-related subprocess smokes now also rebuild their helper server targets explicitly before execution, which closes the stale-helper regression that previously surfaced as a misleading `resources/list` hang in the client smoke
- `llama-agent-tool-runtime-smoke`, verifying structured trace history across plan creation, tool execution, and final response completion
- `llama-agent-tool-runtime-ctest`, verifying that an active final answer is
  suspended while reflection repairs a failed tool step and is accepted only
  after the repair completes
- `llama-agent-tool-provider-ctest`, verifying deterministic dotted-name
  normalization, unique fuzzy resolution (for example `join` to `data.join`)
  and preservation of ambiguous candidates
- `llama-agent-tool-runtime-smoke` also verifies the current reflection guardrail for incomplete `memory_get` repairs, so that an empty or underspecified `memory_get` tool step degrades to reasoning with a specific "requires an id from a prior memory_search or recorded memory reference" diagnostic instead of surfacing only the generic schema-contract error
- `llama-agent-resource-store-smoke`, verifying the first host-owned resource/blob store contract for scoped reads, size limits, content-addressed filesystem blob reuse, and Cozo-backed resource metadata in a Cozo-enabled build
- `llama-agent-runtime-session-manager-smoke`, verifying the new per-session lane bookkeeping, internal mailbox/disposition slice, host-owned cancellation, active-turn cancel routing, reset, and close without needing a live model
- `llama-agent-runtime-operation-manager-smoke`, verifying deadline cancellation, terminal cleanup outside the operation-manager mutex, and preservation of terminal timeout state without needing a live model
- `llama-agent-runtime-session-manager-smoke` also verifies that `reset_all()` now routes through lane-owned close semantics instead of bypassing the lane lifecycle with a raw map clear
- `llama-agent-runtime-session-manager-smoke` now also verifies manager-owned parked states for both `awaiting_tool` and `awaiting_inference`, including `wait_for_tool`, `wait_for_inference`, host-driven cancellation out of the parked tool state, and resumption into host execution after the parked inference state is released
  - that same parked-turn smoke now also checks that the lane/session descriptors preserve `pending_operation_kind` and `pending_operation_detail` while a turn is actually parked
- that same parked-turn smoke now also verifies the new internal wait-entered events for both `turn.waiting_for_tool` and `turn.waiting_for_inference`
- `llama-agent-daemon-wait-events-smoke`, verifying those same `turn.waiting_for_tool` and `turn.waiting_for_inference` events survive dispatcher/service projection and appear in the final daemon command result without needing a live model
- that same daemon wait-events smoke now also checks typed `command.queued`, `command.started`, and typed wait-entered event metadata on the projected dispatcher result
  - `llama-agent-daemon-jsonl-protocol-smoke`, verifying the JSONL/admin status contract now projects lane-owned pending operation kind/detail on both the top-level active turn and the keyed session binding summary
  - `llama-agent-daemon-protocol-smoke`, verifying the daemon-side status serializer emits the same pending-operation fields before the JSONL/client parser ever sees them
  - `llama-agent-daemon-protocol-smoke` now also verifies typed daemon event metadata for service-owned `status`, `drain`, `shutdown`, and early `turn.failed` result shaping
  - that same protocol smoke now also verifies typed metadata on service-owned admin/listing/resource failure paths such as `turn.cancel_rejected`, `session.lookup_failed`, and `resource.read_failed`
  - `llama-agent-daemon-dispatcher-smoke` now also verifies typed dispatcher-side rejection/cancellation metadata for `command.rejected`, queued `turn.rejected`, queued `turn.cancelled`, and active-turn `turn.cancel_requested`
- `llama-agent-daemon-client-smoke`, verifying the child-process admin/client path renders the same pending active-turn and session-binding state through `/sessions`, `/session`, and lifecycle/admin summaries
- ordinary chat smoke with local Qwen plus Nomic embedding
- agent planning smoke with `--agent-inference-backend server-context`
- `ctest --test-dir build-agent-deliberation-vs -C Release -L agent-deliberate --output-on-failure`
- resident host multi-turn smoke with `llama-agent-resident-smoke`, verifying the same `server_context` keepalive across two turns
- resident host `n_predict` keepalive smoke, verifying the same `server_context` keepalive survives when the second turn raises the decode limit
- foreground daemon smoke with `llama-agent-daemon`, verifying ready/turn/reuse/shutdown over JSONL
- Cozo-backed foreground daemon smoke, verifying the same daemon path can open persistent memory/plan stores through the existing Cozo store factories
- daemon multi-session smoke, verifying keyed resident-lane reuse across `session A`, `session B`, `session A again`
- daemon `n_predict` reuse smoke, verifying a resident session still reports runtime reuse when only the per-turn decode limit changes
- `llama-agent-daemon-dispatcher-smoke` with a local Qwen model, verifying both queued-turn cancellation and session-lifecycle pruning of dispatcher-queued work for the same session, so `reset_session` no longer leaves a same-session queued turn waiting behind the global daemon queue
- CLI-to-daemon smoke with `llama-agent daemon-chat`, verifying the CLI can drive the same resident backend through the foreground child-process adapter
- multi-turn CLI-to-daemon smoke with `llama-agent daemon-session`, verifying the same child daemon can answer multiple prompts inside one session and scope envelope
- `llama-agent-daemon-client-smoke`, verifying the child-process client seam itself against a fake JSONL daemon subprocess while exercising `/sessions`, `/session`, `/resources`, `/memories`, `/plans`, `/resource`, `/reset`, `/close`, and `/drain` through the same stdin/stdout admin path as `daemon-session`
- multi-turn CLI-to-daemon tooling smoke with `daemon-session`, verifying the child-process adapter can carry `--tool-profile`, `--repository-root`, and MCP stdio tool wiring through to the same resident daemon session
- direct daemon integration harness with a live foreground process, verifying status, multi-turn chat reuse, reset/close-session lifecycle, project rebinding, native and MCP-configured tooling paths, and agent memory-learning flows against locally available models
- multi-turn daemon `agent` smoke, verifying runtime reuse plus stable `plan_id` reuse across two planning turns in the same resident session
- daemon `agent` learning smoke, verifying resident planning plus post-turn memory-learning summary over the same daemon session when an embedding model is supplied

The foreground daemon `agent` path is now part of the smoke baseline as well. One stabilization issue in this layer was contract drift across wrappers: the daemon request builder was correctly seeded with `server-context`, but a later policy overwrite silently fell back to the default CLI backend, and a second host-execution scope duplication made resident `agent` fragile. The current shape keeps the daemon/backend wiring explicit and reuses `turn_request.scope` as the single host-execution scope source for agent turns.

### Running Agent Tests

The current branch now supports a more targeted serial workflow for agent-heavy verification on this laptop.

The model-backed Qwen/Nomic helper `scripts/test-qwen-nomic-agent.ps1` accepts
`-ThinkingMode reflective|deliberate` for the agent turn. The dedicated
`scripts/test-qwen-nomic-agent-deliberate.ps1` wrapper selects `deliberate`
with the same model, Cozo and work-directory setup. These runs are optional
model-backed checks; the deterministic CTest and model-free smoke baseline
remains the authoritative regression gate.

The helper accepts absolute `-BuildDir` and `-WorkSubdir` values, so model-backed
checks can use external E: build and work trees without creating a second build
convention. The daemon integration harness follows the same rule. Its JSONL reader consumes
`message_type=event` deliveries, including `turn.accepted`, until the terminal
command response arrives; an accepted event is not treated as the completed
turn result. This is required by the asynchronous daemon mailbox/event-stream
contract and keeps the test harness aligned with queue ownership.

For a local Release Qwen/Nomic run, use for example:

```powershell
pwsh -File scripts/test-qwen-nomic-agent.ps1 `
  -BuildDir E:\llama-builds\agent-selection-learning-msvc-release-e2e `
  -WorkSubdir work\qwen-nomic-release-e2e -Threads 2
```

The run verifies Nomic embedding add/search plus static, agent and learning
chat. Model-produced tool repair can still leave the learning plan incomplete;
the process result and the per-step logs must be inspected separately from
the deterministic pass/fail baseline.

The small `scripts/test-qwen-resource-synthesis.ps1` smoke adds a bounded
model check for chunk synthesis through the `llama-agent` CLI. It uses the fixed
`tests/data/agent-resource-synthesis.txt` fixture, projects its four paragraphs
as bounded parent-linked observations, and asks Qwen to preserve all required
facts in one concise answer. This verifies synthesis over chunk-shaped
observations without claiming that controller-owned chunk scheduling or
persistent resource-catalog orchestration is complete; those remain covered by
the deterministic resource-store tests and later daemon integration work.

For example:

```powershell
pwsh -File scripts/test-qwen-nomic-agent.ps1 -ThinkingMode deliberate
```

The CSV join/sum scenario is intentionally a separate integration slice. It
must seed the two CSV inputs into the configured data-store scope before a
model-backed research turn can exercise `data.join` and `data.aggregate`.
The `scripts/test-qwen-nomic-agent-data.ps1` helper now accepts optional
`-OrdersCsv` and `-CustomersCsv` paths; when omitted it creates small local
fixtures, seeds Cozo, and runs the model-backed data turn.

The model-backed CSV smoke is not a substitute for the deterministic tool
tests. In the latest Qwen/Nomic run, the model produced a prose plan that
mentioned a join but the structured plan selected `dataset.inspect` with
invalid arguments and never issued `data.join`, `data.aggregate` or
`statistics.describe`. The smoke therefore failed its expected-tool
assertion, as intended. This indicates a model planning/selection gap, not a
successful join path; the host resolver cannot repair a join that the model
never requested.

The focused `scripts/test-qwen-nomic-document-table.ps1` helper exercises the
model-facing document-table path. It supplies a checked-in Pandoc-style JSON
document representation, exposes the `research` profile, enables tracing and
plan summaries, and asks Qwen to call `document.tables`, select the unique
`Budget summary` table by name, and aggregate its materialized dataset. The
script uses the normal `run` CLI command; `chat` remains accepted as a legacy
alias. The command usage documents the same `--agent-trace` and
`--plan-show-summary` options for `run`, so model-backed diagnostics use the
same host/runtime path as ordinary CLI execution.

The script is fail-closed: it requires completed traces for the expected
`document.tables`, `document.table`, and `data.aggregate` calls and rejects
tool-failure, repair-limit, or unavailable-document diagnostics. A process exit
alone is not evidence that the model actually used the tools.

For diagnostics, `--generation-trace` records bounded model generation,
token-progress details, and a bounded preview of each completed generated
content value. The preview is intended for local debugging of planner,
reflection, repair, and answer formatting; it must not be treated as an
authoritative audit log. `--agent-trace` records the corresponding bounded host
tool lifecycle: normalized tool name, argument keys, selected resource URI,
whether safe defaults were applied, failure code, and a short sanitized
diagnostic. During a `repair*` step it additionally records bounded
`model_args` and, when normalization changed them, `normalized_args`; sensitive
fields such as tokens, passwords, secrets, authorization and API keys are
redacted. It does not record unbounded raw arguments or resource contents.
Keeping these signals under the existing agent trace makes a failed
model-backed tool selection diagnosable without introducing a second tracing
channel.

For a local focused run:

```powershell
pwsh -File scripts/test-qwen-nomic-document-table.ps1 `
  -BuildDir E:\llama-builds\agent-resource-tools-msvc-debug-regen-20260812 `
  -WorkSubdir work\qwen-nomic-document-table-e2e
```

The log is intentionally retained under the selected work directory. A
successful run must show the document-table tool calls, `data.aggregate`, and
the expected total. If a small model emits a prose plan but does not select
those tools, the run is a useful model-planning diagnostic but is not counted
as a passing end-to-end document-table result.

The non-smoke CTest baseline can be run after loading the local agent build
environment:

```powershell
. .\scripts\agent-build-env.ps1
ctest --test-dir build-agent-current-ninja-debug -L agent --output-on-failure
```

The numeric result of a particular sweep belongs in the
[Agent Assurance record](agent-assurance.md), not in this architecture
document. The CTest targets retain explicit provider/runtime wiring: the
inference aggregate links the HTTP MCP client and the lifecycle/repository
tests use the same provider-backed tool-runtime adapter as production code.
This keeps a successful build meaningful after the registry-to-provider
migration.

The CMake test-pack targets provide the same layers when the agent options are
enabled:

```powershell
. .\scripts\agent-build-env.ps1
cmake --build build-agent-current-ninja-debug --target llama-agent-build-pack --parallel 4
cmake --build build-agent-current-ninja-debug --target llama-agent-ctest-pack --parallel 4
cmake --build build-agent-current-ninja-debug --target llama-agent-beta-test-pack --parallel 4
```

`llama-agent-build-pack` is the build gate, `llama-agent-ctest-pack` runs the
non-smoke `agent` label, and `llama-agent-beta-test-pack` runs the structured
deterministic/process pack. On Windows the latter uses
`test-agent-daemon-beta-smoke.ps1`; Linux uses the matching
`test-agent-daemon-beta-smoke.sh`. Both runners emit one `suite=...` result per
step, capture failure logs, enforce per-suite timeouts, clean temporary logs by
default, and return a non-zero exit code for any failed suite. Use
`-KeepLogs` or `--keep-logs` when diagnosing a failure.

Smoke binaries are grouped by the same category split already used in `pocs/agent/smoke`:

- `llama-agent-smoke-runtime`
- `llama-agent-smoke-mcp`
- `llama-agent-smoke-cli`
- `llama-agent-smoke-daemon`
- `llama-agent-smoke-resource`
- `llama-agent-smoke-all`

That means the normal build entrypoint for smoke coverage can stay narrow and category-scoped:

```powershell
. .\scripts\agent-build-env.ps1
cmake --build build-agent-current-ninja-debug --parallel 4 --target llama-agent-smoke-daemon
cmake --build build-agent-current-ninja-debug --parallel 4 --target llama-agent-smoke-mcp
```

The current build registers the agent contract tests under the `agent` label
and the sandbox backend tests under their own backend labels:

| Label | Scope | Current Cozo build |
|---|---|---:|
| `agent` | Model-free agent contracts, memory/plan contracts, tooling, Cozo data-store and agent runtime CTest smokes | 21 |
| `sandbox-docker` | Docker backend smoke; uses CTest skip code 77 when Docker is unavailable | 1 |
| `sandbox-kubernetes` | Kubernetes backend smoke | 1 |

The current build registers 65 CTest cases in total. The three labels above
account for 23 label memberships; the two sandbox labels are separate from the
authoritative `agent` test slice and are therefore not included in its 21-test
count. Other repository tests are registered without an agent label in this
build. Counts are configuration-dependent and should be refreshed with
`ctest --test-dir BUILD_DIR -N` rather than treated as a permanent contract.

Example runs:

```powershell
ctest --test-dir build-agent-current-ninja-debug -L agent --output-on-failure
ctest --test-dir build-agent-current-ninja-debug -L sandbox-kubernetes --output-on-failure
```

The repository also has a Linux-only GitHub Actions workflow at
`.github/workflows/agent-ci.yml`. It runs for non-`master` pushes and for pull
requests targeting `master` or `feature/llama-agent`, enables
the Cozo memory/plan/data path, requires both `clang` and `clangd`, builds the
agent pack with four workers, and runs the `agent` CTest label. The same job
creates an ephemeral `kind` cluster with a local-path storage provisioner and
runs the real `sandbox-kubernetes` CTest label with the Kubernetes smoke
explicitly enabled. This workflow is intentionally separate from the generic
CPU/backend workflows: Cozo provisioning, agent options and Kubernetes
preconditions are part of the agent assurance contract, not global defaults.

On a successful run, the same workflow also installs a relocatable Linux agent
package and uploads it as the `llama-agent-dev-linux-package` workflow artifact. The
package contains `llama-agent`, `llama-agent-daemon`,
`llama-agent-mcp-stdio-server`, the Cozo shared library used by this build, and
the daemon configuration examples under `share/llama-agent/examples`. The
portable `agent-config.example.json` is a complete starting configuration with
disabled network transports and placeholder paths; token values remain outside
the package and are supplied through environment variables.

The development branch also has a separate
`.github/workflows/agent-package.yml` workflow for integration packages. It is
activated by completion of `Agent CI (Linux)`. The package job then waits for
the matching `Agent dynamic analysis` run and publishes only when both
verification workflows have succeeded for the same commit on a non-`master`
branch. Pull-request workflow runs are not packaged. The package always
downloads the already-tested Linux artifact from the Agent CI run; dynamic
analysis is a required gate, not a second package source. This single trigger
avoids duplicate package runs when both verification workflows complete. It
downloads the already-tested Linux package rather than rebuilding it, adds a
small manifest and SHA-256 checksum, and publishes an artifact named with the
UTC date, workflow run id and commit provenance. The manifest records the
source branch and exact commit SHA so the package remains traceable after
download. The workflow also supports a manual run with explicit
successful CI and dynamic-analysis run IDs when the
verification workflows were dispatched manually. It deletes older artifacts
with the same development-package prefix and keeps the five newest packages.
These are short-lived integration artifacts, not versioned releases; stable
release packages remain a separate, manually tagged release process.

The manual/tagged release workflow is `.github/workflows/agent-release.yml`.
It accepts `agent-v*` tags, or can be started manually with an explicit release
tag and source ref. The workflow performs a clean Linux Cozo build, runs the
`agent` CTest label, installs the relocatable package, includes the complete
configuration and daemon/MCP examples, writes release provenance, generates a
SHA-256 checksum, and publishes the archive as a GitHub Release. This is
intentionally separate from the automatic development artifacts: development
packages are short-lived integration snapshots, while an `agent-v*` release is an
explicit versioned distribution decision.

The current `agent` label is the authoritative model-free agent CTest slice;
focused inference, memory, plan and tooling coverage is represented by the
named tests within that label rather than by separate labels in this build.
The smoke groups remain category-oriented and are built through the
`llama-agent-smoke-*` targets above.

That split is intentional: smoke groups stay category-oriented and close to the `pocs/agent/smoke/*` layout, while the longer-lived `tests/` binaries are driven through CTest labels and named scenarios.

Another small runtime stabilization in the current slice is around incomplete `memory_get` planning/repair steps. The model still sometimes proposes `memory_get` without an opaque `id`, especially in reflection-generated repair steps. The runtime now treats that as an incomplete tool call and degrades it back to reasoning with a memory-specific diagnostic rather than preserving the more generic "required contract field is missing" schema failure.

The daemon/session seam also took another small actor-shaped step in the same sweep. Session lanes already owned their own mailbox and lifecycle once a turn had reached the keyed session manager, but the foreground daemon still had a global dispatcher queue in front of that seam. `cancel_turn` already knew how to remove a queued turn there; `reset_session` and `close_session` now do the equivalent for queued turns that belong to the same session before those turns ever reach the lane. That keeps session lifecycle actions from racing with stale same-session work that was still parked in the transport-side queue.

The lane seam now also has first real parked-turn states instead of only placeholder enum values. `awaiting_tool` and the pre-host `wait_for_inference` path are still driven by a deliberately narrow manager-owned pending-operation hook rather than by a fully asynchronous runtime host, but they are enough to prove that the keyed session/lane layer can retain turn identity, expose parked dispositions in status/inspection surfaces, and resume or cancel the same turn without dropping out of the lane. That is the intended stepping stone before making tool execution or inference ownership itself more directly nonblocking.

The daemon status surface is now a little more consistent as a result. Lifecycle responses already carried a status snapshot, but that snapshot now includes the current session descriptors as well instead of leaving populated `sessions` only to the dedicated `status` command. That keeps lane state, queued count, and active/last-turn diagnostics visible through the same host-owned status shape on both explicit status requests and lifecycle/admin replies.

This baseline verifies that the runtime host and CLI adapter refactors preserve the existing synchronous behavior while making the next host boundary easier to grow.

One remaining small diagnostics gap also showed up during this sweep: the dispatcher smoke still emits a large amount of model/server diagnostics when run against a live model. That does not invalidate the protocol assertions, but it is a reminder that smoke/integration harnesses still need a cleaner split between protocol-visible assertions and backend/model diagnostics as the daemon path becomes more service-like.

### State ownership and lifecycle descriptors

The runtime now uses a small common descriptor vocabulary for inspecting existing
state without introducing a universal store or a second source of truth. The
descriptor answers the same questions for each state type: identity, owner,
scope/lifetime, persistence capability, and authoritative owner.

The current state classes are:

| State class | Current examples | Owner/source of truth | Cleanup boundary |
|---|---|---|---|
| `durable_domain` | memory, plan, resource | respective store | store policy, explicit deletion, or scope retention |
| `resident_runtime` | session lane, pending operation | session manager or operation manager | session close, operation terminal cleanup, or process exit |
| `turn_workspace` | research workspace, provisional turn work | runtime-owned workspace object | turn terminal; not persisted by default |
| `event_projection` | daemon event history and subscriptions | event collector | bounded history, unsubscribe, or process exit |

The descriptor is metadata only. It does not replace resource, plan, memory,
session, operation, or event stores. Events describe transitions and are not the
authoritative current state.

Existing types expose read-only description helpers for this boundary:

- `describe_common_plan(...)`
- `describe_agent_resource(...)`
- `describe_common_runtime_operation(...)`
- `describe_agent_runtime_session(...)`
- `describe_common_agent_research_workspace(...)`

Research gap and task status changes use validated transition helpers. The
helpers allow explicit retry/reopen paths and reject invalid jumps such as a
failed task directly becoming completed or an abandoned gap becoming active.
This is validation around the current vectors and controller; it is not yet a
workspace persistence layer.

The ownership rule remains:

> One state object has one owner and one source of truth. Other layers keep IDs,
> references, or bounded projections.

Lane state follows the same ownership rule. The lane turn driver is the sole
writer of `active_turn` and the lane-owned `pending_operation` while a turn is
being drained. The lane mutex protects snapshots, cancellation bookkeeping,
transition handoffs and the helper's state mutations. Pending-operation
callbacks and host execution run outside that mutex. Session reset and close
first mark the lane transition, wait for the current lane message to finish,
and only then clear or erase lane state. This keeps external lifecycle calls
from racing a driver-owned continuation while avoiding a mutex held across
external work.

### Event taxonomy and overflow metadata

Daemon events retain their stable `event_type` names, such as
`turn.completed` or `tool.failed`, and now also expose an explicit
`event_category`. Categories are derived centrally from the typed event kind:
`command`, `turn`, `inference`, `tool`, `memory`, `plan`, `observation`,
`resource`, `session`, `daemon`, `config`, `mcp`, and `agent`. The category is
diagnostic metadata; it does not replace the event kind or change routing and
filter semantics.

Turn and inference lifecycle events are deliberately distinct. `turn.started`
identifies the initial turn start, while `turn.resumed` identifies continuation
after a pending tool operation. `inference.started` is emitted after the host
inference task has been submitted and registered, and `inference.completed` is
emitted when that manager-owned inference operation becomes ready. These events
do not replace the existing `turn.waiting_for_inference`, `inference.queued`,
or `inference.capacity_granted` events; those continue to describe admission
and suspension boundaries. The event names are additive to the JSONL event
projection, so clients can handle the new lifecycle kinds while retaining the
existing `event_type`, `event_category`, and context fields.

An event-stream overflow is no longer represented only by a cursor jump. An
overflow delivery carries `from_sequence`, `to_sequence`, and
`skipped_sequence_count` metadata. JSONL clients receive that metadata under
an `overflow` object, for example:

```json
{
  "message_type": "event",
  "delivery_kind": "overflow",
  "cursor": {"after_sequence": 42},
  "overflow": {
    "from_sequence": 39,
    "to_sequence": 42,
    "skipped_sequence_count": 4
  }
}
```

The sequence range describes the bounded replay gap, including sequence
positions that did not match a subscription filter. Clients should resume
from the delivery cursor and use the metadata for diagnostics or resync
decisions rather than treating the skipped range as recoverable event data.

The collector also exposes two consumption models with deliberately different
semantics. The legacy global `take()` API drains the process-wide pending-event
queue destructively. It does not advance subscription cursors, does not remove
events from bounded history, and is therefore not an alternative subscription
for resumable consumers. Subscriptions have their own cursor, replay history,
filter, and bounded pending queue. New consumers that need independent
observation or reconnect/resync behavior should use a subscription; callers
that use `take()` must own the single-consumer coordination for that global
queue.

Operation kind is currently shared by admission and execution for compatibility
(`inference` may mean either waiting for inference capacity or executing the
host inference turn). Event types and operation details distinguish the two
phases today, such as `inference_queued` versus `turn_waiting_for_inference`.
A future wire-compatible refinement may expose an explicit operation phase, but
clients should not infer admission state from `kind` alone.

Inference task ownership is explicit at the async boundary. After a successful
submit, the inference task owns the capacity lease; if operation registration
fails, the driver requests task cancellation and does not release that lease a
second time. A deadline marks the operation `timed_out` and invokes its
registered cancellation callback outside the operation-manager mutex. Terminal
operation entries are removed under the mutex but destroyed afterward, so
destruction of an asynchronous task cannot block unrelated operation-manager
calls. The reference executor still uses `std::async`; a bounded inference
worker pool remains a later implementation phase.

The current cleanup policy is intentionally conservative. Turn research state
is released with the turn, terminal operations are removed by operation-manager
cleanup, session lanes are removed on session close, resources follow their
scope and expiry metadata, plans and memory follow their stores' retention
policy, and event history remains bounded by the collector. Checkpointing a
research workspace is deliberately left for a later step.
