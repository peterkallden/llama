# Host verification and bounded repair

Implementation status (2026-10-25): source changes and regression fixtures are
written, but this change has deliberately not been compiled or tested. The
previous baseline passed eleven focused CTests and failed a traced physical
Vulkan Qwen/Nomic workflow-action smoke. Those results do not validate this
implementation. Daemon restart and browser verification remain pending.

## Authority and scope

The model selects the meaning of a request and proposes useful investigations.
The host validates references, argument bindings, authorization, declared result
conditions, evidence anchors and execution limits. Schema validity does not
establish that a selected dataset or document answers the user's question.

These changes extend the existing runtime, workflow verification, plan store,
DSL parser, data store and evidence projection. They introduce no user-input
word lists, intent shortcuts, alternate scheduler or evidence store.

## Result contracts and numeric defaults

`common_agent_resolve_numeric_columns` uses descriptor types for automatic
numeric-column selection. An explicit list can identify columns whose schema
types are unknown; missing types never silently become inferred numeric types.
The data adapter reports missing automatic selection as a retryable validation
failure. Backends retain their declared capabilities; SQLite does not gain a
statistics implementation through this change.

`statistics.describe` returns the executed dataset and `selected_columns` in
addition to its existing measures, groups and scan metadata. The adapter checks
these bindings and the registered result schema. Known numeric columns over
zero rows produce count zero and a null mean. Unknown selection is distinct
from that valid empty-input result.

The CSV smoke seed now retains the types of the values it writes, with processor
version 2. Separate Cozo contracts retain an unknown-schema scenario so the
corrected fixture cannot conceal missing-type behavior.

## Bounded evidence presentation

The existing tool-observation projection retains evidence identities, explicit
empty collections, null measures and retained-source truncation status. It
keeps complete assertion lines rather than cutting values or identifiers.
Recent successful observations receive priority; retained observations are
then shown in chronological order. Omitted facts and observations have an
explicit marker, and the outer evidence boundary stays intact.

A successful tool call is described as successful execution, not verified task
completion. A shortened projection does not imply complete source coverage.
Original resources are referenced only when they actually exist. Canonical
host verification continues to use structured observations/results rather than
parsing this model-facing text. Retained observation budgets can themselves
limit available evidence; a compact renderer cannot recover discarded data.

Retry identities such as `tool:<step>:<operation>:attempt:2` use the shared
step-ownership resolver. This prevents a failed/retried step from appearing as
orphaned completed evidence and prevents replay from using another step's
successful result.

## Review lifecycle

Ordinary compact reflection can return:

```text
reflect accept
```

or a bounded correction:

```text
reflect revise
assurance_action revise_plan
guidance "Read the missing source before answering."
```

The existing assurance actions distinguish `revise_response`, `revise_plan`,
`fail_bounded` and authorized escalation requests. JSON mode uses the same
normalized actions. Legacy bare `reflect revise` remains a response revision;
it never means acceptance. Contradictory readiness/action fields are rejected.
Malformed review preserves a controlled error instead of falling back to
acceptance.

`max_reflection_rounds` counts actual review evaluations in a turn. A revised
draft needs another available review. The default is unchanged: when only one
review is configured and it rejects the draft, the runtime stops without
generating an unreviewable rewrite. `max_tool_batches` is shared across initial
execution and subsequent repair slices. No automatic budget increase is added.

The host retains an explicit synthesis step for executable plans that lack one.
Completing tool steps alone cannot complete an answer awaiting verification.
Negative review records contain draft/evidence fingerprints and remain in the
plan store. A successful later review resolves that state. These records are
host-authorized; model plan operations cannot forge them. A pending negative
review blocks synthesis/plan completion, including replay with reflection
disabled. Verified report-export completion uses its existing host-owned path.

For a one-action route, evidence repair calls the existing planner with the
same request scope, route and fixed argument bindings. It accepts bounded
additional read-only work, remaps step/dataflow identities, validates the plan
against the current tool view and previews ordinary plan-policy checks before
publication. Existing successful operations remain history; writes are not
silently repeated. The final answer depends on the repair steps.

For a directed workflow, only the existing host continuation provider can
return additional declared steps. If the graph has no applicable continuation,
the runtime stops with that limitation rather than inventing a graph edge or
expanding tool authority. Denied policy, invalid model format, cancellation and
budget exhaustion remain distinguishable diagnostics. Unaccepted results do
not become successful completion/learning examples.

## Operation verification and numeric evidence

`verification_checks` can select a canonical `tool` in `before_tool` or
`after_tool` checks, allowing the same verifier to protect a one-action route.
Observation/argument operands can likewise select a tool. Existing transition
checks retain their identity guards. `before_answer` continues to use its
declared operands rather than an implicit current tool. The shared package,
Cozo and SQLite workflow codec preserves these additive fields.

`declared_claim_tools` requests structured numeric claims when a declared
operation has produced successful evidence in the current workflow segment.
Later unrelated reads cannot silently remove that obligation. Routes that never
execute a declared operation and ordinary conversation do not acquire it.
Existing dataset-cell assurance can still use the latest retained dataset even
when a later verification-only operation returns no dataset.

Inline numeric results can expose bounded `result_field` handles anchored to
the actual observation, operation and JSON pointer. They do not acquire invented
units, dimensions or dataset URIs. Existing dataset-cell claims still require
their actual units, dimensions and usable status. The DSL draft grammar binds
declared handle/value pairs to host evidence; explicit JSON mode is checked
against the same host handles. Empty or null measures cannot become numeric
zero through this mechanism.

Claim verification covers explicitly declared facts. It does not prove every
sentence of free prose correct, nor establish that the model chose the user's
intended source. Actual task interpretation remains part of model evaluation.

## Starter revision and compatibility

The checked-in starter package publishes dataset analysis/quality workflows at
revision v3, under new workflow record IDs. They declare the selected-column
condition for `statistics.describe` and conditional numeric-claim assurance.
The dataset blueprint retains its identity and binds those v3 definitions.
Previously stored v2 workflows remain readable with their original semantics.
Import the updated starter package to use these declarations; editing source
does not reload the running daemon or its imported library.

## Authored verification, not executed

- `test-agent-planner-review`: exhausted review, revised-draft review, invalid
  review, additional code inspection and explicitly disabled reflection.
- Plan-context, reflection and workflow-grounding contracts: release-active
  checks, retry identity, empty/truncated evidence, DSL/JSON parity, operation
  selectors, persistence codec, typed scalar anchors and document results.
- Cozo contracts: unknown schema, explicit selection and typed empty results.
- Existing tool-runtime smoke: provider/resource/runtime execution with bounded
  negative review and independently accepted rewriting, with diagnostic trace.
- Qwen/Nomic workflow-action smoke: separate argument-contract and executed
  numeric-evidence checks for count 3 and mean 40/3 from `orders.amount`.

The provider smoke is model-free and does not prove daemon/browser behavior.
No new UI controls are required by this change. Live daemon source selection,
restart/resume, cancellation and browser evidence presentation still need the
planned production-path verification after compilation is authorized.
