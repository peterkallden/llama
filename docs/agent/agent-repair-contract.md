# Required-tool repair contract

This document is the model-facing and host-facing contract for repairing a
failed mandatory tool step. The implementation may build the concrete JSON
schema at runtime, but the ownership and validation rules below are stable
contract rules and must not be inferred from a `.cpp` implementation.

## Responsibility boundary

```text
provider/tool host
    -> validates the canonical call
    -> records the failure and repair context
    -> classifies whether retry is safe

reflection model
    -> supplies corrected semantic arguments only

runtime
    -> accepts one bounded replacement
    -> canonicalizes and validates it again
    -> reruns the failed mandatory step
```

The model does not choose the failed step identity, the registered tool
identity, provider authority, scope, credentials or retry permission. Those are
host-owned facts.

## Host failure record

A validation failure that may be repaired carries the normal failure envelope
and a host-created `repair_context`:

```json
{
  "code": "openapi.required_parameter_missing",
  "class": "validation",
  "retryable": true,
  "stage": "tool_execution",
  "step": "step_1",
  "tool": "openalex.listWorks",
  "summary": "A host-required OpenAPI argument is missing",
  "repair_context": {
    "tool": "openalex.listWorks",
    "error": "OpenAPI required operation parameter is missing: listWorks.search",
    "arguments_skeleton": {},
    "compact_contract": "openalex.listWorks ... args: cursor?:string; per_page?:integer; search:string; select?:string"
  }
}
```

The exact provider wording is not model authority. The host uses the failure
class, `retryable` flag, failed mandatory step and registered tool contract to
construct the repair request. If those facts are absent, contradictory or not
retryable, the runtime must remain fail-closed.

## Model-facing schema versus execution schema

The normal model contract must already contain host-known mandatory inputs.
For OpenAPI operations, the provider derives a model-facing projection from
the operation schema and the configured `host_required_parameters`. The
projection is rendered by the existing compact tool-contract generator and is
also used as the argument schema exposed in the resolved chat-tool view. This
means ordinary planning can provide `search`, paging and other host-required
values before a repair is needed.

The projection does not rewrite the execution contract. Host validation,
normalization and provider execution continue to use the canonical
`input_schema_json`; the model-facing projection is only a stricter prompt
and argument-description view. A configured host-required name that is absent
from the canonical schema is a provider/catalog error, not a reason to invent
an untyped field. Native tools already use the same full-schema versus
model-schema distinction, and other providers may populate the optional
projection when they have equivalent host-owned requirements.

## Model-facing repair response

For a retryable failed mandatory validation, reflection receives a narrower
schema than ordinary reflection. It must return exactly one replacement:

```json
{
  "decision": "revise",
  "replace_steps": [
    {
      "step_id": "step_1",
      "tool": "openalex.listWorks",
      "args": {
        "search": "machine learning",
        "per_page": 1,
        "select": "id,display_name"
      }
    }
  ]
}
```

The concrete schema binds `step_id` to the failed host step and `tool` to the
already registered tool, normally by a single-value `enum`. It also imports
the registered tool's argument schema and promotes the actually missing
host-required argument to `required` when that property is present. The model
therefore supplies the value, but cannot redirect the repair to another step
or another operation.

The response contract has these invariants:

| Field | Owner | Invariant |
| --- | --- | --- |
| `decision` | host schema | exactly `revise` |
| `replace_steps` | host schema | exactly one item |
| `step_id` | host schema | exact failed mandatory step |
| `tool` | host schema | exact registered operation |
| `args` | model within host schema | semantic values satisfying the registered contract |
| provider scope and authority | host | never model supplied |

Prose-only guidance, `retry`, `reset`, `abort`, an unrelated added step or a
second replacement is not a repair for this failure class.

## Acceptance and rerun

After structured parsing, the runtime still performs the ordinary repair
pipeline:

1. match the replacement to the host-bound failed step;
2. merge only compatible prior arguments;
3. normalize the model-facing call to the canonical `{tool,args}` form;
4. validate policy, scope and the provider contract;
5. rerun the failed mandatory step within the existing retry bound;
6. unblock final synthesis only after the mandatory step completes.

Every invalid replacement remains an observed failure. It must not produce a
user-facing draft from an unrepaired plan, learning credit, promotion evidence
or activation authority.

## Dependency gating during repair

Tool repair is local to the failed step, but execution is dependency-aware.
If a mandatory step `B` depends on `A`, a failed or incomplete `A` prevents
`B` from executing. This remains true even if a restored or model-authored
plan has already marked `B` active; the runtime rechecks the shared
dependency-ready predicate before execution and defers the active dependent
step back to pending.

```text
A fails with a retryable tool-validation error
    -> A is the repair target
    -> B and other dependents remain deferred
    -> reflection replaces or resets A
    -> A is validated and rerun
    -> only then may B execute with fresh producer evidence
```

The dependency gate is generic and does not name a provider or tool. It
applies equally to native, MCP and OpenAPI calls. Independent branches retain
their existing scheduling policy. A non-retryable failure still blocks its
dependents and fails closed; it is not converted into a repair merely because
later steps exist.

Dependency ownership is also host-safe at plan parsing. Model-emitted step
identifiers do not by themselves declare an independent graph. When a compact
proposal omits both `after` and `depends_on`, the host assigns bounded step
identities and sequential dependencies. Explicit dependency fields are
retained and validated as the advanced form.

## Non-retryable failures

An unavailable tool, policy denial, scope violation, malformed provider
response, missing host identity or a failure explicitly marked non-retryable
does not enter this repair schema. It follows the ordinary bounded reflection
or abort path. A model cannot turn a non-retryable failure into a retry by
changing JSON fields.

## Provider neutrality and compatibility

This contract is an orchestration seam, not an OpenAPI-only feature. Native,
MCP and OpenAPI adapters may provide different host failure details, but a
retryable mandatory validation failure enters the same host-bound repair
boundary. Compatibility normalization may accept bounded older argument
wrappers, but it must end in the same canonical call and must not create a
second repair scheduler or authority model.

The repair path changes execution control only. It does not change tool
semantics, planning policy, reflection meaning, evidence rank, FlyDelta
learning credit or promotion rules.

## Planner diagnostics and model-facing result contracts

The `--agent-trace` option also exposes a bounded `planner_candidate` record
for each structured planner attempt. It reports the attempt number, generation
status, stop reason, decoded-token count, candidate byte count, acceptance
status, parser error and a bounded single-line candidate preview. This is
host-side observability; it does not relax parsing or repair malformed JSON.
The raw candidate is intentionally not part of the durable plan or learning
record.

When `require_tool_execution` is active, the planner prompt and its
model-facing JSON schema describe the same contract: each proposed step is a
registered tool with an ordinary JSON `args` object, and no reasoning or final
step is model-owned. Before a planner candidate is accepted, host-known
required argument names are checked against that model-facing schema; a missing
field is a planner-candidate rejection and bounded regeneration, not an
execution attempt with an incomplete call. When tool execution is optional,
the bounded reasoning form remains available. Host-owned step IDs, dependency
edges and final synthesis remain outside the model-facing contract.

Provider result schemas follow the same projection boundary. An OpenAPI
response schema is available to the compact planner contract as a
model-facing result description, while canonical execution results remain
host-owned. If the response schema describes fields such as
`results[].id`, a later model-facing binding may refer to that typed output;
the provider does not invent fields that are absent from its OpenAPI schema.
