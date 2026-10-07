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

### Model-facing output dialects

The host may select the model-facing tool output format per model profile.
Public choices are `json` and `dsl`. `json` maps to the internal native
structured-output path. The older textual `jsonl` dialect remains available
when loading legacy configurations:

```json
{"tool":"openalex.listWorks","arguments":{"search":"machine learning","per_page":1}}
```

Compact DSL keeps nested serialization out of the model-facing argument form.
The model writes the exact tool name, then one flat field per line:

```text
open! data.aggregate
dataset: $inspect.dataset
group_by: region
measure: sum amount total_amount
measure: count "*" order_count
```

Schemas may declare repeatable directives with `x-agent-flat`, including the
directive name and positional slot order. Positional slots are the default;
the equivalent named form is also accepted:

```text
measure: function=sum column=amount as=total_amount
```

For unannotated nested schemas, use dotted/indexed field paths such as
`operations[0].column: amount`; repeat scalar-array fields on separate lines.
Quote values containing whitespace. Omit unset fields rather than writing an
empty value. The shared schema-aware parser normalizes flat input to the same
canonical `{tool,args}` object used by JSON/native calls, and ordinary host
schema validation remains authoritative. The previous nested one-line DSL is
still accepted for compatibility but is no longer the generated prompt form.

Workflow-action selection uses this same flat codec, but compiles its response
grammar from the currently eligible tools and their schema fields. Its prompt
shows only the selected tool's purpose and actionable argument fields—not
return schemas or nested serialization examples. Host-resolved dataset
arguments are excluded from the grammar and removed if a model nevertheless
emits one. A `null` value for an optional flat field is normalized to omission;
empty strings are not treated as missing values.

The planner also accepts Compact DSL for a complete multi-step plan:

```text
plan goal="Aggregate selected rows"
step as=inspect | open! dataset.inspect
resource: r1
step as=summary | open! data.aggregate
dataset: $inspect.dataset
measure: sum amount total_amount
```

Each step may continue with flat argument lines until the next `step` line.
Steps execute in sequence; optional `as=alias` names a prior result for
`$alias.field` references. The planner decodes this form into its ordinary bounded plan
proposal, then runs the same host argument, binding, route-envelope and
execution checks as JSON plans. JSON remains a valid planner format and is
selected whenever the model profile uses `json` or legacy `jsonl`; Compact DSL is
selected explicitly with `tool_output_format: "dsl"`. It does not
replace or bypass host policy.
The foreground CLI exposes the same explicit choice through
`--tool-output-format json|dsl`; its default is `dsl`. The legacy values
`native`, `jsonl` and `compact_dsl` are still accepted when reading older
configurations and invocations. Internally, `json` maps to the native tool
output mode and `dsl` maps to the compact DSL codec.

The selected output dialect applies to planning and reflection: reflection uses
the same effective `json` or `dsl` preference, with JSON-schema generation for
JSON and a bounded reflection DSL for DSL. Tool identity, provider authority,
host-required arguments, schema normalization, policy, repair execution and
learning/evidence semantics remain shared.

When the host already knows a concrete argument value from a structured
request, workflow slot or fixture, it may attach a
`common_agent_tool_argument_binding` to the turn. The binding is generic across
native, MCP and OpenAPI tools:

```json
{
  "tool": "openalex.listWorks",
  "fixed_args": {
    "search": "machine learning",
    "per_page": 1,
    "select": "id,display_name"
  },
  "source_ref": "fixture://openalex/search-machine-learning"
}
```

The planner may omit these fields. The host merges the fixed arguments before
planner validation and again at runtime before provider validation, so plans
from reflection or another planner receive the same protection. A conflicting
model value is rejected; the host never silently overwrites it. Bindings are
not inferred from arbitrary prose and must not be used to fabricate semantic
values. `model_visible` controls whether the concrete value is rendered in the
planner context; the host binding remains authoritative in either case.

## Model-facing repair response

Reflection follows the turn's selected output dialect. Its ordinary model-facing
form is deliberately not a plan-edit language. With no failed mandatory tool,
the model receives verified observations, the user request and the draft, then
returns exactly one of:

```text
reflect decision=accept
```

or:

```text
reflect decision=revise
guidance "one short correction for the answer"
```

The host owns plan IDs, tool calls, dependencies and final synthesis in this
ordinary review. It does not render tool contracts or ask the model to supply a
`STEP_ID`. A failed mandatory step narrows the DSL instead to `abort` or a
single `retry STEP_ID`, where the allowed failed IDs are explicitly listed.
The broader parser still accepts bounded plan operations for persisted or
legacy callers, but they are not the default prompt surface.

For a retryable failed mandatory validation, reflection receives a narrower
contract than ordinary reflection. The JSON form is:

```json
{
  "decision": "revise",
  "replace_steps": [
    {
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

The host binds the replacement to the failed step and registered tool; neither
internal identifier is requested from the model. The JSON schema therefore
omits `step_id` and constrains `tool` to the registered operation. It also imports
the registered tool's argument schema and promotes the actually missing
host-required argument to `required` when that property is present. The model
supplies only the corrected semantic argument values and cannot redirect the
repair to another step or operation. In DSL mode the equivalent response is:

```text
reflect decision=revise
replace | open! openalex.listWorks search="machine learning" per_page=1 select="id,display_name"
```

The response contract has these invariants:

| Field | Owner | Invariant |
| --- | --- | --- |
| `decision` | host schema | exactly `revise` |
| `replace_steps` | host schema | exactly one item |
| replacement target | host | exact failed mandatory step; omitted from model output |
| `tool` | host schema | exact registered operation |
| `args` | model within host schema | semantic values satisfying the registered contract |
| provider scope and authority | host | never model supplied |

Prose-only guidance, `retry`, `reset`, `abort`, an unrelated added step or a
second replacement is not a repair for this failure class.
If structured reflection remains invalid after its bounded regeneration, the
runtime aborts rather than accepting a draft while a mandatory tool step is
still failed.

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
field is a planner-candidate rejection and enters the dedicated planner
argument-repair form below, not an execution attempt with an incomplete call.
When tool execution is optional, the bounded reasoning form remains available.
Host-owned step IDs, dependency edges and final synthesis remain outside the
model-facing contract.

### Singleton operation selection

When required tool execution exposes exactly one effective host-bound
operation, a bounded native tool-call phase may be used before the ordinary
JSON planner. This phase is only an operation selector: it must emit exactly
one call to the already registered operation. It does not create a second
executor, scheduler or plan authority.

The host removes configured binding names from the native selector's required
fields and materializes those values after the operation name has been
selected. If the model emits a value for a host-bound field anyway, that value
is discarded in this phase. The ordinary planner and provider validation still
reject conflicting values when they receive a model-owned argument. A missing
or malformed native call falls back to the bounded JSON planner projection;
neither path bypasses host binding, argument validation, tool observation,
follow-up, reflection or required-tool completion.

The singleton JSON projection is limited to one executable step and a bounded
goal. These limits reduce output surface for small instruct models; they do
not change the persisted plan representation or the semantics of native,
MCP or OpenAPI execution.

### Planner argument repair before execution

This is distinct from the reflection repair of a failed executed step. It is
used only when the planner has already selected a registered tool and the host
can identify a required argument that is absent from that candidate. The
existing planner step and tool identity remain host-owned; the bounded repair
generation receives only this model-facing shape:

```json
{
  "args": {
    "search": "machine learning"
  }
}
```

The runtime builds the `args` schema from the selected tool's registered
model-facing schema and promotes the identified missing property to
`required` when that property is declared there. The repair prompt includes
the selected operation, its compact contract, the existing arguments, the
original user request and the host validation error. The model may provide
semantic values that are present in that context, but the host does not invent
or infer a missing value on the model's behalf.

On acceptance, the host merges the returned patch into the rejected step's
existing arguments, normalizes it through the existing planner argument
contract, replaces only that step's `args`, and runs the ordinary planner
validation again. The response cannot select another tool, add a step, replace
the plan, or carry arbitrary top-level fields. The repair consumes the existing
single bounded planner regeneration; it is not a second scheduler or an
unbounded retry loop. The planner generation budget is configurable
independently from ordinary response generation; when it is omitted, the
established planner fallback budget remains in effect. If the repair is invalid
or still incomplete, required tool execution fails closed before dispatch.

This path intentionally reuses the provider's model-facing projection. For
OpenAPI, host-required operation parameters therefore appear in the same
dynamic schema that ordinary planning sees; native and MCP tools use their
registered model input schema in the same way. Execution still uses the
canonical provider validation and input contract after planning.

Provider result schemas follow the same projection boundary. An OpenAPI
response schema is available to the compact planner contract as a
model-facing result description, while canonical execution results remain
host-owned. If the response schema describes fields such as
`results[].id`, a later model-facing binding may refer to that typed output;
the provider does not invent fields that are absent from its OpenAPI schema.
