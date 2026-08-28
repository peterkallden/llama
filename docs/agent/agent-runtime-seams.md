# Agent runtime seams and ownership

This document describes the integration seams that connect the host, compact
model contracts, plan runtime, native tools, MCP providers, OpenAPI providers,
resources and datasets. It is a companion to [Agent Runtime](agent-runtime.md)
and records the ownership rules that keep those components from becoming
parallel orchestration systems.

The central rule is:

```text
model proposes meaning
host resolves identity, scope and authority
runtime validates and executes the canonical operation
```

The model may propose a tool, a name, a column or a relationship. It does not
choose a filesystem path, dataset URI, provider credential, network target,
scope, backend or execution permission.

## Boundary map

```text
JSON configuration and host policy
          |
          v
effective provider/tool catalog
  native + MCP + OpenAPI, filtered by profile/capabilities
          |
          v
host-owned turn snapshot
  scope + resources + dataset inventory + tool view
          |
          +--> family preflight --> ordinary chat/tool driver (narrow fast path)
          |
          +--> compact planner --> normalized plan --> tool runtime
          |                                      |
          |                                      +--> native adapter
          |                                      +--> MCP provider
          |                                      +--> OpenAPI provider
          |
          +--> deliberate/research review and bounded synthesis

resource refs, dataset_refs, observations and events
          ^
          |
host-owned result normalization and materialization
```

There is one runtime execution path. Family routing, workflow slot filling and
provider adapters are projections or adapters around that path; they do not
grant authority or introduce a second plan representation.

## Helper placement and ownership

Cross-cutting helpers follow the contract they interpret, not the caller that
happens to need them. This keeps the implementation discoverable and prevents
CLI, daemon and provider layers from growing parallel versions of the same
rule:

| Helper responsibility | Home | Must not depend on |
| --- | --- | --- |
| Agent scope, turn identity and request contracts | `common/agent/` | CLI, daemon or a provider |
| Dataset refs, dataset URI, provenance and dataset scope | `common/agent/` dataset contract area | OpenAPI or a transport |
| Resource identity and read authority | `common/resource/` | Tool selection or model prompts |
| Runtime inventory and policy snapshots | `tools/agent/runtime/` or a shared common contract when transport-independent | CLI-specific defaults |
| Tool exposure facts and policy projection | `tools/agent/tooling/` | One particular transport |
| OpenAPI response materialization | `tools/agent/openapi/` | Native/MCP implementation details |

Helpers should have names that state the contract they serve, such as
`dataset-uri`, `dataset-ref-json`, `agent-turn-inventory` or
`agent-tool-policy`. Avoid generic `utils` files for rules that affect
authority, binding, persistence or model-visible output.

The distinction is important:

```text
same meaning + same safety invariant
    -> one shared helper at the lower common layer

same meaning + different boundary representation
    -> shared codec/projection with an explicit representation name

same-looking condition + different authority
    -> separate helpers, composed at the owning boundary
```

For example, native and MCP tool filtering should share normalized exposure
facts and policy predicates, but retain transport-specific authentication and
caller-session checks. Likewise, a full dataset checkpoint projection must not
be silently substituted for a compact model-facing dataset inventory.

Every new helper should answer three questions in its header or documentation:

1. Which canonical object or contract does it interpret?
2. Which layer owns the decision and which layers only consume the result?
3. Is its output full, checkpoint-safe, model-facing, or another bounded
   projection?

This is deliberately not a single universal validator. Small helpers at the
correct level are preferable to a central utility that accumulates CLI,
daemon, provider and persistence policy.

## Identity and representation vocabulary

The following values look similar in logs but have different owners:

| Value | Meaning | Owner | May the model invent it? |
| --- | --- | --- | --- |
| `resource://...` | Canonical identity for bounded source bytes, MIME type and lineage | Resource authority | No |
| `dataset://...` | Canonical identity for a structured analytical view | Dataset/data authority | No |
| `r1`, `r2`, ... | Prompt-only handles for current-turn attachments | Host prompt projection | No; select from the presented list |
| `s1`, `s2`, ... | Prompt-only handles for host-listed scoped resources | Host prompt projection | No; select from the presented list |
| `d1`, `d2`, ... | Prompt-only handles for the current host dataset snapshot | Host prompt projection | No; select from the presented list |
| `$datasets.datasets[index].dataset` | Stable reference into that host snapshot | Planner binding normalizer | No; use only an advertised index |
| `$step.field` | Binding to an earlier completed plan step | Plan runtime | No; the producer must exist and be complete |
| `dataset_refs` | Host-generated output references for later steps | Tool/result contract | No |

The prompt handles are not storage URIs. They are resolved by the host before
an adapter receives the call. `dataset://` is the canonical dataset scheme;
there is no second `agent-dataset://` identity in the current design.

## Seam 1: configuration to effective tool catalog

Configuration, command-line options and provider fragments are inputs to the
host configuration loader. The loader produces one validated effective
configuration and one host-owned provider/tool catalog:

```text
main JSON + include_dir fragments + CLI policy
    -> validated host configuration
    -> active profile and capability map
    -> native/MCP/OpenAPI provider resolution
    -> immutable per-instance tool view
```

Provider fragments are expanded before validation; they are not merged into a
second runtime configuration. Provider IDs must remain unique across inline
and fragment definitions. A reload validates a complete candidate and keeps the
previous active configuration if the candidate is invalid. See
[Agent configuration fragments](agent-config-fragments.md).

The active profile, capabilities, network policy, write policy and provider
availability are host decisions. A caller may narrow an exposed view where the
protocol allows it, but cannot select a host profile or widen the view. The
generated family index is made from this already filtered view, never from the
raw provider configuration.

### Failure pattern at this seam

The dangerous failure is a catalog that is visible in diagnostics but not
executable, or a model-facing catalog that is wider than the executor's
authority. The invariant is:

```text
model-facing tools == executable tools == host-approved tools
```

The catalog, provider and tool-view tests should therefore check both exposure
and execution, not only that a definition can be parsed.

## Seam 2: host turn selection to runtime request

The host assembles the turn scope and discovers current inputs before creating
the runtime request. The request must carry the same scoped snapshot that was
used to prepare the model context:

```text
authenticated request
    -> bind namespace/project/session/turn
    -> resolve current resources and scoped resource candidates
    -> list usable dataset descriptors
    -> resolve tool profile and family view
    -> common_agent_runtime_driver_execution
    -> common_agent_request
```

The important fields are:

```text
input_resources       current-turn attachments
available_resources   host-listed r/s resource candidates
available_datasets    host-listed dataset descriptors for this turn
scope                 namespace, project, session and turn authority
tooling               resolved tool view and execution policy
```

`available_datasets` is a request field, not a model-generated binding. The
runtime planner renders it as a bounded `<runtime_dataset_inventory>` context,
including the host name, canonical URI and the advertised
`$datasets.datasets[index].dataset` reference. A host that populates the
inventory but drops it while rebuilding `common_agent_request` creates a
misleading partial success: the model can emit a plausible dataset reference,
but the binder has no declared candidates. The runtime request builder must
therefore copy resources, datasets and scope as one snapshot.

The same rule applies to attachments. A single current attachment may be the
default source for an inspection request. Multiple attachments or multiple
host-owned matches require an explicit selection. An unknown or ambiguous name
produces a bounded `Choose one of` diagnostic; it is never interpreted as a
filesystem path.

## Seam 3: compact model proposal to canonical plan

The model-facing planner contract is intentionally smaller than the execution
contract. The canonical compact plan shape is:

```json
{
  "goal": "inspect the selected dataset",
  "steps": [
    {"tool": "dataset.select", "args": {"name": "orders"}, "as": "orders"},
    {"tool": "data.query", "args": {"dataset": "$orders.dataset"}}
  ]
}
```

For planner output, `args` is an ordinary JSON object. It is not a JSON-encoded
string. The internal `agent_tool_call.arguments_json` field is the serialized
transport representation used after normalization and validation. It is not a
second model-facing plan field. A provider executor receives that canonical
serialized object through its adapter contract.

The compatibility boundary may unwrap bounded older wrappers such as nested
`tool`/`arguments` shapes, but it must end in the same canonical `{tool,args}`
operation. Compatibility normalization must not create another scheduler,
another binding language or another execution path. Unknown tools, malformed
arguments and references outside the host view remain failures.

The host owns generated step identity, dependency order, canonical bindings,
scope and defaults. The model owns semantic choices such as a dataset name,
columns, a filter or an OpenAPI operation's business parameters.

### Binding rules

Binding is valid only when the producer is available at the point of use:

```text
dataset.list       -> discovery result
dataset.select     -> selected dataset binding
data.join          -> left and right completed dataset bindings
data.aggregate     -> one completed dataset binding
statistics.*       -> one completed dataset binding
```

A direct host dataset URI may satisfy a step without a preceding selection
step. An indexed `$datasets.datasets[index]` reference is valid only against the
inventory snapshot in the same request. An out-of-range index, unknown name or
ambiguous name fails with bounded candidates rather than a guessed identity.

## Seam 4: resource, dataset and scope authority

Resources are the authority for source bytes and lineage. Datasets are
structured views over an authorized source or a bounded derived result. The
dataset adapter must resolve every structured-data input through one shared
scope helper, including:

```text
dataset.inspect / schema / sample
dataset.select and dataset.list
data.query / aggregate / statistics
data.join.left and data.join.right
```

The scope helper accepts only descriptors whose source is readable through the
active resource authority, a current-turn derived dataset, or an explicitly
allowed repository-backed source under the configured repository root. A
descriptor shown by listing must therefore be usable by the next authorized
operation.

Materialization is bounded and host-owned:

```text
source resource or provider response
    -> bounded classification
    -> resource registration
    -> optional dataset projection
    -> dataset_refs + provenance
```

An eligible shallow, reasonably regular JSON array may become a temporary
dataset. A nested or heterogeneous response remains a bounded JSON resource.
The dataset's `source_resource_uri` points to the saved response. Derived
results carry `dataset://` URIs, turn/session scope and lineage. Rows or pages
are not fetched implicitly beyond configured byte, row and page limits.

`dataset_refs` is a result contract and can be carried by native, MCP and
OpenAPI results. It is not an instruction to the model and does not make an
external URI local. A later `data.*` step must still pass the same source and
scope authority checks.

## Seam 5: provider execution and materialization

Native, MCP and OpenAPI providers converge at `agent_tool_view`. They expose
the same high-level responsibilities:

```text
resolve_tools(context)
    -> filtered tool view
validate(agent_tool_call)
    -> canonical arguments or bounded error
call(agent_tool_call)
    -> normalized agent_tool_result
```

The provider-specific differences belong below that seam:

| Provider | Provider owns | Host still owns |
| --- | --- | --- |
| Native | Adapter implementation and local store/backend call | Scope, profile, policy, bounds and result normalization |
| MCP | Transport, remote definition and remote call | Exposed name, argument normalization, policy, deadline and result bounds |
| OpenAPI | Operation catalog, path/parameter mapping and bounded HTTP request | Spec/base URL policy, auth reference, network policy, access class, redirects and materialization |

The OpenAPI HTTP executor returns the bounded provider result. An optional
host-owned result materializer may save the response as a resource and derive
`dataset_refs`, but it must not start another model generation, follow an
unvalidated pagination/export URL or hide a second tool call. OpenAPI Links
are planning hints; they are not automatic execution chains.

For collection/item APIs, a row-level candidate is only a selection handle. The
host retains the provider, collection operation, item operation, parameter name
and actual item identifier, then maps the selection back to a validated API
operation. The model never turns a row value into a free-form URL.

## Seam 6: family routing, planning and review

Family selection is a preflight over the host-approved tool view:

```text
request
  -> compact family selection
      NO_TOOLS -> ordinary conversation
      TOOLS: ... -> selected tools -> plan or narrow singleton fast path
```

Selecting a family means that the request needs external evidence. The host
sets the existing `require_tool_execution` contract, so an answer-only model
response cannot silently replace the required tool work.

The singleton fast path is limited to one authorized argument-free tool. It
uses the ordinary chat/tool driver with required tool choice and then the
normal tool follow-up. Multiple tools, dataflow, required semantic arguments,
resource workflows and active plans remain in ordinary planning. The fast path
does not bypass validation, scope, policy, cancellation, deadlines, result
bounds or tool-round limits. It also does not create a hidden plan, retry or
completion observation.

Reflection is a bounded quality pass after execution, not a second discovery
orchestration. Its context is compact and bounded. Once required tool work has
closed, reflection may revise the answer but may not reopen completed tool work.
Deliberate planning owns multi-step ordering and bounded repair. Research owns
its acquisition controller and evidence workspace; it does not create a
parallel tool scheduler.

## Seam 7: protocol and event projection

JSONL is the transport-neutral command/event contract. TCP, Unix sockets, HTTP
and SSE are adapters around it:

```text
runtime event
    -> common event contract
    -> JSONL/daemon serialization
    -> SSE or UI projection
```

The web client may format, group or collapse events, but it must not reinterpret
runtime success, tool completion, scope or policy. Diagnostics belong in the
bounded event/detail fields and logs; raw resource contents and credentials do
not belong in events.

This seam is particularly important for asynchronous operations. A pending
operation has one owner for cancellation, deadline and completion. A UI retry
or a protocol reconnect must not create a second hidden runtime call.

## Verification at the seams

Tests should be named after the boundary they protect:

| Seam | Minimum verification |
| --- | --- |
| Config/catalog | Parse, validation, profile filtering and executable tool view |
| Host/request | Scope binding plus resources and dataset inventory in the first planner request |
| Compact plan | `{tool,args}` normalization, binding order and bounded repair |
| Dataset authority | `dataset://`, `source_resource_uri`, provenance, TTL/turn scope and bounds |
| Providers | Native/MCP/OpenAPI result normalization and policy enforcement |
| Orchestration | Family preflight, required execution, fast-path boundaries and reflection closure |
| Protocol | JSONL/SSE event mapping without semantic changes |

Model-free tests must exercise the runtime unconditionally. Do not put the
runtime call or the only meaningful assertion inside `assert`, because release
builds define `NDEBUG`. Use explicit failure checks for setup, side effects,
request contents and observed calls; assertions may remain as additional debug
checks.

The Qwen+Nomic smoke is complementary rather than a replacement for these
contracts. It checks that a compact model can follow the visible inventory and
produce a useful plan, while deterministic tests prove that a malformed or
ambiguous proposal cannot escape host validation.

## Change checklist

When changing one of these components, verify the adjacent seams in this
order:

1. Update the common contract or host-owned binding first.
2. Update the model-facing compact projection and state what is deliberately
   not exposed.
3. Update the native/MCP/OpenAPI adapter without adding a parallel executor.
4. Add a deterministic contract test that fails in release builds as well as
   debug builds.
5. Add or update the relevant model-free smoke, then run Qwen/Nomic if the
   change affects compact planning or context size.
6. Update [Agent Bugs](bugs.md) when a seam failure exposed a new invariant.

The final documentation should name the owner of every identity, permission,
retry, pagination decision and materialization step. If ownership cannot be
named, the design is probably introducing an accidental second orchestration
path.
