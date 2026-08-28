# OpenAPI tool provider

This document describes the host-owned OpenAPI provider for `llama-agent`.
The provider turns approved OpenAPI operations into ordinary agent tools. It
does not add a second planner or a second tool runtime.

## Position in the runtime

OpenAPI is a provider beside native tools and MCP:

```text
native provider ─┐
MCP provider ────┼─> composite agent_tool_view -> agent runtime
OpenAPI provider ┘
```

The host calls an OpenAPI endpoint directly. If inbound MCP is enabled, the
same resolved `agent_tool_view` may be projected through the MCP server. The
host must not create an internal MCP server and route its own OpenAPI calls
through that transport.

## Configuration

Host configuration is JSON. OpenAPI entries use the existing
`tools.providers` array and are selected by `"type": "openapi"`. Existing
`"type": "mcp"` entries remain valid.

The initial contract is:

See also the copyable provider fragment
[`openapi-tool-provider.json`](../examples/openapi-tool-provider.json).
Its standalone OpenAPI contract is
[`openapi-sales-contract.json`](../examples/openapi-sales-contract.json).

```json
{
  "tools": {
    "providers": [
      {
        "type": "openapi",
        "id": "sales-api",
        "enabled": true,
        "required": false,
        "spec_path": "configs/sales.openapi.json",
        "base_url": "https://api.example.test",
        "prefix": "sales",
        "allow_private_network": false,
        "policy": {
          "access": "read_only",
          "exposure": "auto",
          "operations": {
            "searchSales": {
              "access": "read"
            }
          }
        },
        "auth": {
          "type": "bearer",
          "token_env": "SALES_API_TOKEN"
        },
        "limits": {
          "connect_timeout_ms": 5000,
          "request_timeout_ms": 30000,
          "max_result_bytes": 1048576
        }
      }
    ]
  }
}
```

`spec_path` and `base_url` are host-owned. The model never chooses an
OpenAPI document, endpoint, HTTP method, credential, or arbitrary header.
Secrets are referenced by environment-variable name and are not written into
the serialized configuration or model-facing tool description.

The configuration layer validates the provider shape and preserves it through
JSON roundtrip. The current branch also builds a filtered OpenAPI 3 catalog,
exposes it through the normal `agent_tool_view`, and provides a bounded
host-owned HTTP executor. A missing optional spec is skipped; a missing or
invalid required spec fails provider resolution.

Provider entries may be kept in separate files with `tools.include_dir`; see
[Agent configuration fragments](agent-config-fragments.md). Each OpenAPI
fragment is one complete provider definition. The fragment directory is a
deployment/configuration concern and does not change the model-facing tool
contract.

## Access and exposure policy

The global `policy.access` is an upper bound:

| Access | Meaning |
| --- | --- |
| `read_only` | Read operations only; default and recommended starting point |
| `read_write` | Read and non-destructive writes, subject to confirmation policy |
| `full` | Read, writes and destructive operations, subject to host policy |

Default method classification is conservative:

```text
GET / HEAD  -> read
PUT / PATCH -> write
DELETE     -> destructive
POST       -> write unless explicitly classified as `read`
```

An operation override may restrict or classify an operation, but may not
escalate a `read_only` provider. In particular, `POST /search` can be marked
`read`, while an unsafe `GET /trigger` must not become safe merely because of
its HTTP method.

`exposure` has these meanings:

* `auto` exposes operations admitted by classification and global policy.
* `include` exposes only explicitly selected operation IDs, still bounded by
  global policy.
* `exclude` removes explicitly selected operation IDs from the otherwise
  eligible set.

Unsupported or ambiguous schemas must not be exposed as partially understood
tools. They are omitted or make a required provider fail validation, with a
host diagnostic explaining why.

## Model-facing operation contract

The provider creates a stable tool name from the provider identity and
OpenAPI `operationId`, for example `sales.searchSales`. The current schema
projection supports OpenAPI parameters and an `application/json` request body.
The model supplies one flat JSON object whose keys are the OpenAPI parameter
and body names, for example:

```json
{
  "customer_id": "c-42",
  "limit": 20,
  "body": {}
}
```

An OpenAPI parameter is included in the compact contract as `may be inferred`
only when the contract explicitly opts in with the vendor extension
`x-agent-inferable: true` on that parameter. This keeps the shared autowire
meaning safe: ordinary path and query parameters remain explicit, while a
host-approved parameter can participate in the existing bounded binding path.

The host maps these fields to the fixed operation in the OpenAPI catalog.
The model cannot replace the path, method, base URL, authentication or
headers. The advertised schema is the bounded subset understood by the
agent's existing JSON-schema validator.

Before a deliberate plan starts, the resolved tool view validates dynamic
collection-to-item bindings. A reference such as `$previous.id` for an item
operation is accepted only when the matching collection/search operation
precedes it in the plan. An explicitly supplied literal ID remains valid for a
direct item lookup. This check is host-owned and is also applied through a
composite provider view.

### Collection and item relations

The catalog recognizes a conservative REST-shaped relation when a read
operation on a collection path has a matching read `GET` operation on the same
path with one templated segment, for example:

```text
GET /sales             -> sales.listSales
GET /sales/{sale_id}   -> sales.getSale
```

The catalog records this as a host-side relation with `sale_id` as the item
parameter. It is a planning hint, not an instruction to issue a second HTTP
request automatically. Deliberate planning may use the relation to create the
bounded sequence `list/search -> choose or bind ID -> get`. The host remains
responsible for ordering, candidate selection and argument binding.

The relation is intentionally not inferred from path similarity alone when the
item operation is not a read `GET`, or when the collection operation is not
admitted as read-only. Ambiguous APIs can add an explicit provider-level
relation contract later; OpenAPI Links remain hints and never trigger hidden
follow-up calls.

### Deliberate and research boundaries

Collection-to-item relations belong to the deliberate plan because they are
ordered dataflow: a collection/search result must first produce a candidate
identifier before an item operation can consume it. The host validates that
ordering before execution and keeps the binding explicit in the plan.

Research has a different contract. Its runner uses explicitly selected research
tools and records bounded evidence; it must not silently turn an OpenAPI
relation into extra API calls. A research task may carry dependencies, and the
runner will not execute it until every dependency is completed. OpenAPI
collection/item execution in research therefore requires a future explicit
research-task adapter or plan handoff; the current provider exposes the
relation to deliberate planning only.

Results are bounded and should contain status, content type and a bounded
response body. Credentials, unrestricted response headers and unbounded
binary payloads must not enter model context.

## Security and lifecycle requirements

The provider must enforce all of the following:

* local specification loading by default; the current implementation accepts
  a host-provided local spec path;
* an explicit host `base_url`, with no use of arbitrary `servers` values from
  the document without policy validation;
* HTTPS by default, with narrowly scoped localhost HTTP only for development;
* private-network and redirect/SSRF checks; private/local targets require the
  explicit host opt-in `allow_private_network: true`, the validated DNS
  address is pinned for the request, and redirects are never followed
  automatically;
* bounded request, response and timeout limits;
* host-owned credentials and a safe header allowlist;
* caller policy intersected with provider policy for inbound MCP;
* writes and destructive operations hidden or confirmation-gated by default;
* `required: true` affecting startup/readiness, while optional provider
  failures leave the rest of the host usable.

OpenAPI Links are dataflow hints. They may be returned through
`describe_tool_dataflow()` for planning, but they must not silently trigger
follow-up HTTP calls or create automatic operation chains.

## API result representations

An OpenAPI result is first an ephemeral, host-owned API result. The host may
derive a second representation when the bounded response is useful for later
work:

```text
tabular JSON collection       -> temporary dataset view
CSV/Excel or other file body  -> resource -> existing processor/importer
JSON object or heterogeneous   -> bounded JSON resource
```

The conversion is host-side and is not an additional model-visible tool call.
A collection is eligible for a dataset view only when it is a bounded array of
reasonably stable objects. The original response remains the source material;
the dataset is an analysis projection and may be discarded with the turn or
session. A single object is normally a record/resource, not a one-row dataset.

Every derived view must preserve provenance back to the provider, operation,
canonical request parameters, retrieval time and content hash. Dataset lineage
also records the source representation and operation. Sensitive response
content may be allowed by an explicitly local deployment policy, but it must
not be copied into credentials, provenance fields or unbounded event/log
payloads.

Collection pagination is explicit. Only the configured first-page/response
limits may be materialized in one step; `next`, `cursor` and export links are
host-validated continuations and require a later plan step.

The provider exposes this boundary through an optional host-owned result
materializer callback. The HTTP executor only returns the bounded response;
the callback may classify it, register a turn/session-scoped resource or
dataset using the existing stores, and attach the resulting references to the
tool result. The callback is deliberately not part of the model contract: it
must not issue another model generation, follow an unvalidated URL, or turn
pagination into hidden background work. When no callback is installed, the
provider returns the bounded JSON inline and does not persist a dataset.

Collection rows that have a scalar item identifier may also receive opaque
host-owned candidates such as `getSale#1`. The candidate is only a selection
handle; the host retains the provider, collection operation, item operation,
parameter name, row position and actual identifier. Selecting it therefore
binds `/sales/{id}` through the catalog rather than allowing the model to
invent a URL or path. Rows without a valid scalar identifier are not
selectable, and duplicate identifiers remain distinguishable by row position.

## Verification targets

The provider work should add tests in this order:

1. JSON parse, validation and roundtrip for mixed MCP/OpenAPI configuration.
2. Method classification, access upper bounds and `auto`/`include`/`exclude`.
3. OpenAPI schema projection and path mapping.
4. A local fake HTTP server covering read calls, network capability denial and
   result limits. Auth, redirects and write-specific cases remain follow-up
   coverage.
5. Composite native/MCP/OpenAPI resolution and inbound MCP projection.
6. Reload/readiness behavior and required-versus-optional provider failures.

The configuration contract is covered by
`llama-agent-daemon-mcp-config-ctest`. Catalog, provider and local HTTP
behavior are covered by `llama-agent-openapi-catalog-ctest`,
`llama-agent-openapi-provider-ctest` and `llama-agent-openapi-http-ctest`.
