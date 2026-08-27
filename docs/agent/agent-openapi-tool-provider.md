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
The model supplies explicit sections such as:

```json
{
  "path": { "customer_id": "c-42" },
  "query": { "limit": 20 },
  "body": {}
}
```

The host maps these fields to the fixed operation in the OpenAPI catalog.
The model cannot replace the path, method, base URL, authentication or
headers. The advertised schema is the bounded subset understood by the
agent's existing JSON-schema validator.

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
