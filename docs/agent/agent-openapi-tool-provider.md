# OpenAPI tool provider

This document describes the host-owned OpenAPI provider for `llama-agent`.
The provider turns approved OpenAPI operations into ordinary agent tools. It
does not add a second planner or a second tool runtime.

The ownership and result-flow rules shared with native and MCP providers are
documented in [Agent runtime seams and ownership](agent-runtime-seams.md).
This document focuses on OpenAPI-specific configuration, operation
classification, path binding and response materialization.

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

## Session-scoped dynamic admission

The `analysis` and `research` profiles also expose `openapi.connect`. The
model supplies only `spec_resource`, a resource handle returned by
`web.fetch`; it cannot supply a server URL, credentials, HTTP method, or
operation name to the admission step. `web.fetch` now materializes its bounded
result whenever the resource store is available, including small OpenAPI
documents, and includes the stable handle in its result.

The host reads the resource, parses the fetched text as OpenAPI JSON, and
reuses the normal catalog builder. Admission requires an explicit HTTPS
server and registers only anonymous read-only operations. Private-network
access remains disabled, and the existing HTTP executor checks the server
before each operation. Unsupported or external `$ref` values are rejected by
the catalog builder. No operation is contacted during admission.

Dynamic providers are held in an in-memory registry owned by the runtime
session. They are resolved into the ordinary OpenAPI provider/tool view on the
next turn, so route compilation sees their capabilities at the normal host
boundary. They are removed when the session is reset or closed and are not
persisted across process restarts. Results from dynamic operations are
materialized as bounded host resources with provider/operation provenance.

The model-facing family index describes `web` as able to fetch public content,
including API specifications, and `openapi` as loading a fetched spec and
exposing its permitted operations. The starter package also includes an
OpenAPI-onboarding blueprint and workflow. Its optional `openapi.read`
capability resolves to the admitted provider's anonymous read-only operations,
so the route can resume with those tools on the next turn.

## Fetch and connector smokes

The offline runtime smoke verifies that a stubbed `web.fetch` result is
materialized and can be read through `resource.read`. The OpenAPI host smoke
feeds a fetched-result-shaped resource into `openapi.connect`, resolves the
provider on the next turn, and invokes a mocked operation. Both are part of the
normal agent CTest set.

The opt-in live smoke exercises the two network paths independently:

```sh
cmake --build build-agent-vulkan-cozo --target llama-agent-web-openapi-live-smoke --parallel 3
./build-agent-vulkan-cozo/bin/llama-agent-web-openapi-live-smoke --mode resource
./build-agent-vulkan-cozo/bin/llama-agent-web-openapi-live-smoke --mode openapi
```

`resource` fetches and reads a public page. `openapi` fetches the checked-in
OpenAlex contract from the public repository, admits it, then performs a
bounded live OpenAlex request through the newly exposed operation. These live
smokes need network access and are not registered as default CTests.

## Eurostat example

Eurostat's public Statistics API is documented with WADL rather than OpenAPI.
The repository therefore includes a deliberately small OpenAPI adapter and
configuration in `docs/examples/eurostat-statistics-openapi.json` and
`docs/examples/agent-host-config-eurostat.json`. The adapter describes the
stable `/data/{datasetCode}` operation and leaves dataset codes and filters as
normal tool arguments. It is an adapter for the generic provider, not a claim
that the upstream WADL is an OpenAPI document.

The live integration smoke is opt-in because it requires external network
access:

```sh
./build-agent-packaging/bin/llama-agent-openapi-eurostat-live-smoke
```

It requests `DEMO_R_D3DENS` with `geo=SE` and `time=2020` and verifies the
JSON-stat 2.0 dataset response. The live smoke is not registered as a default
CTest so ordinary CI remains deterministic.

## OpenAlex example

OpenAlex has two relevant addresses: the public API base URL is
`https://api.openalex.org`, while the machine-readable OpenAPI 3.1 document is
published at `https://help.openalex.org/openapi.json`. The provider bootstrap
helper can now keep those addresses separate:

```sh
./scripts/agent-provider-bootstrap.sh \
  --type openapi \
  --id openalex \
  --prefix openalex \
  --base-url https://api.openalex.org \
  --spec-url https://help.openalex.org/openapi.json \
  --spec-output openalex-openapi.json \
  --output openalex-provider.d/openalex.json \
  --required \
  --anonymous-public-reads \
  --allowed-operation listWorks \
  --allowed-operation getWork \
  --allowed-operation listAuthors \
  --allowed-operation getAuthor \
  --allowed-operation listInstitutions \
  --allowed-operation getInstitution \
  --allowed-operation listTopics \
  --allowed-operation getTopic \
  --default-projection listWorks=id,display_name,publication_year,doi,cited_by_count,primary_topic,topics \
  --default-projection getWork=id,display_name,publication_year,publication_date,doi,authorships,primary_topic,topics \
  --default-projection listAuthors=id,display_name,works_count,cited_by_count,orcid,last_known_institutions \
  --default-projection getAuthor=id,display_name,works_count,cited_by_count,orcid,last_known_institutions \
  --default-projection listInstitutions=id,display_name,country_code,works_count,ror \
  --default-projection getInstitution=id,display_name,country_code,works_count,ror \
  --default-projection listTopics=id,display_name,works_count,subfield,field,domain \
  --default-projection getTopic=id,display_name,works_count,subfield,field,domain \
  --default-page-size 5 \
  --max-page-size 10 \
  --parameter-description 'listWorks.filter=Filter works with field:value syntax. Combine filters with commas for AND or pipes for OR. Examples: publication_year:2024, author.id:A..., institutions.id:I..., topics.id:T... (any assigned topic), primary_topic.id:T... (top-ranked topic).' \
  --parameter-description 'listWorks.search=Full-text search across work titles, abstracts and other text fields. Use display_name to return the work title.' \
  --parameter-description 'listAuthors.search=Search author records by name and other indexed author text.' \
  --parameter-description 'listInstitutions.search=Search institution names and other indexed institution text.' \
  --parameter-description 'listTopics.search=Search research-topic names and other indexed topic text.'
```

In an installed POSIX agent package the same helper is available as
`llama-agent-provider-bootstrap`. It is a configuration-authoring utility,
not part of the daemon's request path: it uses only the Python standard
library and needs Python 3 plus the system certificate store only when it
downloads an HTTPS specification. The native OpenAPI provider does not invoke
Python after the provider configuration has been written.

The generated spec and provider fragment are checked in as
[`openalex-openapi.json`](../examples/openalex-openapi.json) and
[`openalex-provider.d/openalex.json`](../examples/openalex-provider.d/openalex.json).
The host example loads the fragment through `tools.include_dir`. It uses
`exposure: include` and allows only eight read operations: list/get works,
authors, institutions and topics. The upstream document defines many more
operations; they remain hidden from the model.

OpenAlex assigns each work up to three topics and exposes a `primary_topic`.
Resolve a human topic name through `openalex.listTopics`, then filter works by
the returned `topics.id:T…`. This matches works carrying that assigned topic;
`primary_topic.id:T…` is narrower and matches the top-ranked topic. The
`listWorks.search` field provides text discovery over titles, abstracts and
other text fields, while `display_name` returns the work title. The operation
filter description adds examples for author, institution, topic and year
filters. OpenAPI parameter descriptions are retained in each model-facing
input schema, so those search/filter semantics appear beside the corresponding
fields rather than only in host documentation. The upstream OpenAPI document
declares global API-key security and a required `api_key` parameter; the
generator's explicit
`--anonymous-public-reads` option clears both for selected safe operations.

This host config is a static-provider example: startup resolves the generated
local spec and exposes only the allowlisted read operations. It is suited to
recurring scholarly work and does not require a turn to call `web.fetch` or
`openapi.connect`. The starter package includes work search, work lookup,
bounded literature review, and entity-filtered search by author, institution,
topic and year.

The model-free live client smoke is opt-in:

```sh
cmake --build build-agent-packaging \
  --target llama-agent-openapi-openalex-live-smoke --parallel 3
./build-agent-packaging/bin/llama-agent-openapi-openalex-live-smoke \
  --spec docs/examples/openalex-openapi.json
```

It validates all eight allowlisted operations, performs bounded title/text,
author, institution and topic searches, then applies returned entity IDs and
`publication_year` filters to works. OpenAlex list responses stay bounded
`meta`/`results` JSON resources.

### Paging is contract-derived

The host classifies collection operations from the parameters declared by the
OpenAPI contract. It recognizes bounded page sizes, page numbers, offsets and
cursors, and records the exact parameter names together with an optional
projection parameter. For example, the generated OpenAlex `listWorks`
operation is classified as a cursor-paged collection with `per_page`, `cursor`
and `select`.

That metadata is projected into the generated model-facing tool description:
the model is told to bound each request, use a continuation cursor only when
the response supplies one, and request only the fields needed for the task.
The host still owns the endpoint, arguments validation, result-size limit and
execution. Classification does not invent a cursor, silently issue follow-up
requests, or turn an oversized result into a successful answer. Automatic
pagination remains a future provider/runtime operation. A host may additionally
set `limits.default_page_size` and `limits.max_page_size` for a provider. When
both are positive, the HTTP executor supplies the declared page-size parameter
when the model omits it and clamps an oversized supplied value. This is a
single-request safety bound, not automatic pagination or cursor following.
For APIs with large item schemas, an operation policy may additionally set
`default_projection`; the host supplies it only when the operation declares a
projection parameter and the model did not provide one. This keeps projection
defaults operation-specific and avoids inventing fields for unrelated APIs.
An operation policy may also set `required_parameters` for parameters that are
semantically required by the host-approved use of an otherwise broader API.
The provider keeps the canonical OpenAPI schema unchanged for execution, but
adds these already-declared properties to the optional model-facing schema
projection used by the compact tool contract. Ordinary planning therefore
sees `query`/`search` and similar host-required values as required before the
first call. If a configured name is absent from the canonical properties,
provider resolution fails rather than inventing an untyped model field.
The host still checks the requirements immediately before HTTP execution and,
if one is missing, returns a retryable required-parameter result with repair
context so the existing reflection path can revise the tool step. The host
does not invent a search term or silently execute an unscoped collection
request.

For a workflow that already has concrete host-owned values, an operation policy
may additionally set `bound_arguments` as a JSON object. These values are
copied into the generic `common_agent_tool_argument_binding` request seam
before planning and are merged again before execution. The planner may omit
the fields, but a conflicting model value is rejected. This is a structured
workflow binding, not inference from the user's prose:

```json
{
  "listWorks": {
    "required_parameters": ["search"],
    "bound_arguments": {
      "search": "machine learning",
      "per_page": 1,
      "select": "id,display_name"
    }
  }
}
```

Only callers that actually know the concrete values should set
`bound_arguments`; otherwise the existing model-facing required-parameter and
tool-repair path remains authoritative.

The optional model-backed smoke uses the same host/provider path and forces
the model to use `openalex.listWorks` rather than the generic web tools:

```sh
LLAMA_AGENT_BUILD_DIR=build-agent-packaging \
LLAMA_AGENT_MODEL=/path/to/model.gguf \
./scripts/test-agent-openalex-model-smoke.sh
```

The script accepts `LLAMA_AGENT_EMBEDDING_MODEL` when a separate embedding
model is desired, but OpenAlex itself does not require an embedding model.
Model-backed execution is deliberately not a default CTest because it needs a
real model and external network access.

The model smoke only succeeds when the runtime records an actual tool
execution (`stage=tool ... tool=openalex.listWorks`). A plan mention or a
textual answer is not enough. This makes small-model planner/argument
contract failures visible instead of allowing a false-positive smoke result.

For a profile containing one host-bound operation, the runtime may first ask
the model for one native operation selection. That selection is not a second
tool runtime and does not grant authority: the selected name must be the
already filtered registered operation. The host then materializes an ordinary
plan step, merges the configured bindings, validates the model-facing and
provider contracts, and continues through the same execution and reflection
seams. A native selection that is unavailable or malformed falls back to the
bounded JSON planner projection; it never bypasses planner validation.
Native selection arguments do not have authority over configured OpenAPI
bindings: bound fields are discarded before host materialization. Ordinary
planner output that conflicts with a binding still enters the existing repair
and conflict-rejection path.

The singleton projection is derived from the effective tool view. Its family
grammar permits exactly one family, and its plan grammar permits exactly one
executable step. This is an output-size reduction for small models, not a
change to required tool execution, host authority, reflection, research, or
final-answer policy.

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
              "access": "read",
              "required_parameters": ["query"]
            }
          }
        },
        "auth": {
          "type": "bearer",
          "scheme": "bearerAuth",
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

The auth object deliberately stays flat and provider-scoped. `type` selects the
host implementation, `token_env` names the environment variable containing the
credential, and optional `scheme` selects an OpenAPI `securitySchemes` entry.
The scheme supplies placement details such as the header name; the provider
configuration does not repeat them. Future methods add their credential
references as sibling fields rather than introducing a nested credential graph.

The first extended method is `oauth2_client_credentials`:

```json
"auth": {
  "type": "oauth2_client_credentials",
  "scheme": "salesOAuth",
  "client_id_env": "SALES_CLIENT_ID",
  "client_secret_env": "SALES_CLIENT_SECRET",
  "scopes": ["sales.read"]
}
```

The token endpoint is taken from the selected OpenAPI OAuth
`clientCredentials` flow unless `token_url` is explicitly supplied by the
host. The host caches the bounded token in memory, never exposes it to the
model or event stream, rejects redirects, and applies the same HTTPS/private
network policy as the API request.

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
projection supports OpenAPI parameters, an `application/json` request body and
the local component references commonly used by generated specifications.
The host resolves bounded local references for `parameters`, `schemas`,
`requestBodies`, `responses` and `headers`, including nested schema references.
Path-level parameters are inherited by operations. External references and
unsupported component sections are rejected; the host never downloads another
document while building a catalog.

For example, both of these forms are supported:

```json
"requestBody": { "$ref": "#/components/requestBodies/CreateVehicle" }
```

and:

```json
"responses": {
  "200": { "$ref": "#/components/responses/VehicleResponse" }
}
```

The model still receives one bounded JSON argument object. A resolved request
body is exposed under its `body` property, and a resolved response schema is
used to describe the result; the raw HTTP response is not changed. Recursive
references remain bounded and are not expanded indefinitely.

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

OpenAPI names use a dot between the configured provider prefix and the
operation ID: `sales.listSales`, `sales.getSale`. This is the model-facing
family-qualified name and is the name that must appear in a deliberate plan.
The ordinary MCP provider keeps its existing underscore convention by default,
for example `github_search_issues`; the two conventions are intentional and
are not interchangeable aliases. A provider may still supply a prefix that
already ends in `.` or `_`, in which case the host does not add a second
separator.

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
reasonably stable, shallow objects with scalar values. The host stores the raw
bounded response as a turn-scoped resource and returns a `dataset_refs` entry
whose `source_resource_uri` points to that exact resource. The dataset is an
analysis projection and may be discarded with the turn or session. A single
object is normally a record/resource, not a one-row dataset.

The dataset reference is host-generated and contains its URI, display name,
row/column counts, source representation and provenance. It is attached to
the tool result, plan observation, working state and continuation checkpoint;
it is never accepted as a model-supplied dataset identity. If the response is
nested, heterogeneous, oversized or otherwise outside the projection limits,
the host keeps it as a bounded JSON resource instead.

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
pagination into hidden background work. The standard CLI host materializer
creates datasets only when both stores are available; otherwise it keeps the
bounded response inline or as a resource according to the same limits.

The same `dataset_refs` result field is preserved when a native or MCP-backed
tool returns a dataset reference. Native and MCP provider adapters first check
the reference shape; when a `data.*` operation uses a host-owned `dataset://`
URI, the host additionally resolves its descriptor and source resource under
the current authority. An external MCP reference is not
automatically treated as a local dataset merely because its JSON shape is
valid.

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
The OpenAlex-specific parameter-reference regression is part of the catalog
contract test; the two OpenAlex live smokes above remain opt-in.
