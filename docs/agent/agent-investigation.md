# Investigation Workspace

Investigation is a host-owned, read-only search and evidence workflow. It uses
configured Elasticsearch or OpenSearch sources, virtual datasets, Cozo case
revision storage, immutable resources for collected evidence, and the packaged
Vue/Cytoscape MCP App. It is disabled by default and requires a Cozo-enabled
build and a Cozo data store. Configuration changes take effect after daemon
restart.

## Configure a source

Add an `investigation` section to the daemon JSON configuration. Provider IDs,
index allowlists, namespace/project scope, field mappings, network policy and
authentication stay host-owned. The model can request supported typed filters
over the declared fields; it cannot submit arbitrary search DSL, scripts or
index administration requests.

```json
{
  "stores": {
    "data": { "backend": "cozo", "path": "var/investigation.cozo" }
  },
  "investigation": {
    "enabled": true,
    "max_events_per_query": 2500,
    "sources": [
      {
        "id": "security-logs",
        "dialect": "elasticsearch",
        "base_url": "https://elastic.example:9200",
        "allowed_indices": ["logs-security-2026.10"],
        "allowed_namespaces": ["tenant-a"],
        "fields": {
          "event.id": "event.id",
          "event.type": "event.category",
          "@timestamp": "@timestamp",
          "user.id": "user.id",
          "host.name": "host.name"
        },
        "auth": { "type": "bearer", "token_env": "ELASTIC_TOKEN" },
        "max_page_size": 200
      }
    ]
  }
}
```

Use `"dialect": "opensearch"` for OpenSearch. Index names must be explicit
and wildcard-free in the current implementation; list each permitted index or
data stream. The example above uses a concrete index name. Private/local endpoints require the
source's explicit private-network setting and the runtime network capability.
Never put credentials directly in the JSON file; use the existing host auth
references and environment-backed secrets.

## Use a case

The investigation tool family supports case creation, scoped case listing and
opening, bounded collection from a virtual dataset, graph expansion, event
inspection, deterministic graph/path analyses, hypothesis recording and
freezing a report from an exact case revision. The MCP App provides the same
case list, graph, time filter, event evidence view and report action. Case list
cursors are ordered by case ID and are restricted to the active namespace plus
project, or namespace plus session when there is no project.

Collected source pages and normalized event references are retained as
immutable resources. Cozo keeps the current case revision and revision history;
the case-list index is rebuilt from published revisions when an older Cozo
store is opened. A partial, timed-out or truncated page remains marked as
partial. A relation records the deterministic rule and source event IDs that
support it; it does not establish cause. The model may record hypotheses but
cannot mark them as reviewed conclusions.

This version performs bounded historical search. It does not yet provide
collector/Kafka subscriptions, PIT/search-after cursors, a live event stream,
formal evidence chain-of-custody, or model-supported smoke coverage. Case
listing and evidence reads are scope-checked on every request.

## Verification

The deterministic CTests cover configuration, connector bounds, Cozo case
revisions and list-index persistence, analysis contracts, MCP resource
registration and tool adapters. The connector test can also run against a real
local service populated with the synthetic `llama-agent-investigation-smoke`
index:

```bash
LLAMA_AGENT_INVESTIGATION_REAL_ENDPOINT=http://127.0.0.1:19200 \
  LLAMA_AGENT_INVESTIGATION_REAL_DIALECT=elasticsearch \
  ctest --test-dir build-agent-vulkan-cozo -R '^test-agent-investigation-http$' --output-on-failure

LLAMA_AGENT_INVESTIGATION_REAL_ENDPOINT=http://127.0.0.1:19201 \
  LLAMA_AGENT_INVESTIGATION_REAL_DIALECT=opensearch \
  ctest --test-dir build-agent-vulkan-cozo -R '^test-agent-investigation-http$' --output-on-failure
```

Without `LLAMA_AGENT_INVESTIGATION_REAL_ENDPOINT`, this CTest remains an
offline contract test and does not contact a search service.
