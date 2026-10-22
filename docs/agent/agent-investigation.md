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
    "workflow_package": "docs/examples/agent-investigation-workflows-v1.json",
    "max_events_per_query": 2500,
    "sources": [
      {
        "id": "security-logs",
        "dialect": "elasticsearch",
        "base_url": "https://elastic.example:9200",
        "allowed_indices": ["logs-security-2026.10"],
        "allowed_namespaces": ["tenant-a"],
        "fields": {
          "event_id": "event.id",
          "event_type": "event.category",
          "event_time": "@timestamp",
          "user_id": "user.id",
          "host_name": "host.name"
        },
        "relation_profiles": [
          {
            "id": "auth-observed-v1",
            "event_type": "authentication",
            "from_kind": "user",
            "to_kind": "host",
            "relation_kind": "authenticated_to",
            "from_role": "user",
            "to_role": "host"
          }
        ],
        "auth": { "type": "bearer", "token_env": "ELASTIC_TOKEN" },
        "max_page_size": 200
      }
    ]
  }
}
```

When enabled, `workflow_package` is imported through the existing bootstrap
and plan-store path. The checked-in package declares identity, communication,
event-chain, integration-trace, hypothesis-testing and report-composition
workflows, plus actor/relationship, narrative/comparison and document-provenance
workflows; it does not add a second workflow engine. Relative paths resolve
against the daemon config file's directory, and the file is checked while the
config loads. Use an absolute path when the deployment layout is managed
outside the config directory. The package's procedures use the existing
bootstrap embedding path, so configure `model.embedding_model` when the
package contains procedures.

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
freezing a report from an exact case revision. `investigation.case.open`
returns only case metadata, counts and the first graph entity. Read entity,
event, relation, claim or gap records through `investigation.case.items` pages.
`investigation.events` also accepts a page cursor. These case cursors bind the
case revision and query filter; requests reauthorize the case and reject stale
revisions. Pages are bounded by item count and serialized size. The MCP App
uses the same APIs and exposes a “Load more events” action. Case list cursors
remain ordered by case ID and are restricted to the active namespace plus
project, or namespace plus session when there is no project.

Collected source pages and normalized event references are retained as
immutable resources. Cozo keeps the current case revision and revision history;
the case-list index is rebuilt from published revisions when an older Cozo
store is opened. A partial, timed-out or truncated page remains marked as
partial. A relation records the deterministic rule and source event IDs that
support it; it does not establish cause. The model may record hypotheses but
cannot mark them as reviewed conclusions.

Configured relation profiles match the mapped event type exactly and declare
endpoint kinds, relation kind and optional endpoint roles. When a source has no
profiles, a small built-in exact event-type profile set is used. Unknown or
near-matching event types remain unlinked observations and add a case
limitation.

Declared endpoint roles must be present. Older authentication events without any
role metadata can resolve `user` and `host` only when each kind has one endpoint;
missing or ambiguous endpoints remain unlinked and add a coverage limitation.
Normalized evidence preserves the endpoint roles and host collection time.

This version performs bounded historical search. Case-page cursors paginate
already published case contents; they do not yet provide PIT/search-after
continuation against Elasticsearch or OpenSearch. It also does not provide
collector/Kafka subscriptions, a live event stream, formal evidence
chain-of-custody, or model-supported smoke coverage. Case listing and evidence
reads are scope-checked on every request.

## Mixed document and log sources

A case may contain retained documents and log files alongside configured
search-connector sources. Creating a document-only case does not require a
search source or a time window. A configured search source is marked
`not_collected` and incomplete until a collection operation records its actual
coverage. Use `investigation.source.list` to discover resources and currently
authorized search sources, `investigation.source.attach` to add a source, and
`investigation.document.prepare` to create a text representation with the
existing resource processors. These operations do not start model analysis by
themselves. The workspace asks the user to choose sources and an analysis
profile before dispatching the normal host-authorized workflow.

An authorized search connector can also be attached to an existing document-only
case. The first connector added this way must establish the case time window;
additional connectors use that same case-level window. `investigation.collect`
accepts only virtual datasets whose provider is attached to the case and keeps
each provider's event and entity IDs in that source's identity domain. Repeated
collection for one provider rebuilds that provider's relations while
preserving relations already collected from other providers.

The original upload is retained as an immutable resource with its parent
reference and content hash. PDF, DOCX and OCR processing use the processors
already configured by the host. A missing processor or lossy OCR path is
recorded as a limitation in the source revision and frozen report. A verified
quote anchor proves that the quoted UTF-8 bytes occur at the recorded offset in
the retained text representation; it does not prove that OCR faithfully
represents the source image or that the source statement is true.

The actor/relationship, narrative/comparison and document-provenance
workflows record versioned source statements and comparisons. A statement
must reference a case source and an exact, verified text anchor. Actor IDs are
references attached to those statements, not verified real-world identities;
name similarity does not merge actors. Model-produced comparisons remain
`model_proposal`, and source-dependency claims remain proposals until an
authorized analyst review or explicit deterministic rule is added. Reported
times in documents remain separate from event times observed in collected log
records. A document statement does not become an independently observed log
event by being linked to one.

`investigation.statement.record` publishes statements, comparisons and their
analysis provenance together in a new case revision. Its idempotency key binds
the full typed payload, including source versions, profile, parameters and
completeness. `investigation.case.items` exposes bounded pages of sources,
statements, comparisons, analyses and hypotheses for the selected revision.
Frozen report composition carries the source manifest, exact source and text
representation hashes, processor limitations, analysis versions and the
resulting coverage assessment. Later case revisions do not change an existing
report.

This is a bounded document-analysis path, not a general document-forensics
engine: it does not provide OCR-region coordinates, native actor resolution,
automatic source-independence verification, image/audio/video analysis,
lie detection or formal chain-of-custody. Uploaded log files can be retained
and processed as text, but they do not become normalized independent event
records unless a configured structured collection path supplies those
records.

The workspace can bind a selected case revision and selected entity, relation
and event IDs to the next chat turn. The daemon reloads the case in the active
scope, rejects stale revisions and unknown IDs, and sends only the selected
projection to the runtime. Selected event values are untrusted data and are
explicitly marked as such in the runtime prompt.

## Dataset visualizations

The analysis and investigation tool profiles also expose `visualization.create`
for a first host-integrated relationship renderer. Select a registered
`dataset://` reference, map its source and target columns, and optionally map
node labels and an edge label. The host authorizes the dataset in the current
scope, runs a bounded `data.query` with deterministic ordering, validates the
result against the shared relationship contract, and stores the frozen JSON
projection as a normal resource. The resource uses project scope when a project
is active, otherwise session scope; the tool result contains its resource
reference and the projection for the existing workspace app to render.

`visualization.open` reopens that resource in the current host scope. It checks
the resource authority and media type, reads within the visualization byte
limit, validates the stored contract and returns the frozen projection. The
workspace has an Open graph control for a saved resource URI, so reopening does
not rerun the dataset query. A graph node in a case-backed view still expands
through the existing scoped `investigation.graph.expand` operation.
The workspace persists a PNG snapshot with `visualization.snapshot`. It can
snapshot either an authorized saved visualization or the currently validated
relationship projection; client-supplied projections are explicitly marked as
presentation data, not independent evidence. The host checks the resource
scope, relationship contract, PNG signature and content hash, and stores the
image as an immutable `image/png` resource whose lineage points to the saved
visualization projection. Persisted snapshots are limited to 512 KiB to fit the existing HTTP request limit together with the graph projection; the
renderer caps image dimensions at 4096 pixels per side.

To include a snapshot in an investigation report, add its resource URI to the
workspace's snapshot attachment list before freezing the report. Report
composition reauthorizes every URI, verifies its media type, source operation,
PNG signature and resource hash, then records the selected snapshot URI,
hash, size and source visualization URI in both the Markdown report and its
provenance metadata. The report calls these explicitly selected attachments;
it does not infer that they represent the case or establish a causal relation.
Composition writes a frozen source manifest, a structured document resource
using the existing document contract, and its Markdown rendering. The
structured resource is accepted by `document.open` and `artifact.export`, so
the existing editor and Markdown/DOCX export paths can be reused. Snapshot
references retain explicit resource links and provenance. The interactive
report reading view also renders the saved PNGs as figures with links to the
snapshot and visualization source. It verifies the source manifest and each
image hash, uses the existing scoped resource download path, and discards late
loads when the report or session changes. Missing images appear as individual
errors. Static DOCX image embedding remains separate.

Frozen reports are discoverable through a separate Cozo report index keyed by
case scope, case ID, evidence revision and report content fingerprint. The
index points to the immutable document, Markdown and source-manifest resources;
it does not alter or advance the case evidence revision. `investigation.case.items`
accepts `type: "reports"` to read a bounded page for the opened revision. The
workspace loads those entries when reopening a case and opens a selected report
in the existing document editor through the normal read-only document access
check. Recomposition with the same report fingerprint resolves to the existing
indexed resource references.

The current operation is limited to 100 source rows per call and reports source
truncation, skipped identity rows, and non-exact dataset coverage. It creates a
relationship view only. Generic dataset-backed node expansion and a renderer-
independent snapshot format remain later steps.

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
offline contract test and does not contact a search service. To run the
repeatable Podman smoke against both engines, build the HTTP contract target
first, then run:

```bash
tests/smoke/agent-investigation-search-engines.sh build-agent-vulkan-cozo
```

The live smoke has been exercised against Elasticsearch 8.17.3 and OpenSearch
2.19.2. It creates a temporary mapped index with synthetic authentication and
network events, checks a typed filter, deterministic sort and bounded first
page, then removes the containers. It does not yet exercise PIT continuation,
case publication/restart or daemon-backed collection.
