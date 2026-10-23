# Investigation Workspace

Investigation is a host-owned investigation and evidence workflow with
read-only acquisition from its sources. A case can combine retained documents,
explicitly imported JSONL/CSV logs and configured Elasticsearch or OpenSearch
sources. It uses virtual datasets, Cozo case revision storage, immutable
resources for collected evidence, and the packaged Vue/Cytoscape MCP App. It is disabled by default and requires a Cozo-enabled
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
coverage. Use the host resource browser to select resources,
`investigation.source.list` to discover currently authorized search sources,
`investigation.source.attach` to add a source, and
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
name similarity does not merge actors. Model-produced comparisons and
source-dependency claims remain `model_proposal`. The
[actor association review path](#user-review-of-actor-associations) records
explicit user decisions about actor identity or relationships; it does not
establish source independence or verify the truth of a statement. Reported
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

The shared source-access check is used by both tool calls and chat-context
binding. Every attached search provider must still be authorized, and every
retained original and text representation must be accessible with its recorded
hash. Source record IDs and provider identity domains are validated separately;
adding another provider does not merge its identifiers with existing sources.
Processor outputs are retained to the case's project/session lifetime before
publication, including outputs produced by a narrower-lived processor session.
Generic resource retention also copies resources with transient expiration,
even when their organizational scope already matches the target.

Workspace reads use separate request gates for graph, events and resource
queries. A new query invalidates older responses for that view; changing case
invalidates all views. Upload completion cannot attach a file to a different
case opened in the meantime. Starting an analysis preserves the user's chat
draft.

This is a bounded document-analysis path, not a general document-forensics
engine: it does not provide OCR-region coordinates, native actor resolution,
automatic source-independence verification, image/audio/video analysis,
lie detection or formal chain-of-custody. Uploaded logs become ordinary case
events only through an explicit `investigation.log.import` operation with a
supported JSONL/CSV format and field mapping, described under
[uploaded structured logs](#uploaded-structured-logs). Free-form log text stays
retained source material unless a supported parser profile supplies events;
uploading or reading text does not itself normalize events.

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

## Incremental document structure and uploaded log events

The v2 definitions in `agent-investigation-workflows-v1.json` extend the same nine
workflow families. The filename continues to identify the package schema. The
package version and procedure, blueprint and workflow definition identities are
v2 because the bootstrap importer preserves already installed definitions. Old
v1 plans remain available; restarting imports the new definitions alongside them.
No running plan is rewritten and no semantic routing precedence is introduced.

### Uploaded structured logs

A retained `log_file` source can be normalized using `investigation.log.import`
after text preparation. The workspace exposes an explicit JSONL/CSV format and
field mapping form. Upload and preparation do not start an import or analysis.
For example, a JSONL source with nested ECS fields can use:

```json
{
  "case_id": "case-example",
  "expected_revision": 2,
  "source_id": "source-example",
  "format": "jsonl",
  "fields": {
    "event.type": "/event/type",
    "@timestamp": "/@timestamp",
    "user.id": "/user/id",
    "host.id": "/host/id"
  },
  "limit": 200,
  "idempotency_key": "import-attempt-1"
}
```

Mapping values are exact JSON keys, JSON pointers, or exact CSV headers. JSON
arrays and objects are not coerced into scalar identifiers. JSONL records and
CSV records retain UTF-8 byte offsets, byte lengths and physical start lines;
quoted CSV delimiters and embedded newlines are supported. Duplicate CSV headers
are rejected. Unknown formats require a separately implemented parser profile;
there is no heuristic format classifier, executable parser or model-generated
regular expression path.

The shared collection normalizer creates ordinary case events, entity roles and
bounded evidence-backed relations. Event identities include the source domain,
parser profile and original record position, rather than trusting a possibly
reused event ID. Original text, rejected-record positions, import parameters and
normalized pages remain available as evidence. Missing or invalid timestamps
remain unknown and are recorded as limitations.

A page scans at most 500 records, including rejected records. Input remains
limited by the existing 4 MiB source-retention boundary; records are limited to
64 KiB and JSON nesting to 64 levels. These are bounded historical imports, not
a streaming tail or arbitrary-size file reader.

The host persists the parser profile and next record separately from the original
processor version. Subsequent calls with the same profile can resume saved
progress without a browser cursor; supplied cursors must match revision, scope,
source and profile. The parser profile cannot silently change after publication.
Reusing an idempotency key with the same request replays the published response;
using it for another page or profile produces a conflict. `complete` describes
the current page, `exhausted` describes traversal, and `source_complete` also
accounts for earlier rejected records and source limitations. None establishes
complete coverage of the real-world incident.

### Actor mentions and proposed cross-source links

`investigation.statement.record` accepts optional `mentions` inside a statement:
`name`, `kind` and absolute UTF-8 `byte_offset`. The host validates the exact name
inside the verified statement quote and creates a stable mention ID and distinct
actor-candidate ID. Equal names do not merge candidates. Candidates can exist
without relations and are paginated through `investigation.case.items` types
`actors`, `actor_mentions` and `evidence_links`, also shown in the workspace.
Older free-form `actor_ids` remain readable; they are not retroactively resolved
into the new anchored candidate structure.

Optional `evidence_links` propose `possible_same_actor` between candidate IDs or
`possible_same_event` between a `reported_event` statement and a normalized log
event. The selected sources, endpoints and supporting statement IDs are checked
by the host. Actor-link evidence must include the mentions of both candidates;
event-link evidence must include the reported statement. These are proposals,
not identity merges, independent corroboration or analyst decisions. Physical candidate merging/splitting is not used. The user review path described
below groups the presentation while preserving candidates and their mentions.

Hypotheses can use `supporting_statement_ids` and `contradicting_statement_ids`
separately from their existing log-event references. Verifying a text position
never verifies that the reported event occurred. Source-dependency comparisons
remain available and should prevent dependent retellings from being described
as independent evidence; there is no numerical independence score.

### Source follow-up and frozen reports

`investigation.analyze` with `analysis: "source_followup"` and explicit
`source_ids` returns a bounded structural work list: prepare missing text, declare
or continue a log import, extract missing statements, or compare selected source
pairs without a complete published pair analysis. It does not infer identity or
contradiction from names or wording, and does not execute the suggestions.
The workspace provides a separate button for this deterministic read; users can
then start the relevant existing chat workflow with their selected sources.

Frozen report manifests preserve source parser profiles and progress, normalized
event positions, actor candidates and mentions, proposed links, hypotheses and
analysis records at the selected revision. Later documents add new revisions;
they do not update old report evidence. Actor/mention tables and proposed links
are available now; the user-reviewed actor projection described below uses the shared relationship
renderer; automated identity resolution is not added by this extension.

### Verification status for this extension

Contract coverage has been added for parser boundaries and rejection accounting,
UTF-8/CSV positions, actor anchoring and identity separation, proposed event
links, hypothesis evidence roundtrips, follow-up budgets, import pagination and
idempotency, window-free event reads, isolated actor presentation and late import
responses after navigation. The existing CTest and frontend registrations are
reused. Compilation, CTest, frontend tests, browser smokes and model smokes have
intentionally not been run for this extension; they remain required before the
new paths are described as verified functionality.


## User review of actor associations

The Investigation Workspace now offers explicit actor selection and the actions
**Same actor**, **Different actors**, **Leave identity unresolved**, **Confirm
relationship** and **Leave relationship unresolved**. Identity confirmation can
cover 2–32 selected candidates in one atomic publication; other actions use a
pair. The selected candidate order determines relation direction. Users can add
a comment describing their knowledge of the material. The `other` relation type
requires a comment.

`investigation.association.review` is an app-only operation. The native provider
uses a separate, internally authorized adapter registry only after verifying the
first-party Investigation Workspace provenance and normal write permissions.
Ordinary native/model/async calls cannot authorize a review, including a call
that guesses the tool name. Reviewer identity and timestamps are host-supplied,
not accepted as tool arguments. App-only native tools are separated from model
selection using the existing `app_tools()` contract; the same correction applies
to app-only snapshot operations.

The current authentication model records the authorized host namespace/session
as `reviewer_id`; it does not infer a person's name from a token or document.
This is session-level attribution, not an independently verified personal
identity. The web host additionally requires the app's acknowledged case and
revision before forwarding a review. Late opens or results cannot restore an old
case binding.

Decisions are immutable records with operation identity, request fingerprint,
reviewer, timestamp, published case revision and predecessor decision. A retry
with the same key and payload returns the published result; changed payloads or
stale revisions fail explicitly. A later decision supersedes the effective
meaning of the earlier decision for that pair, without deleting history.
Identity and relationship decisions have separate histories. Directed relationship
decisions are keyed by type and direction, so employment and ownership between
the same actors can coexist; withdrawing one does not remove the other.

The domain derives actor groups from current `same_actor` decisions. Original
candidates, source-local mentions, statements and proposed links are never
rewritten. `different_actors` decisions constrain grouping. If a new decision
contradicts a transitive confirmation chain, publication stops with an explicit
conflict; users must withdraw the conflicting confirmations first. This avoids
silently choosing which knowledge to discard.

`investigation.case.items` exposes revision-pinned `actor_groups`,
`actor_relationships` and `association_reviews`. Group member lists are bounded
and state their total counts and abbreviation status. Review history is shown
newest first. The shared relationship renderer shows loaded groups and reviewed
relations, including isolated actors. Missing pages and projection limits are
explicit; no unseen association is manufactured to complete a diagram.

The history lets users select an earlier pair for reconsideration. A failed
publication preserves the decision draft and its retry key; users can retry the
same operation or explicitly unlock the draft for editing. An unknown outcome
cannot overwrite newer state because a new attempt still requires the expected
case revision. Navigating away during publication does not reopen the old case
when its response arrives.

Reports freeze both review history and the derived actor projection at the
selected revision. They label decisions as analyst-confirmed and retain the
underlying source positions. Analyst confirmation does not become machine-
verified causality or independent log corroboration.

Contract cases have been written for grouping, withdrawal, transitive conflicts,
review provenance and roundtrip, host-only execution, model/app visibility,
idempotency, revision conflicts, case binding and late responses. Frontend cases
also cover explicit selection and decision submission without a model turn.
These changes have only received static review and `git diff --check`;
compilation, CTest, frontend tests and functional/browser/model smokes have not
been run, as requested.
