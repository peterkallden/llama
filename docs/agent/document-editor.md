# Document editor MCP App

The first document editor is a first-party MCP App at
`ui://llama-agent/apps/document-editor`. It edits immutable report revisions
and explicitly imported TXT, Markdown, and DOCX documents. The editor is a standalone embedded app using
the pinned `@toast-ui/editor` package; Markdown/GFM parsing and block identity
remain host responsibilities.

## Revision and review flow

`document.import` accepts an explicitly selected, host-authorized source
resource and an operation ID. It supports TXT, Markdown, and DOCX. The
original resource is never changed; imported documents retain its URI,
content fingerprint, and importer version. TXT is represented as a fenced
literal-text block so Markdown punctuation is not interpreted as formatting.
DOCX import uses the configured Pandoc DOCX-to-Markdown processor; missing
processors, invalid files, limits, and conversion failures are explicit and
leave the original intact. Unsupported formatting or images may be reported
as import limitations. Reimporting the same source content and importer
version reopens the existing document's latest published revision. A distinct
uploaded resource creates a distinct document identity.

The attachment list offers an explicit “Open in editor” action for supported
file types; uploading alone does not import or open a document.
`document.open` returns the saved immutable report or imported document revision and an editable
Markdown projection. `document.save` compares `expected_revision` with the
current host manifest, parses the full document, and publishes a new immutable
revision. A conflicting save leaves the browser's local draft intact. An
unchanged save does not increment the revision. `operation_id` makes a
retry idempotent, including unchanged saves. The editor retains the pending
operation and payload until the host confirms it. A retry uses that same
operation even if newer local edits already exist.

Edited report resources are not treated as published heads until the head
manifest exists and matches the saved revision and Markdown. An interrupted
operation/head write can therefore resume from its original base without
exposing a partially published revision.
A later published report composition supersedes an older editor head. Both
export and editor revision selection consult the same current-revision check,
so continued editing follows the recomposed report's revision and lineage.

“Save” performs no model call. “Save and review” saves first and submits a
normal daemon turn with a typed document URI, immutable revision, and selected
block IDs. The host re-authorizes the resource and validates the revision and
block IDs before adding bounded document context. An empty block selection
means the whole document. The turn asks the agent to
review the changed content. The agent may use existing research/tool routes
and may create a bounded `document.propose_patch` proposal. The model supplies
`section_id`, `block_id`, replacement `text` (up to 16 KiB), and `summary`.
The IDs come from `document.open`; the host rejects unknown blocks and text
that escapes the selected block's structure. The legacy full-Markdown input
remains readable by host callers. A proposal never
changes the document. The user must accept it in the editor, which invokes
`document.apply_patch` against the expected current revision. Stale proposals
are rejected and must be reviewed again. Accepting a proposal is not a factual
verification verdict.
The sidebar displays the host-computed diff before acceptance. Accepted model
changes remain model-authored and unreviewed; accepting them does not transfer
the original block's factual support. Save/review failures are returned to the
editor rather than acknowledged as successful review starts.

Changed blocks are marked as user-authored and unreviewed. Their old evidence
references may remain for navigation, but they do not confer support on the
changed text. Unchanged blocks preserve their provenance and review status.
The editor sidebar presents saved block review/support states separately from
the bibliography; these statuses refresh after a successful save.
For reports, the evidence snapshot is immutable; new research belongs to a
later explicit report composition/review, not a mutation of the original
snapshot. Imported documents retain their original upload and may link review
findings to newly gathered research without claiming the original file changed.

The host grants write access only to the exact first-party document-editor
resource and its currently opened document revision. Late invocations cannot
replace this grant while a local draft has changes. Other MCP Apps retain their existing read-only restrictions. Direct
model calls to `document.save` and `document.apply_patch` require host
confirmation. The app bundle is embedded and advertised only when its build
dependencies are available; ordinary C++ builds do not require Node.js.

## Sandbox drafts and asynchronous results

The sandbox uses an opaque origin and cannot access browser `localStorage`.
The first-party extension `ui/llama-agent/document-draft` asks the web host to
read/write a draft for its host-confirmed open document. Storage keys include
session, provider, document identity and base revision. The host uses browser
storage when available and retains the newest copy in memory when a quota or
privacy restriction blocks persistence. The editor reports this limitation.

Open responses are checked against the edit sequence. Newer edits survive late
loads, saves and patch acceptance; concurrent save clicks do not create parallel
mutations. A tool invocation updates the mounted app instead of remounting it.
The saved revision and local draft stay distinct until a save succeeds.
At narrow widths the sidebar stacks below the editor and the preview uses tabs.
The editor uses TOAST UI's bundled dark theme for readable controls and text.

The catalog allows up to 8 MiB for `document.open` and 64 KiB for bounded
save/proposal results. The configured host result ceiling still applies;
oversized results fail explicitly. Stored reports/proposals have a 4 MiB
serialized read limit, checked before publication or acceptance.

## Build and dependency notices

The app bundle uses the exact TOAST UI version in
`examples/llama-agent.web/package-lock.json`. Build it with:

```sh
cd examples/llama-agent.web
npm ci
npm run build:document-editor
```

The build disables TOAST UI usage statistics and embeds all CSS and JavaScript
in the MCP resource; it does not load a CDN at runtime. The host rejects UI
resources above 2 MiB. The Debian package installs third-party notices beside
the application documentation. Project code remains MIT; bundled dependencies
retain their own licenses. See [`../third-party/NOTICE.md`](../third-party/NOTICE.md).

## Current scope

The first version supports immutable report and imported-document revisions,
local draft recovery, review proposals and user-approved patch application.
Ordinary documents have no report coverage status or automatic factual
support. Research stays in normal agent turns and has separate provenance.
It does not provide multi-user simultaneous editing, arbitrary local filesystem
access, HTML, editable PDF/OCR, spreadsheets, or a legal chain-of-custody system. The browser workflow has unit and
host-bridge tests; a full installed-package/browser/daemon end-to-end test is
still a separate verification requirement.
Regression tests also execute the editor script with simulated host replies,
covering sandbox draft storage, late opens, save retries and concurrent edits.
