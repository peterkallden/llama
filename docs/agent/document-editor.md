# Document editor MCP App

The first document editor is a first-party MCP App at
`ui://llama-agent/apps/document-editor`. It edits host-composed report
revisions, not arbitrary files. The editor is a standalone embedded app using
the pinned `@toast-ui/editor` package; Markdown/GFM parsing and block identity
remain host responsibilities.

## Revision and review flow

`document.open` returns the saved immutable report revision and an editable
Markdown projection. `document.save` compares `expected_revision` with the
current host manifest, parses the full document, and publishes a new immutable
revision. A conflicting save leaves the browser's local draft intact. An
unchanged save does not increment the revision. `operation_id` makes an
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
normal daemon turn that receives the document reference and asks the agent to
review the changed content. The agent may use existing research/tool routes
and may create a bounded `document.propose_patch` proposal. The model supplies
`section_id`, `block_id`, replacement `text` (up to 16 KiB), and `summary`.
The IDs come from `document.open`; the host rejects unknown blocks and text
that escapes the selected block's structure. The legacy full-Markdown input
remains readable by host callers. A proposal never
changes the report. The user must accept it in the editor, which invokes
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
The report's evidence snapshot is immutable; new research belongs to a later
explicit report composition/review, not a mutation of the original snapshot.

The host grants write access only to the exact first-party document-editor
resource and its currently opened report revision. Late invocations cannot
replace this grant while a local draft has changes. Other MCP Apps retain their existing read-only restrictions. Direct
model calls to `document.save` and `document.apply_patch` require host
confirmation. The app bundle is embedded and advertised only when its build
dependencies are available; ordinary C++ builds do not require Node.js.

## Sandbox drafts and asynchronous results

The sandbox uses an opaque origin and cannot access browser `localStorage`.
The first-party extension `ui/llama-agent/document-draft` asks the web host to
read/write a draft for its host-confirmed open report. Storage keys include
session, provider, report identity and base revision. The host uses browser
storage when available and retains the newest copy in memory when a quota or
privacy restriction blocks persistence. The editor reports this limitation.

Open responses are checked against the edit sequence. Newer edits survive late
loads, saves and patch acceptance; concurrent save clicks do not create parallel
mutations. A tool invocation updates the mounted app instead of remounting it.
The saved report and local draft stay distinct until a save succeeds.
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

The first version supports immutable report revisions, local draft recovery,
review proposals and user-approved patch application. It does not provide
multi-user simultaneous editing, Yjs synchronization, arbitrary Markdown file
editing, or a legal chain-of-custody system. The browser workflow has unit and
host-bridge tests; a full installed-package/browser/daemon end-to-end test is
still a separate verification requirement.
Regression tests also execute the editor script with simulated host replies,
covering sandbox draft storage, late opens, save retries and concurrent edits.
