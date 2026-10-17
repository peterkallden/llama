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
immediate retry idempotent.

“Save” performs no model call. “Save and review” saves first and submits a
normal daemon turn that receives the document reference and asks the agent to
review the changed content. The agent may use existing research/tool routes
and may create a bounded `document.propose_patch` proposal. A proposal never
changes the report. The user must accept it in the editor, which invokes
`document.apply_patch` against the expected current revision. Stale proposals
are rejected and must be reviewed again. Accepting a proposal is not a factual
verification verdict.

Changed blocks are marked as user-authored and unreviewed. Their old evidence
references may remain for navigation, but they do not confer support on the
changed text. Unchanged blocks preserve their provenance and review status.
The editor sidebar presents saved block review/support states separately from
the bibliography; these statuses refresh after a successful save.
The report's evidence snapshot is immutable; new research belongs to a later
explicit report composition/review, not a mutation of the original snapshot.

The host grants write access only to the exact first-party document-editor
resource. Other MCP Apps retain their existing read-only restrictions. Direct
model calls to `document.save` and `document.apply_patch` require host
confirmation. The app bundle is embedded and advertised only when its build
dependencies are available; ordinary C++ builds do not require Node.js.

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
