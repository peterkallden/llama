# Document editor MCP App

The first document editor is a first-party MCP App at
`ui://llama-agent/apps/document-editor`. It edits immutable report revisions
and explicitly imported TXT, Markdown, and DOCX documents. The editor is a standalone embedded app using
the pinned `@toast-ui/editor` package; Markdown/GFM parsing and block identity
remain host responsibilities.

## User-authored documents

The web workspace offers **New**, **Upload**, and an **Open or import** picker.
New creates and publishes an empty first revision with a stable document ID;
the title and Markdown body are saved together as later immutable revisions.
Uploaded files are stored in the active project scope, or the current session
when no project is selected, and are imported as new documents with the source
resource and importer version retained as provenance. Uploading from this
workspace does not attach the file to a chat or start an agent turn.

Opening a different document, importing a file, or creating a new document
checks the editor draft first. The user can save, discard only local unsaved
changes, or cancel the switch. Download exports the editor's current Markdown
and title directly through the host, including unsaved content; it does not
publish a revision. The host performs the download because the embedded app's
sandbox does not grant direct download permission.
Closing a user document uses the same save/discard/cancel choice and does not
start a review turn. Report documents retain their existing review-on-close
flow; Review remains an explicit toolbar action for user documents.

The picker lists structured document and report revisions as **Open**, and
supported plain text, Markdown, and DOCX resources as **Import**. It reads at
most 1,000 resources per scope in a refresh; when more are available, the UI
directs users to the Resources workspace to find older items. Project resources
remain project-owned when opened from a project chat, while session resources
remain restricted to their originating session.

## Revision and review flow

`document.import` accepts an explicitly selected, host-authorized source
resource and an operation ID. It supports TXT, Markdown, and DOCX. The
original resource is never changed; imported documents retain its URI,
content fingerprint, and importer version. TXT is represented as a fenced
literal-text block so Markdown punctuation is not interpreted as formatting.
DOCX import uses the configured Pandoc DOCX-to-Markdown processor with GFM
output so tables are compatible with cmark-gfm and the editor; missing
processors, invalid files, limits, and conversion failures are explicit and
leave the original intact. Unsupported formatting or images may be reported
as import limitations. Reimporting the same source content and importer
version reopens the existing document's latest published revision. A distinct
uploaded resource creates a distinct document identity.

Markdown filename extensions take precedence over a generic `text/plain`
upload type. A successful empty DOCX conversion is an empty editable document.
The host accepts either a direct processor output or the persisted resource
returned by the processing service, re-authorizing and checking its media type
and byte limit. Interrupted imports publish a verified head manifest on retry;
an existing corrupt manifest is an error and is not silently replaced.
DOCX images become alternative text with an explicit import limitation. The
original DOCX retains its media; exported Markdown does not refer to temporary
image files belonging to the conversion process.

The attachment list offers an explicit pen icon named “Redigera dokument” for
supported file types; uploading alone does not import or open a document. The
editor opens in a shared workspace: the document stays in the main pane and the
existing chat moves to a narrow side column. The editor iframe remains mounted
while chat updates, so its selection and local draft survive the layout change.
At widths below 1,000 px the document and chat stack vertically.
Closing restores the prior workspace view. Connection, status, metrics, app
launchers and events remain in the independent toolbar floater; authentication
expands when required and collapses when connected. The main workspace follows
viewport width, with a compact chat column beside the editor or another app.

The document title and close icon form one workspace tab. While editing,
the Chat tab, app launchers, new-session action, and other document-open actions
are disabled. Close the document through its tab to save and queue review before
switching views. The embedded editor has no second close button. Attachment
removal uses an outlined trash icon and removes only the session attachment;
it does not delete the original file or saved document revisions.

Saved report revisions appear in chat as read-only structured cards. Their Edit
action opens this same editor, and Download uses the host's `rendered_markdown`
for that saved revision. Report blocks expose their saved support/review status
and related source/observation references. Hover/focus previews are local;
clicking a reference opens a scrollable inspection dialog. Dataset sources use
Data Explorer, other sources offer original or scoped saved-file links, and
tool results come from the frozen snapshot. Presentation does not rerun a tool
or promote reference integrity to factual verification. Browser rendering is
sanitized and does not fetch embedded media. Large reports initialize their
Markdown presentations as blocks enter the viewport.
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

The toolbar uses outlined icons: a counterclockwise arrow reloads the saved
revision, a floppy disk saves, and a scientist face saves and requests review.
Each icon retains a tooltip and accessible action name. Licenses remains a text button.
The editor toolbar stays visible while scrolling the document.

“Save” performs no model call. Before an ordinary chat message is sent while a
document is open, the editor saves pending changes and returns a typed binding
for the saved document URI, immutable revision, and selected block IDs. The
message is not sent if save or revision checks fail, and the chat draft remains
in place. The host re-authorizes the document and validates the revision and
block IDs, then creates a Markdown resource containing the selected blocks
and their immediate neighbors. It does not serialize document JSON metadata
or attach the complete document to the model. An empty selection projects the
whole Markdown revision; large projections use the existing host-controlled
chunk path. The typed resource-read binding requires an actual read of that
projection; metadata-only inspection and unrelated tool calls do not satisfy
it. Document content remains data, not instructions.
The projection retains host-owned `evidence_ids` and `source_refs` for each
selected block, together with document URI, revision, content fingerprint and
block-selection lineage. These references preserve provenance but do not
transfer factual support to changed user-authored text. Large projections use
the same processor and chunk path as other text, and the read gate checks the
exact projection revision and required chunk coverage before synthesis.
Sending a chat message saves the document but does not start a separate review.

"Save and review" registers a normal daemon turn after saving. The document tab's × button
“Spara, granska och stäng” saves and queues the changes since the editing
session opened or since the previous review request, then restores the prior
workspace as soon as the request is registered. If another turn is running, the
review waits in a session-scoped browser queue and starts when that turn
finishes. A failed review stays visible in chat with a retry action. Queue
entries use a stable operation/turn ID so retry does not create duplicate chat
turns. The review turn asks the agent to
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

The typed review binding optionally includes `base_document_uri`. The session
host re-authorizes that resource, checks that it is an earlier revision of the
same document, and renders both revisions into the scoped Markdown reading
resource. The initial prompt contains the review task, not accumulated JSON
diff previews. The comparison follows the existing chunk and full-coverage
requirements. Its fingerprint identifies the actual projection content, while
metadata retains the source revisions and selected blocks. Complete-document
selection uses an empty block-ID list; partial selections allow up to 128 IDs.
Queue persistence must succeed before the editor is told that review has been
registered. A browser storage failure leaves its pending review available for
retry in the open editor.
Review context retains document order and the selected flag on adjacent selected
blocks. It includes the immutable document URI, revision, section IDs and block
IDs needed for bounded patch proposals. The first-party host checks the review
binding against the editor's currently opened revision before submitting it.
The sidebar displays the host-computed diff before acceptance. Accepted model
changes remain model-authored and unreviewed; accepting them does not transfer
the original block's factual support. Save failures keep the editor open. A
queued review is reported separately from its eventual model result; turn
failures appear in chat and can be retried without saving another revision.
If newer edits appear during review registration, the registered review keeps
its saved revision and the editor stays open with the newer draft. Navigation
only proceeds after the editor explicitly confirms that it has closed.
The host waits for that acknowledgement before unmounting a navigation-driven
close; an editor-driven close restores the layout through its close event.
Locally opening an attachment does not pin its initial tool result: later
agent patch proposals continue to reach the same mounted editor.
Queued reviews wait for the current HTTP turn response as well as its event
stream, so an early completion event cannot start the next turn prematurely.
Draft persistence records whether text is actually unsaved. A clean saved
projection reopens from host-normalized Markdown rather than being mistaken
for a new edit because whitespace or rendering was normalized during save.
Failure and cancellation events abort the browser's pending HTTP turn request
and retain the daemon's reason, including deadline expiry. A cancelled turn
therefore cannot leave the composer permanently in its sending state.
Ordinary chat turns submitted while another turn is active are persisted in the
browser's session-scoped queue with selected resource references and document
binding. They run in order after the active HTTP turn settles. Failed entries
remain visible with their original request context and can be retried. Starting
a new session invalidates in-flight browser callbacks so they cannot mutate the
new session's attachments or queue.

Queue draining is also bound to the session epoch: a late review, orientation,
or queued-chat completion cannot release a newer session's queue lock or write
its stored queue. After each queued chat completes, pending reviews and
attachment orientations are reconsidered, even when no further chat is queued.
Turn lifecycle events for a different turn ID do not change the current turn's
busy or error state; the events remain available in the event log.

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
The daemon's generic policy-gated-write default remains off. A first-party
MCP App invocation may reach the provider's write tools only through the
separate app/resource binding and per-tool write authorization; this scoped
app path must not be rejected by the generic default before those checks run.
The web example reports a missing upload URI or editor authorization instead
of silently ignoring “Open in editor”.

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

## Verification

`test-agent-artifact` covers whole-document and selected-block context, source
order, stable patch references, and DOCX image alternatives with retained GFM
tables. `test-tool-adapters` covers persisted processor results, empty documents,
import publication failures, retry, and cached DOCX export. The DOCX processor
CTest keeps assertions enabled in release builds and checks the `--to=gfm`
command, output media type, and conversion direction.

Run the focused tests with the configured build tree:

```sh
ctest --test-dir <build-dir> -R '^(test-agent-artifact|test-tool-adapters|llama-agent-docx-text-processor-ctest)$' --output-on-failure
```

The existing Swedish table fixture can also check the installed converter;
put its output in the configured smoke directory:

```sh
pandoc --from=docx --to=gfm --wrap=none pocs/agent/smoke/data/fixtures/document-table/TAB6695_sv.docx -o <smoke-dir>/imported.md
```

This converter check and the model-free tests do not establish a completed
browser/daemon/model editing session.

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
