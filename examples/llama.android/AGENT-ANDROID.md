# Android agent connections

The Android example can run an embedded agent with a local GGUF model or an
HTTPS inference profile. The settings dialog configures the model endpoint,
model ID, context and output limits, provider capabilities and optional API
key. The key is kept in Android Keystore backed storage and is not included in
the model profile. An OpenAI compatible endpoint may require tool name mapping,
`max_completion_tokens` and an explicit reasoning effort; these are independent
profile settings.

The Server button opens the existing agent web application at a configured
HTTPS address. Its project, session, research and document functions depend on
the connected host's configuration and tool authorizations. Enter the web
server's bearer token in the web application's connection settings when needed.
The local session settings select a session and optional project. Saved HTTP
profiles can be selected in the HTTP configuration dialog. Save reply creates a
new app-private text attachment, and Edit saves another copy that can be used in
the next turn; neither operation changes the source document.

Imported local attachments are copied into app private storage. The embedded
agent issues scoped resource handles and exposes the existing resource inspect
and paged read tools. Text support depends on the media type reported by the
Android document provider; other formats require an available processor or an
agent server. The embedded resource store uses a SQLite catalog and filesystem
blobs, so issued resource handles can survive a process restart. MCP resource
tools with names in the local resource, dataset, data and statistics families
are exposed with a `remote_` prefix to distinguish them from local tools.
Research drafts use the same host checkpoint and scoped revision contracts as
the daemon, with SQLite indexes and immutable resource payloads. Local document
conversion and source availability still depend on the tools actually exposed
in the active view; provider-owned resource handles are not local handles.

HTTP profiles and the server view are not substitutes for device validation.
Verify authentication, attachment selection, session recovery and document
editing on an Android device or emulator with the intended agent server.

## Starter bootstrap: remaining work

Android initializes the `research` tool catalog, and the Gradle source set
includes the shared `pocs/memory/bootstrap` directory as assets. The embedded
runtime does not yet read or install those assets. Consequently, packaging the
JSON does not make its procedures, blueprints, or workflows available to the
agent. Asset inclusion has not yet been verified in a newly built APK.

Complete the following before treating the starter library as operational:

- Connect asset loading to the existing package parser and bootstrap installer.
  Preserve the installer's identity, conflict, and version rules across app
  restarts and upgrades; do not overwrite user-owned data.
- Resolve procedure retrieval for both local and HTTP inference. The existing
  installer requires the same embedder as normal memory retrieval, while the
  SQLite memory search also supports text-based retrieval when a query has no
  embedding. Evaluate an explicit host-selected lexical mode or a correctly
  configured embedding provider; do not manufacture vectors or mix embedding
  spaces. Verify that installed procedures are discoverable by the actual
  runtime retrieval path.
- Provide installed blueprint candidates to the common session host and ensure
  their workflow bindings resolve through the existing plan store. Persisting
  records alone does not make them selectable. Check capability eligibility
  against the tools actually available on Android and the connected MCP host.
- Add contract coverage for initial installation, restart, package upgrades,
  conflicts and retrieval, followed by an Android device or emulator scenario
  that exercises selection and execution with the packaged library.

# MCP resources and research persistence

When an authorized remote tool returns a resource link, Android acquires a
complete copy in the current session before passing the result to the agent.
The normal local resource and research checkpoint stores then retain the copy
across restarts. When the remote provider exposes an authorized paged reader,
the host checks each page's resource identity, size and SHA-256; otherwise it
requires a complete MCP resource response with a verifiable declared size.
The host limits one result to 16 links, 10 MiB and 64 paged reads. A resource
that cannot be fully acquired within these limits fails explicitly; partial
bytes are not reported as a complete resource. The page budget is separate
from research's later chunked analysis and does not yet provide resumable
remote acquisition for larger resources.
