# Agent Beta Assurance

Status: Conditional for the executed Windows/Debug beta scope; the focused
resource-processor verification and the PDF page-image path through local,
Docker, and Kubernetes execution passed. Broader sandbox coverage, OCR,
cache reuse, and model-backed coverage remain qualified.

This document is the assurance record for the `llama-agent` beta milestone.
It separates milestone criteria from the test evidence collected for a
specific branch, commit, platform, and date. A checked item is only valid
when the corresponding verification record contains evidence for it.

## Milestone

- Milestone: Agent runtime beta
- Integration branch: `feature/llama-agent`
- Baseline: `origin/master`
- Scope: reflective, deliberate, research, tools, sessions, scheduling,
  resources, memory, plans, daemon, JSONL, and MCP hosts

The latest resource-processor extension also covers the model-free Pandoc
direction contracts for ODT-to-Markdown and HTML-to-Markdown. Live sandbox
execution remains environment-dependent and is not claimed by this entry.
The current model-free processor smoke also covers DOCX-to-document-JSON and
HTML-to-document-JSON command contracts. These outputs are bounded derived
document representations; table-to-dataset extraction remains a separate,
limited adapter concern.

The `run` CLI now documents the tracing and plan-summary switches used by the
focused Qwen/Nomic document-table helper. The helper is an optional
model-backed diagnostic, not a deterministic assurance gate: in the recorded
Windows Debug attempt both models loaded, but the small Qwen CPU inference did
not complete within more than 30 minutes and produced no tool trace. This is
reported as not complete below, not as model-backed pass coverage.

## Gate definitions

### Build

- [ ] Supported Linux build passes
- [x] Supported Windows build passes for the configured Debug agent scope
- [ ] Agent targets compile without new agent warnings

### Core runtime

- [x] Reflective assurance/runtime contracts are covered by model-free tests
- [x] Deliberate mode creates and executes a plan
- [x] Research records sources and evidence
- [x] Late escalation is bounded to one escalation per turn
- [x] Research verification can perform its bounded reopen
- [x] Cancellation reaches a terminal state
- [x] Cancellation and deadlines stop automatic continuation before the next slice
- [x] Continuation checkpoints remain host-owned and are transported through the daemon turn result

### Tools and authority

- [x] Tool calls are validated against the resolved tool view
- [x] Unknown tools fail safely
- [x] Policy-gated tools cannot bypass host policy
- [x] Host configuration resolves capabilities into an immutable tool profile snapshot
- [x] Startup readiness reports the active profile and resolved tools
- [x] Client profile/write-authority overrides are rejected at the daemon boundary
- [x] User-supplied resources can flow through resource references
- [x] Memory and resource authorities remain host-owned
- [x] Developer workspace tools are implemented with bounded host-native tests
- [x] Document-table tools are exposed in the research and all-configured model tool views

### Sessions and scheduling

- [x] Turns are serialized per session lane
- [x] Multiple sessions can progress independently
- [x] Pending operations can be cancelled
- [x] Session reset and close are covered
- [x] Inference capacity is bounded independently of daemon workers
- [x] Inference priority, FIFO ordering, and cancellation are covered

### State and persistence

- [x] Memory scope is enforced
- [x] Plan scope is enforced
- [x] Resource authority and scope are enforced
- [x] Resource lineage and bounded range-read contracts are modelled in the shared runtime/tool seams
- [x] Chunk observations validate parent lineage and duplicate indexes
- [x] Continuation checkpoints carry bounded chunk progress through daemon transport
- [x] Deterministic text chunk boundary policy is covered by an agent smoke
- [x] Resource chunk ranges are planned and read through bounded store calls
- [x] Runtime preflight attaches the first planned chunk through host-resolved tooling
- [x] One active chunk chain advances through the existing session lane
- [x] Processed chunks are recorded as parent-linked plan observations
- [x] Chunk planning and processing are projected as typed daemon events
- [x] Context-budget accounting selects chunk size from resolved turn budgets
- [x] Chunk synthesis distinguishes complete, incomplete, and structural conflict states
- [x] Final synthesis is gated on complete chunk observations in the runtime driver
- [x] Context-pressure evaluation can stop before draft inference and create a continuation checkpoint
- [x] Continuation checkpoints carry a bounded working-state projection with plan/resource provenance
- [x] Context-pressure continuation re-enters the same driver operation with bounded working state
- [x] Bounded context compaction reuses working-state, policy-pack, and input-resource contracts
- [x] Chat output stopped by the generation limit is marked incomplete rather than accepted as final
- [x] Truncated chat tool-call output is rejected before parsing or dispatch
- [x] Tool-free plain-text chat can continue within the existing bounded turn lane
- [x] Planner JSON gets one bounded structural regeneration without partial-object concatenation
- [x] Reflection JSON gets one bounded structural regeneration before plan operations are accepted
- [x] Memory-learning JSON gets one bounded structural regeneration before candidate policy
- [x] Planner, reflection, and memory regeneration share one common bounded helper
- [x] Research workspace remains turn-scoped and ephemeral by default
- [x] Resource processing resolves MIME types before host-controlled processor selection
- [x] Derived processor outputs persist through the existing resource store and retain lineage
- [x] Normalized processor text enters the existing bounded resource chunking path
- [x] Simple Pandoc document tables reuse the existing dataset importer with document-table provenance
- [x] Document table catalogs resolve unique names, indexes and node IDs and reject ambiguous names
- [x] Document table list/resolve tools use host-owned callbacks and bounded locator validation
- [x] The first bounded local PDF text-layer processor has a model-free smoke
- [x] Resource processor source/output limits and processing lifecycle events are covered
- [x] Resource processor execution policies are host-configured, round-trip serialized, and fail closed for incompatible required backends
- [x] PDF page-image command construction is typed, bounded, host-owned, and covered by a model-free smoke without requiring renderer installation
- [x] Sandbox workspace staging uses the byte-oriented resource-store boundary and has a model-free binary materialization smoke
- [x] Local sandbox execution reuses `common_subproc`, maps virtual workspace paths, and has a bounded model-free smoke
- [x] Host sandbox helper exposes bounded raw execution results for resource-processing adapters without adding a second queue
- [x] Generic resource-processing host adapter is covered by a model-free CTest
- [x] Host adapter artifact collection rejects unsafe, missing, and oversized outputs
- [x] Hosted PDF page-image processing converts a bounded artifact into the existing derived-resource output and lineage contract
- [x] Tesseract OCR command construction is typed, bounded, host-owned, and covered by a model-free smoke
- [x] Tesseract OCR processor converts a bounded image artifact into a derived text/hOCR/TSV output and retains lineage
- [x] Local MuPDF E2E verifies PDF store input through PNG derived-resource persistence
- [x] Isolated Docker/Kubernetes PDF page rendering is verified with the local MuPDF worker image
- [x] Tesseract OCR local/Docker/Kubernetes E2E is verified
- [x] DOCX media detection and Pandoc text-processor contracts are covered by model-free CTests
- [x] Local Pandoc DOCX E2E creates a derived text resource with preserved lineage
- [x] Local Pandoc reverse E2E creates a DOCX artifact from a Markdown resource with preserved lineage
- [ ] Isolated Docker/Kubernetes DOCX processing is verified
- [ ] Derived-resource cache reuse is verified
- [x] Research workspace checkpointing has been evaluated through the existing turn-owned workspace contract
- [x] Model-free research CSV smoke verifies bounded resource reads, two-source provenance, comparison, checkpoint reload, and deferred incomplete coverage
- [x] Completed research is bridged into the outer plan as bounded evidence with source/resource provenance
- [x] Research tracing exposes bounded gap/task/iteration, source, evidence, comparison, checkpoint, completion, and failure transitions
- [ ] Full inference continuation and context compaction are verified

The resource chunking implementation is intentionally narrower than general
context management. It bounds and resumes large text resources through the
existing resource store, plan observations, checkpoints, and session lane. A
checkpoint now contains a bounded working-state projection, but arbitrary
conversation-history/tool-result compaction and model-output compaction remain
separate future assurance activities.

The working-state projection also has explicit per-field count and value-size
limits. Context-pressure continuation carries the projection through the
existing request field; the normal prompt renderer emits the compact-state
block once, so the continuation instruction does not duplicate it. These are
projection-boundary guarantees, not a claim that exact tokenizer accounting or
general context-window compaction is complete.

The runtime also exposes an optional host-owned context token estimator. It can
replace the conservative fallback estimate for an assembled inference path;
when it is absent or returns no value, the existing bounded estimate remains in
force. The estimator seam is covered by the continuation smoke contract, while
backend-specific tokenizer accuracy remains a separate assurance activity.

The lineage/range item is a contract milestone, not a claim that automatic
large-resource chunk scheduling or full context compaction is complete. The
original resource remains authoritative, derived chunks must retain parent
lineage, and non-zero range reads must remain bounded by host policy. Chunk
progress, cancellation, deadlines, and continuation belong to the existing
session lane and plan/checkpoint state; chunk observations are not automatic
long-term memories. Persistent catalog migration and controller-owned chunk
scheduling require separate verification before the assurance scope can claim
them.

Resource processors are organized below `tools/agent/resource/processors/`.
The local PDF text processor handles bounded direct text-layer extraction, and
the generic Pandoc processor handles DOCX-to-text, Markdown-to-DOCX, ODT/HTML
normalization and the planned structured spreadsheet direction. Neither
exposes a converter executable as a model-selected tool. PDF rendering, OCR,
and isolated DOCX processing require their respective execution-provider
assurance; the DOCX sandbox E2E is available as an explicit Docker/Kubernetes
run but is not checked until that live environment has passed.

Dataset handling remains a separate assurance track. A resource is the
authoritative source, a dataset is a typed analytical handle, a data tool
operates on authorized dataset references, and an exported artifact is a new
derived resource. The current CSV/Cozo foundation does not yet prove full XLSX
workbook fidelity or dataset-reference tool execution. The model-free XLSX
processor contract covers ZIP/XML-to-structured-JSON through the checked-in
bounded normalizer, and the bounded worksheet
importer now materializes separate typed dataset descriptors with source
workbook/sheet provenance and host-owned sheet selection. The broader claims
must be added only after focused contract tests and a bounded end-to-end fixture
pass. Advanced
statistics, formula evaluation, workbook visual fidelity and full spreadsheet
feature preservation remain explicitly out of scope for the first slice.

The importer/store seam has focused model-free coverage: a Pandoc table AST is
normalized into a worksheet envelope, which produces per-worksheet descriptors
and bounded row writes through the existing data-store interface. The live Cozo
store now also materializes bounded filter results as immutable derived datasets
with parent-dataset lineage and source-resource provenance. The corresponding
Cozo CTest verifies descriptor persistence, reload, materialization, lineage,
and fail-closed handling for truncated results.

The separate `llama-agent-data-manipulation-ctest` also covers the intended
model-free workbook-shaped path: bounded filter, inner join, grouped count/sum,
and immutable aggregate materialization with source-resource provenance.

Dataset descriptors now also have a generic origin projection for document
tables and spreadsheet worksheets. It can preserve a semantic representation
URI, document node ID, table index, caption and bounded header classification
without turning document context into a second store. Contract coverage checks
the bounded header classifier and Cozo descriptor reload of document-table
origin metadata. This is not yet evidence for automatic DOCX/HTML table
discovery or a complete document AST.

The Pandoc worksheet normalizer also carries explicit header metadata and
bounded header-classification status into the importer envelope. Headerless
tables are not silently assigned columns unless the bounded sample clearly
supports a first-row interpretation; other orientations remain a
normalization-required failure. PDF table extraction remains outside this
assurance scope.

The Cozo data-store CTest also covers bounded descriptive statistics, including
numeric count/min/max/mean/stddev, schema-selected numeric columns and grouped
statistics, plus IQR outlier detection with grouped thresholds. The
tool-adapter CTest covers the corresponding small-model normalizations.

The adapter CTest also covers the first dataset artifact-export slice. A
bounded dataset query is serialized as CSV through the existing
`artifact.export` seam, stored as a derived resource, and checked for
`text/csv` media type and dataset-parent lineage. Unsupported formats are
rejected by the tool contract. This does not yet claim streaming export, XLSX
writer support, or chart artifact generation.

The XLSX reference normalizer is covered by a deterministic model-free script
smoke. This still does not claim full XLSX workbook formula/layout fidelity or
a live XLSX-to-Cozo vertical slice. The current evidence covers the dataset/store
boundary and derived-operation behavior using deterministic descriptors and
bounded operation results; full configured XLSX-to-Cozo execution remains a
separate follow-up verification.

### Protocols and events

- [x] JSONL daemon protocol smoke passes
- [x] MCP stdio smoke passes
- [x] MCP HTTP smoke passes
- [x] Event stream is separate from the terminal response
- [x] Event ordering is covered for the relevant runtime modes
- [x] Event-stream reconnect behavior is tested

### Test suites

- [x] Agent CTest batch passes
- [x] Agent beta smoke groups pass for the executed model-free scope
- [x] Runtime smoke group passes
- [x] MCP smoke group passes
- [x] Daemon smoke group passes
- [x] Multi-session smoke passes
- [x] MCP vertical smoke passes
- [ ] Long-running stability test passes

### Tool assurance

Tools are assessed independently from the general runtime milestone. A tool is
not considered mature merely because its catalog entry exists or its target
builds.

| Criterion | Status | Assessment rule |
|---|---|---|
| Catalog contract | [x] | Name, schema, result shape, risk and executor are registered and validated |
| Capability/profile authority | [x] | Host configuration resolves the tool into an immutable profile view; clients cannot widen it |
| Boundary enforcement | [x] | Workspace, repository, sandbox, store and artifact boundaries reject out-of-scope access |
| Native tool behavior | [x] | Bounded workspace/repository/data/diagnostic adapters have direct tests |
| Mutation safety | [x] | Workspace mutation uses confirmation, scope checks and expected-content tokens |
| Sandbox execution | [ ] | Kubernetes live smoke passed; Docker live smoke timed out after 300 seconds, so Docker execution is not assured |
| Backend availability | [x] | Tools depending on unavailable backends are removed from the effective tool view |
| Result normalization | [x] | Tool results expose bounded status, summaries, diagnostics and resource references where applicable |
| Semantic diagnostics | [ ] | Symbol/reference keep a bounded text fallback and call hierarchy requires a semantic provider; clangd/LSP or a project index is not yet bound |
| Tool-specific smoke coverage | [x] | The complete model-free smoke executable set passed; model-backed resident coverage remains separate |

The tool-repair path has separate deterministic coverage. The provider CTest
checks host-owned dotted-name normalization, unique high-confidence fuzzy
resolution and preservation of ambiguous candidates. The runtime CTest checks
that a failed tool step suspends the active final answer until its repair pass
has completed. These tests do not claim that a model will select the intended
tool in every prompt.

The latest Qwen/Nomic CSV data smoke remains a known model-backed failure: the
structured plan selected an invalid `dataset.inspect` call and did not issue
`data.join`, `data.aggregate` or `statistics.describe`, even though the prose
plan mentioned a join. The smoke correctly reports this as a failed expected
tool assertion. The deterministic Cozo/tool contracts and repair tests remain
separate from that model-selection result.

The Cozo data-store test is conditional on `LLAMA_MEMORY_COZO=ON` and a
configured Cozo C API. The default agent build keeps that option disabled, so
the ordinary adapter tests verify the backend seam and tool contracts, while
the Cozo-specific test verifies bounded scans, result limits, ordering,
grouping, inner/left joins and operation-specific validation when the backend
is enabled.

The maturity labels used in `agent-runtime.md` describe the implementation
stage: `Foundation`, `Limited`, `Contract-level` and `Experimental`. Assurance
criteria describe whether a tool is safe and verified for the current host;
they are not a replacement for those maturity labels.

## Verification record

### Latest verification - 2026-08-13 document-table CLI/model slice

- Branch: `kallden/agent-resource-tools`
- Code changes: `run` usage now documents `--agent-trace` and
  `--plan-show-summary`; the CLI run/MCP parser smoke asserts both flags; a
  checked-in document representation and Qwen/Nomic document-table helper
  were added.
- Focused build: passed; `llama-agent-cli-run-mcp-smoke` and `llama-agent-cli`
  built in the E: Debug tree with MSVC and the E: sccache cache.
- CTest result: passed (`2/2`, `0` failed, `0` not-run):
  `llama-agent-cli-run-mcp-ctest` and
  `llama-agent-cli-document-table-ctest`.
- Model-backed result: not complete. Nomic loaded and generated the query
  embedding; Qwen loaded and entered CPU inference, but the run produced no
  stdout/stderr or tool trace after more than 30 minutes and was stopped.
- Qualified: this does not prove a document-table model path. The model-free
  host/tool contracts pass, while the small-model planning/inference path
  remains an explicit diagnostic and is not counted as assurance.

Each record must identify exactly what was run. Counts use `passed/total`;
skipped and unavailable tests are recorded separately rather than counted as
passes.

### Latest verification - 2026-08-12 dataset/tool focused slice

- Branch: `kallden/agent-resource-tools`
- Commit: `03ccea89b` (code); this assurance update is a separate local commit
- Build: clean Windows/MSVC Debug tree with Cozo enabled,
  `E:\llama-builds\agent-resource-tools-msvc-debug-clean`, serial build after
  the known MSVC PDB-lock retry
- Focused build: passed; Cozo data-store target and dataset/tool smoke targets
  built successfully
- CTest result: passed (`5/5`, `0` failed, `0` not-run) for
  `test-tool-adapters`, `test-agent-data-store-cozo`,
  `llama-agent-tool-provider-ctest`, `llama-agent-dataset-contract-ctest`,
  and `llama-agent-dataset-importer-ctest`
- Covered behavior: dataset-reference tool contracts, Cozo descriptor
  persistence/reload, bounded derived-dataset materialization, parent-dataset
  lineage, source-resource provenance, and rejection of truncated results
- Not claimed: full XLSX workbook formula/layout fidelity or a live configured
  XLSX-to-Cozo vertical slice, advanced
  statistics, formula evaluation, workbook visual fidelity, cross-backend
  transactions, or model-backed dataset analysis

### Latest verification - 2026-08-11 full agent/resource assurance

- Branch: `kallden/agent-resource-tools`
- Commit: `52cb88a53`
- Build: Windows/MSVC Debug, Cozo-enabled, `E:\llama-builds\agent-resource-tools-msvc-debug-fast`, four build threads
- Build result: passed (`112/112` build steps)
- CTest result: passed (`47/47` tests in `-L agent`, `0` failed, `0` not-run)
- Model-free resource smoke result: passed for the focused contract, backend,
  lineage, chunking, PDF, OCR, host-adapter, local PDF/OCR E2E, Docker
  sandbox, Docker PDF/OCR E2E, and Kubernetes PDF/OCR E2E checks
- Docker result: passed; Docker Engine `29.6.1` and the local
  `llama-agent-pdf-ocr-worker:local` image were available
- Kubernetes result: passed for the PDF/OCR E2E; the configured node was
  `Ready`. The generic Kubernetes sandbox smoke reported its backend exercise
  as skipped and is therefore not counted as passed coverage.
- Qwen model result: the static Qwen chat smoke passed with
  `Qwen2.5-1.5B-Instruct-Q4_K_M.gguf`
- Nomic model result: the embedding add/search diagnostics loaded
  `nomic-embed-text-v1.5.Q4_K_M.gguf`, added the test note, generated a query
  embedding, and returned the stored note
- Qwen/Nomic agent result: not complete. The current Debug `llama-agent`
  executable reports that plan Cozo support is unavailable when a plan DB is
  requested; a bounded reflective run without a plan DB remained too slow on
  the test CPU and was stopped. This is not counted as model-backed agent
  assurance.
- Phi follow-up: `Phi-3.5-mini-instruct-q4km-current.gguf` loaded successfully
  and reached the deliberate agent inference after Nomic retrieval, but the
  bounded CPU run also exceeded the practical smoke duration without a final
  result. It is therefore not counted as model-backed agent assurance.
- Not-run/qualified: full model-backed plan/continuation flow, derived-resource
  cache reuse, and full general context compaction remain outside this run
- Environment note: CMake UI asset download was unavailable in the isolated
  environment; CMake used its documented no-embedded-UI fallback. This did not
  affect the agent build or the executed agent tests.

This checkpoint is the current top-level evidence record. Earlier verification
blocks below are retained as historical records and are intentionally ordered
after this checkpoint.

### Resident server-context initialization verification - 2026-08-07

| Field | Value |
|---|---|
| Branch | `kallden/agent-selection-learning` |
| Commit | `4813c990a` |
| Platform | Windows / MSVC |
| Build configuration | Debug, Cozo enabled, Visual Studio Build Tools, build artifacts on `E:\llama-builds\agent-selection-learning-msvc-debug-17` |
| Focused build | `llama-agent-inference-smoke` and `llama-agent-daemon` rebuilt successfully with two build workers |
| Model-free CTest | `llama-agent-inference-ctest`: `1/1 passed`, `0 failed`, `0 not-run` |
| Focused smoke | `llama-agent-inference-smoke runtime-server-context-host-invalid-model-paths`: passed |
| Model-backed daemon smoke | Qwen2.5 1.5B instruct GGUF through `llama-agent-daemon --model ... --default-mode chat --agent-plan off --n-predict 8 --context-size 2048 --worker-count 1 --queue-capacity 2`: passed, response `OK.`, `turn.completed` |
| Negative daemon smoke | Missing model path failed quickly with `turn.failed` and `resident server_context model does not exist` |
| Not run | Full agent CTest label, Docker sandbox backend, Kubernetes backend, Nomic-backed embedding flow |

The regression was isolated to the daemon-owned resident `server_context`
path on Windows. The host called server-context model loading before
initializing the llama/ggml backend and timer state used by the server loading
progress path. The agent-side resident host now performs the same one-time
common/backend/NUMA initialization before `server_context::load_model()` and
rejects empty or missing model paths before entering model loading.

### Latest verification - 2026-08-10

| Field | Value |
|---|---|
| Branch | `kallden/agent-resource-tools` |
| Commit | `64a18e549` |
| Platform | Windows / MSVC |
| Build configuration | Debug, Cozo enabled, Visual Studio 17 2022, build artifacts on `E:\llama-builds\agent-resource-tools-msvc-debug-fast` |
| Agent CTest | `37/37 passed`, `0 failed`, `0 not-run` |
| Focused resource/tool/runtime smokes | `9/9 passed`, `0 failed`, `0 not-run` |
| Initial regression during verification | `test-tool-catalog` failed because its profile-count assertion was stale (`8` instead of the current `9`) |
| Regression fix | `c607c5a60` updates the test expectation; the focused CTest and complete agent label were rerun successfully |
| Research/context sweep | `0091b4354`, `1b4c1c936`, and `64a18e549` add workspace checkpoints, bounded context compaction, and checkpoint propagation across slices |
| Docker/Kubernetes backend execution | Not-run in this checkpoint |
| Model-backed Qwen/Nomic execution | Not-run in this checkpoint |
| Decision | Windows model-free resource-processing/tool assurance passed; PDF rendering/OCR, isolated execution, cache reuse, live sandbox backends, and model-backed gates remain separate |

The focused resource/tool smoke set covered the generic processing contract,
service event path, media-type resolution, bounded PDF text extraction, and
the existing text/range chunkers. The first broad CTest invocation exposed only
a stale test expectation; it was corrected as a test-only commit and did not
indicate a runtime regression.

### Latest verification - 2026-08-14 compact model tool descriptions

| Field | Value |
|---|---|
| Branch | `kallden/agent-resource-tools` |
| Commit | 663664877 (with c249d08a7 and e50c3dfc0) |
| Build configuration | Windows / MSVC Debug, Cozo-enabled CPU tree, four build threads |
| `test-agent-runtime-json` | Passed |
| `test-tool-catalog` | Passed |
| Failed | `0` |
| Not-run | `0` for the focused contract set |
| Scope | Strict schemas generate bounded compact tool descriptions; resource handles use rN; host normalizes id to canonical URI |
| Not covered | Full agent CTest suite, model-backed Qwen/Phi execution, and dynamic tool-view narrowing |

The model-facing description is generated from the strict schema and is used
by native and MCP tool views. JSON parameters and host validation remain
unchanged. Focused tests cover schema rendering, resource handle
normalization, and the registered tool bridge.

### Previous verification - 2026-08-10 resource-processing checkpoint

| Field | Value |
|---|---|
| Branch | `kallden/agent-resource-tools` |
| Commit | `a4d28383c` (unique operation identity per resource read); `811d110ee` (operation-scoped resource processor binding); `9936db1dd` (bounded binary CLI resource upload); CLI PDF resource-read smoke `06f65c7b7`; CLI assembly `d3143098c`; resource_read integration `69915617d`; language metadata `06a56d51d`; cache foundation `59f5902a9`; Cozo persistence `75b0283a5` |
| Platform | Windows / MSVC |
| Build configuration | Debug, Cozo enabled, artifacts on `E:\llama-builds\agent-resource-tools-msvc-debug-fast` |
| Focused model-free CTest | `3/3 passed`, `0 failed`, `0 not-run` for the Cozo-enabled cache/language slice |
| Resource-read processing CTest | `1/1 passed`, `0 failed`, `0 not-run`; PDF text was materialized through the host provider and reused from cache |
| CLI assembly/provider CTests | `2/2 passed`, `0 failed`, `0 not-run`; native provider and CLI selection retain the host processing service, and CLI selection materializes a PDF text read |
| CLI resource upload CTests | `3/3 passed`, `0 failed`, `0 not-run`; text resources remain compatible and a bounded binary PDF upload preserves `application/pdf` metadata |
| Local MuPDF PDF E2E | Passed; PDF store input became a PNG derived resource with lineage |
| Docker MuPDF PDF E2E | Passed with `llama-agent-pdf-worker:local` |
| Kubernetes MuPDF PDF E2E | Passed through the ephemeral Job backend; diagnostic resources were cleaned up |
| Local Tesseract OCR E2E | Passed; rendered PNG became a derived text resource with lineage |
| Docker Tesseract OCR E2E | Passed with `llama-agent-pdf-ocr-worker:local` |
| Kubernetes Tesseract OCR E2E | Passed after retry with the same worker image; project-specific resources were cleaned up |
| Helm deployment | Not required for the current Job-per-operation backend |
| Operation-scoped processor binding | Passed; `resource_read` carries a host-owned operation id and can create a bounded provider for configured sandbox-backed page-image/OCR processors |
| Focused operation-provider CTest | `3/3 passed`, `0 failed`, `0 not-run`; tool provider, CLI selection, and daemon MCP configuration paths |
| Remaining scope | General automatic backend resolution for every processor family, true automatic language detection, broader sandbox assurance, and model-backed Qwen/Nomic execution |

This checkpoint verifies the same hosted PDF and Tesseract processor contract
across local, Docker, and Kubernetes execution placement. It does not claim that all
sandbox operations or all processor families have live backend coverage. The
Kubernetes run used the local Docker Desktop cluster and the development worker
image; a remote cluster requires a registry-visible, preferably digest-pinned,
processor image.

The language metadata slice was verified with Cozo enabled. Resource metadata
survives a store reopen, `language=auto` prefers accepted resolved or declared
metadata, missing language metadata fails closed without a fallback, and the
selected language participates in derived-resource cache identity. The current
`auto` mode is metadata-driven; broad language detection remains a separate
bounded processor capability.

The resource-read integration is verified for a local PDF text processor. A
semantic `resource_read(representation="text")` request can invoke the
host-owned processing provider for a non-text source, read the resulting
derived text resource, and receive the same bounded resource response as a
native text resource. Repeating the request reuses the processing cache. The
common tool adapter remains independent of the concrete processor service. The
operation-scoped factory seam is covered model-free: it preserves the active
turn operation identity and can be used by host assembly to create sandbox
bound page-image/OCR providers without adding a second scheduler. Live
processor execution still requires the configured backend and matching
sandbox execution class.

### Latest verification - 2026-08-07

| Field | Value |
|---|---|
| Branch | `kallden/agent-selection-learning` |
| Commit | `334fd987a041` |
| Platform | Windows / MSVC |
| Build configuration | Debug, Cozo enabled, Visual Studio 17 2022, build artifacts on `E:\llama-builds\agent-selection-learning-msvc-debug-13` |
| Agent CTest | `32/32 passed`, `0 failed`, `0 not-run` |
| Model-free agent smokes | `15/15 passed`, `0 failed`, `0 not-run` |
| Kubernetes sandbox CTest | `1/1 passed` (contract smoke only; live backend path not enabled) |
| Kubernetes client/cluster check | Not-run; `kubectl` is installed, but the local kubeconfig was inaccessible (`Access is denied`) |
| Docker sandbox CTest | `0/1 passed`, `0 failed`, `1 skipped` because Docker API access was denied |
| Model-backed Qwen/Nomic execution | Not-run in this checkpoint |
| Decision | Windows model-free assurance passed; sandbox backend execution, Linux, and model-backed gates remain separate |

The Kubernetes CTest entry validates the host/runtime contract, including
required image and network-policy rejection. It does not claim that a pod was
created. The live backend branch is opt-in through
`LLAMA_AGENT_KUBERNETES_SMOKE=1` and requires an accessible kubeconfig and
cluster. Docker is similarly kept separate from the passing contract and
model-free counts; a skipped sandbox test is not reported as a backend pass.

### Latest verification - 2026-08-04

| Field | Value |
|---|---|
| Branch | `kallden/agent-selection-learning` |
| Commit | `18d8e5bb8` |
| Platform | Windows / MSVC |
| Build configuration | Debug, Cozo enabled, Visual Studio 17 2022, build artifacts on `E:\llama-builds\agent-selection-learning-msvc-debug-13` |
| Memory contract change | Memory ID references use bounded strings: `minLength: 1`, `maxLength: 256`; catalog tests cover get/update/forget/link/compact schemas |
| Agent CTest | `24/24 passed`, `0 failed`, `0 not-run` |
| Kubernetes sandbox CTest | `1/1 passed` |
| Docker sandbox CTest | `0/1 passed`, `1 failed` (`Timeout` after 300 seconds) |
| Docker direct check | `docker info`/bounded `alpine:3.20` run also timed out; no Docker pass is claimed |
| Decision | Conditional assurance; Kubernetes is verified, Docker remains an open failed gate, and long-running/model-backed gates remain qualified |

The latest agent label includes the schema and adapter regressions for bounded
memory references. The native and reflection paths also accept compatibility
forms and canonicalize them before validation, while the catalog continues to
communicate the strict canonical object contract to the model.

The Docker result is recorded as failed rather than skipped because the smoke
executable was available and the backend was invoked. The smoke did not reach
its declared unavailable-backend return path (`77`); it remained alive until
the CTest timeout. A direct bounded Docker command reproduced the same host
behavior. This distinction must remain visible until the Docker subprocess or
daemon issue is resolved.

### Continuation sweep verification - 2026-08-04

| Field | Value |
|---|---|
| Branch | `kallden/agent-selection-learning` |
| Platform | Windows / MSVC |
| Build configuration | Debug, serial MSVC build, artifacts on `E:\llama-builds\agent-selection-learning-msvc-debug-cont1` |
| Full agent CTest after continuation and checkpoint changes | `26/26 passed`, `0 failed`, `0 not-run` |
| Continuation CTest | `1/1 passed` |
| Daemon protocol CTests | `2/2 passed` |
| Scope | Slice cancellation/deadline control and daemon checkpoint transport |
| Not claimed | Full context compaction, structural chat-output regeneration, or arbitrary context-window overflow handling |

### Resource chunk completion verification - 2026-08-05

| Field | Value |
|---|---|
| Branch | `kallden/agent-selection-learning` |
| Commit | `2b14b5bbb` |
| Platform | Windows / MSVC |
| Build configuration | Release, Visual Studio 17 2022, two build threads, artifacts on `E:\llama-builds\agent-selection-learning-msvc-release-e2e` |
| Focused CTest | `4/4 passed`, `0 failed`, `0 not-run` |
| Tests | continuation, daemon protocol, resource store, resource range chunker |
| Qwen resource synthesis | `passed`, four fixed resource chunks, two inference threads |
| Qwen log | `work/qwen-resource-synthesis-next/resource-synthesis.log` |
| Scope | Session-lane chunk advancement, resume selection, typed daemon events, synthesis completeness projection, configured byte budgets |
| Not claimed | Full arbitrary context compaction or Linux/Docker backend assurance |

### Chunk synthesis gating verification - 2026-08-07

The deliberate runtime smoke was verified on Windows/MSVC Debug using the
fresh E: build tree. Complete chunk observations allow final synthesis;
incomplete observations defer synthesis, and conflicting observations block
the synthesis step while preserving the plan evidence. The corresponding
`llama-agent-deliberate-runtime-ctest` entry is now registered in the agent
CTest block; a post-registration CTest run remains pending because the
existing Visual Studio build tree requires a clean regeneration.

### Full agent assurance verification - 2026-08-05

| Field | Value |
|---|---|
| Branch | `kallden/agent-selection-learning` |
| Commit | `b43f61e14` |
| Platform | Windows / MSVC |
| Build configuration | Release, Visual Studio 17 2022, `llama-agent-build-pack`, two build threads, artifacts on `E:\llama-builds\agent-selection-learning-msvc-release-e2e` |
| Agent CTest | `30/30 passed`, `0 failed`, `0 not-run` |
| Test selection | `ctest -C Release -L agent --output-on-failure --parallel 2` |
| Sandbox Docker | Not-run by request; Docker backend was not running |
| Sandbox Kubernetes | Not-run by request; Kubernetes backend was not running |
| Decision | Windows model-free agent assurance passed; sandbox and broader Linux gates remain outside this run |

### Full agent assurance verification - 2026-08-06

| Field | Value |
|---|---|
| Branch | `kallden/agent-selection-learning` |
| Commit | `8b1cb0747` |
| Platform | Windows / MSVC |
| Build configuration | Release, Visual Studio 17 2022, `llama-agent-build-pack`, two build threads, artifacts on `E:\llama-builds\agent-selection-learning-msvc-release-e2e` |
| Agent CTest | `31/31 passed`, `0 failed`, `0 not-run` |
| Test selection | `ctest -C Release -L agent --output-on-failure --timeout 900` |
| Covered additions | Context-pressure continuation, bounded text continuation, truncation guards, planner/reflection/memory JSON regeneration |
| Sandbox Docker | Not-run; Docker daemon was unavailable |
| Sandbox Kubernetes | Not-run; Kubernetes was unavailable |
| Decision | Windows model-free agent assurance passed; model-backed Qwen/Nomic and sandbox/Linux gates remain separate |

The Qwen/Nomic helper was also exercised against the same Release artifact.
The Nomic-backed memory add/search phase passed and returned the expected
`note-1` result. The first static Qwen chat phase did not produce output or
progress on this Windows run and was stopped after the process remained alive
without CPU activity; the retry used an external E: work directory and an
explicit `NUL` stdin, but showed the same behavior. Consequently, the
model-backed chat phases are recorded as not-run/blocked rather than as a
passing end-to-end result. This does not change the deterministic 31/31 CTest
result. The helper now accepts absolute `-BuildDir` and `-WorkSubdir` values
for external build and result trees.

### Previous current run

| Field | Value |
|---|---|
| Branch | `pocs/agent-tool-profiles` |
| Commit | This checkpoint; see `git log` for the immutable commit id |
| Date | `2026-07-27` |
| Platform | Windows / MSVC |
| Build configuration | Debug semantics, Cozo enabled, Ninja, four-way build; separate debug information disabled to avoid PDB/ILK disk exhaustion |
| Cozo | `LLAMA_MEMORY_COZO=ON`, `LLAMA_PLAN_COZO=ON`, configured Cozo 0.7.6 MSVC release library and DLL |
| CTest | Agent label 20/20 passed; Kubernetes label 1/1 passed; Docker label skipped in the normal run and hung when elevated, so no Docker backend pass is claimed |
| Complete model-free smokes | 28/28 passed; Docker backend smoke skipped (exit 77); resident model-backed smoke not run (requires `--model`) |
| Decision | Conditional assurance; model-backed, Linux, Docker backend and long-running gates remain open |

### Async lifecycle hardening

The async lifecycle checkpoint (`722b3f645`) closes the async lifecycle findings and
the later poll/cancel terminal-state race:

- After a successful inference submit, the task owns the inference-capacity
  lease. If operation registration fails, the driver requests task
  cancellation and does not release the lease a second time.
- Operation deadlines invoke the registered cancellation callback outside the
  operation-manager mutex while preserving `timed_out` as the authoritative
  terminal state.
- Terminal operation entries are moved out of the manager under the mutex and
  destroyed after the mutex is released, so destruction of an asynchronous
  task cannot block other operation-manager calls.

The regression coverage includes operation-registration failure with exact
single cancellation/release, deadline cancellation, and cleanup concurrency.
Operation terminal transitions are now monotonic from `running`, and the
operation-manager smoke deterministically covers the poll-ready versus cancel
race. The turn-driver smoke coverage also verifies the existing admission and
registration cancellation paths. The two focused runtime smokes passed, and
the complete agent CTest label passed 16/16 after the changes.

### Async tool cancellation and preemptive deadlines

The 2026-07-27 tool-runtime checkpoint adds explicit cancellation and deadline
propagation for host-owned asynchronous native and MCP tool calls:

- Each async call owns a cancellation state that is exposed through the runtime
  adapter and can be signalled by the operation manager.
- Effective per-call deadlines are copied into the pending operation reference,
  so manager polling can transition the operation to `timed_out` and invoke the
  cancellation callback outside the manager mutex.
- MCP stdio enforces cancellation/deadlines through bounded request handling and
  subprocess termination; MCP HTTP uses the remaining deadline to bound
  connect/read timeouts.
- Native synchronous handlers observe the same state before and after execution;
  interruption during a handler remains cooperative and is intentionally not
  claimed as hard preemption.

The native provider smoke now covers explicit cancellation of an in-flight async
fetch. Focused tool provider/runtime CTests passed 2/2 after the implementation;
the full agent label remains the final assurance gate for this checkpoint.

### Continuation control and checkpoint transport

The bounded continuation slice checkpoint (`kallden/agent-selection-learning`)
was verified on Windows/MSVC Debug using the E: build tree. The continuation
smoke passed cancellation and deadline cases that stop after the completed
slice and before the next inference request. The daemon protocol and JSONL
protocol smokes also passed checkpoint serialization/parsing, including
request/turn identity, plan revision, reason, sequence and resource
references. Full context compaction remains a separate, unimplemented gate;
this checkpoint does not claim that arbitrary model context can exceed a
model window.

### CTest evidence

| Batch | Passed | Failed | Skipped | Total | Result |
|---|---:|---:|---:|---:|---|
| Agent contracts/runtime (`test-agent-*`) | 8 | 0 | 0 | 8 | Passed |
| Tooling (`test-tool-*`, clangd, Cozo) | 5 | 0 | 0 | 5 | Passed |
| Agent runtime CTest smokes | 4 | 0 | 0 | 4 | Passed |
| **Agent CTest total** | **17** | **0** | **0** | **17** | **Passed** |
| Sandbox Kubernetes (`sandbox-kubernetes`) | 1 | 0 | 0 | 1 | Passed |
| Sandbox Docker (`sandbox-docker`) | 0 | 0 | 1 | 1 | Skipped; backend unavailable |

### Registered CTest inventory

The current Cozo-enabled build registers 64 CTest cases in total. The agent
verification slice is the `agent` label with 20 tests; the latest run passed
20/20. The sandbox labels are separate backend slices:

| Label | Registered tests | Current result |
|---|---:|---|
| `agent` | 20 | 20 passed |
| `sandbox-kubernetes` | 1 | 1 passed |
| `sandbox-docker` | 1 | 1 skipped; backend unavailable |
| Other repository tests | 44 | Not part of the agent assurance sweep |

The 64-test inventory is configuration-dependent. It includes general
repository tests whose executables were not built or run in this focused agent
verification, so the total inventory must not be reported as a 63-test agent
pass.

Commands and full output should be retained in the task handoff or CI log;
this file records the summarized result and the commit it belongs to.

### Smoke evidence

| Group | Passed | Failed | Skipped/not run | Total | Result |
|---|---:|---:|---|---:|---|
| Runtime smokes | 11 | 0 | 1 skipped, 1 not run | 13 | Passed with Docker/resident qualifications |
| MCP smokes | 7 | 0 | 0 | 7 | Passed |
| CLI smokes | 2 | 0 | 0 | 2 | Passed |
| Daemon smokes | 7 | 0 | 0 | 7 | Passed |
| Resource smoke | 1 | 0 | 0 | 1 | Passed |
| **Complete model-free smoke set** | **28** | **0** | **1 skipped, 1 not run** | **30** | Conditional |

## Known limitations

- Linux and model-backed Qwen/Nomic validation require their respective
  environments and are not implied by a Windows smoke run.
- The recorded model-free smoke groups are the executable groupings used by
  the current `pocs/agent/CMakeLists.txt`; there is no separate executable
  called “beta aggregate” in this assurance record.
- The resident model-backed smoke was not executed in this focused run because
  it requires an explicit `--model` GGUF path.
- Developer workspace tools are now in scope and have bounded adapter coverage;
  semantic indexing and end-to-end build/test execution remain separate gates.
- Kubernetes contract and CTest coverage passed, while the Docker backend smoke
  and CTest are skipped when the Docker daemon is unavailable.
- Research workspace checkpointing is not part of the first version.
- The continuation checkpoint contract and bounded agent-driver multi-slice
  behavior are covered by the dedicated continuation CTest; full
  conversation/tool result compaction, arbitrary structural regeneration, and
  model-backed continuation remain outside the verified beta scope. Bounded
  planner, reflection, memory-learning, and plain-text chat regeneration or
  continuation are covered by the current agent assurance run.
- Runtime-driver and host inference coverage now lives under the agent smoke
  tree and links the complete runtime-support closure; `tests/` remains the
  home for pure contract and component tests.
- Long-running stability and resource-retention testing remain separate from
  bounded functional smokes.
- A test is not considered passing merely because its target builds; its
  executable or CTest case must also have run successfully.

## Current tool-profile branch delta

The `pocs/agent-tool-profiles` work adds host-owned capability/profile
resolution, policy-bearing profile snapshots, startup `tooling` diagnostics and
explicit rejection of client attempts to select or widen tool authority. The
daemon reload contract treats `tools.profile`, `tools.capabilities` and
`tools.profiles` as restart-required; this preserves one immutable host tool
view for an instance instead of changing the exposed catalog during a live
session. Developer workspace tools now have bounded native adapter coverage,
including symbol/reference fallback and grouped test-failure diagnostics.
Kubernetes contract coverage passed; Docker remains environment-skipped when
daemon access is unavailable. Data-analysis execution classes are
represented by host configuration, while full operation-manager integration and
semantic project indexing remain separate verification gates.

## Exit decision

The milestone can be marked `Passed` only when all required gates are either
verified or explicitly accepted as out of scope, no open P0/P1 defect remains,
and the verification record identifies the exact branch, commit, platform,
date, and test counts.

### Decision record

- Decision: Conditional beta assurance for Windows/Debug model-free scope
- Date: 2026-07-26
- Commit: `88c6e4bb9`
- Reviewer: pending
- Notes: The complete model-free agent CTest label passed 17/17; Kubernetes
  passed 1/1 and Docker was skipped. Direct model-free smokes passed 28/30,
  with one Docker backend skip and the resident model-backed smoke not run.
  Linux, model-backed Qwen/Nomic execution, semantic indexing, checkpointing
  and long-running stability remain outside this run.

## Assurance history

Add a dated entry for every meaningful verification run. Do not overwrite a
previous result when the branch or test configuration changes.

### 2026-07-26 — Windows Cozo Debug assurance run

- Branch: `pocs/agent-tool-profiles`
- Commit: `acc1a5015`
- Build: Cozo-enabled Ninja build with four-way compilation; separate debug
  information disabled after PDB/ILK disk exhaustion was observed.
- CTest: agent 16/16 passed; Kubernetes 1/1 passed; Docker skipped.
- Smokes: runtime 11 passed, 1 Docker skip, 1 resident model-backed not run;
  MCP 7/7, CLI 2/2, daemon 7/7 and resource 1/1 passed.
- Overall model-free smoke result: 28/28 executed passed.
- Not run: resident model-backed execution, Linux build, Qwen/Nomic model runs,
  Docker backend execution and long-running stability.

### 2026-07-26 — Async lifecycle hardening checkpoint

- Commit: `722b3f645`
- Scope: inference lease ownership, deadline cancellation and terminal-entry
  cleanup outside the operation-manager mutex, monotonic terminal state
  transitions and failed-turn state cleanup.
- Focused smokes: operation-manager and session-manager async lifecycle
  regressions, including the poll/cancel race, passed.
- Agent CTest: 16/16 passed with Cozo enabled.

### 2026-07-26 - Multiple CLI text resources checkpoint

- Commit: `e57f5839d`
- Scope: repeatable `--resource PATH` imports bounded text files into the
  host-owned resource store and attaches multiple read-only input resources to
  the runtime request.
- Focused smoke: CLI/MCP smoke imported two text resources and verified their
  media types and resource references.
- Agent CTest: 16/16 passed with Cozo enabled.

### 2026-07-26 - CLI resource CTest checkpoint

- Commit: `88c6e4bb9`
- Scope: register the CLI/MCP smoke, including multi-file text-resource import,
  as an `agent`-labeled CTest instead of relying on manual smoke execution.
- Agent CTest: 17/17 passed with Cozo enabled.
- Deferred backlog: binary resources, PDF/document extraction, and byte-oriented
  resource transport remain out of scope for this text-only slice.
- The MCP agent-tools executable remained subject to a local Windows
  `LNK1104` output-lock failure during rebuild; it was not counted as a test
  pass or failure.

### 2026-07-26 - Resource listing and fetch checkpoint

- Commit: `10f0638fb`
- Scope: `daemon-session /resources` now renders bounded descriptors for each
  registered resource, while `/resource <uri>` remains the explicit scoped
  text read/fetch operation. Workspace files and sandbox files are not treated
  as downloadable resources unless a backend publishes resource references.
- Focused daemon-client smoke: passed, including resource listing and fetch.
- Agent CTest: 15/17 passed in the current build. `test-tool-adapters` still
  exits with MSVC `0xc0000409`, and `test-agent-data-store-cozo` cannot open
  its Cozo database (`code 14`); both failures reproduce independently of
  this change.

### 2026-07-26 - JSONL resource import checkpoint

- Commit: `ab5e728af`
- Scope: add the scoped `put_resource` JSONL command and the interactive
  `/resource-put <path>` admin-client wrapper for bounded text imports. The
  response returns the created descriptor through the resource response path.
- Focused protocol and client smokes: passed, including JSONL resource import,
  listing, and read-back coverage.
- Agent CTest: 19/19 passed with the repo-local `TEMP`/`TMP` build-test
  directory and four-way build.

### 2026-07-26 - Daemon CLI resource forwarding checkpoint

- Commit: `f3dcce0d2`
- Scope: daemon `--resource PATH` now imports through JSONL `put_resource`
  before `run_turn`, and forwards the returned URIs as scoped `resource_refs`.
  The daemon session path keeps imported resource references available for
  subsequent prompts.
- Focused daemon protocol and client CTests: 2/2 passed, including CLI resource
  forwarding and turn-request resource references.

### 2026-07-26 - Text-resource import semantics checkpoint

- Scope: validated all repeated CLI resource paths before any store write,
  normalized extension-based media types case-insensitively, kept ordinary
  references optional, and separated requiredness from explicit primary-source
  authority. Research sources no longer report a resource URI as a content hash.
- Reflective, deliberate, and research prompt paths already render the same
  bounded host-approved input-resource catalog; no mode-specific transport was
  added.
- Binary resources, UTF-8 validation, content digests, and document extraction
  remain backlog items for the text-only resource slice.

### 2026-07-27 - Tool execution hardening checkpoint

- Scope: made native and MCP call-slot reservation atomic after validation,
  enforced a shared MCP result-size limit over structured plus text content,
  executed normalized MCP arguments, and reduced repair skeletons to required
  schema properties with safe defaults/minimums.
- Deferred by design: hard interruption for synchronous native executors,
  interactive per-call confirmation, and stricter fuzzy resolution for
  policy-gated tools.

### 2026-07-26 — Lane-state ownership checkpoint

- Commit: `94cb3169d`
- Scope: pending-operation helper snapshots and mutates lane state under the
  lane mutex while keeping callbacks and host execution outside the lock.
- Focused smokes: operation-manager and session-manager runtime smokes passed.
- Agent CTest: 16/16 passed with Cozo enabled.

### 2026-07-21 — Windows Debug assurance run

- Branch: `feature/llama-agent`
- Commit: `6747c18f7`
- CTest: 11/11 agent tests passed after enabling `LLAMA_BUILD_TESTS=ON`.
- Smokes: runtime 8/8, MCP 7/7, daemon 7/7, resource 1/1.
- Overall model-free smoke result: 23/23 passed.
- Not run: resident model-backed smoke, Linux build, Qwen/Nomic model runs,
  long-running stability.

### 2026-07-26 - Event taxonomy checkpoint

- Commit: `ad7e68649`
- Scope: distinguish initial turn start, tool-driven turn resume, and
  manager-owned inference start/completion with explicit typed event kinds.
- Focused smokes: event-stream contract, operation-manager and
  session-manager runtime smokes passed.
- Agent CTest: 16/16 passed with Cozo enabled.

### 2026-07-27 - Cross-platform agent assurance checkpoint

- Branch: `pocs/agent-tool-profiles`, synchronized with the current
  `feature/llama-agent` upstream merge.
- Focused Windows/MSVC Debug build: completed for the registered agent CTest
  targets and sandbox targets using four parallel build workers. The focused
  Ninja invocation completed 188/188 build steps.
- Agent CTest: 19/19 passed with Cozo enabled. This includes the tool provider,
  tool runtime, CLI/MCP, daemon protocol/client, sandbox contract, and the
  model-free agent/runtime contract tests.
- Direct focused smokes: tool provider, tool runtime, sandbox contract, CLI/MCP
  run, daemon protocol, and daemon client passed.
- Docker sandbox: skipped with the declared return code because the Docker
  backend was unavailable. This is not counted as a Docker pass.
- Kubernetes sandbox: the contract CTest passed, while the live Kubernetes
  backend remained skipped because `LLAMA_AGENT_KUBERNETES_SMOKE=1` and a
  configured cluster were not supplied.
- Local Windows configuration confirmed that the CANN option and SOC/ACL graph
  arguments are accepted, but the CANN configure step cannot proceed without
  `ASCEND_TOOLKIT_HOME` or `CANN_INSTALL_DIR`. CANN hardware/runtime coverage
  remains not run.
- The local configuration did not provide OpenSSL, so HTTPS support was
  disabled in this Windows build. HTTP/MCP contract coverage therefore does not
  claim an OpenSSL-enabled HTTPS runtime pass.
- The broad all-agent target sweep was intentionally stopped after it exposed
  redundant per-smoke compilation of shared agent sources. The focused CTest
  targets were then rebuilt and passed; the duplicate-source build shape is a
  follow-up build-system optimization, not a test failure.
- Remaining unregistered MCP, daemon, resource, deliberate, research, and
  live-backend smoke executables were not rebuilt or counted in this checkpoint.

### 2026-07-27 - Agent library build-structure checkpoint

- Scope: extracted the shared `tools/agent` implementation used by the agent
  CLI, daemon, resource, runtime, and MCP-client targets into the reusable
  `llama-agent-runtime-support` library. Agent-enabled `pocs/memory` builds now
  link that library instead of compiling a second copy of the runtime source
  list.
- The library follows the repository `BUILD_SHARED_LIBS` policy, uses explicit
  target-scoped dependencies, and carries the existing Cozo link contract when
  `LLAMA_MEMORY_COZO=ON`.
- Focused agent build: 106/106 steps passed with four build workers after the
  shared library's first Cozo compile was rerun serially to avoid the known
  MSVC object-file lock on this Windows workspace.
- Memory POC target: 7/7 steps passed with four build workers.
- Agent CTest: 19/19 passed with Cozo enabled.
- Compile database inspection shows core runtime sources compiling once per
  configuration; remaining repeated sources are explicit MCP-server and daemon
  fixtures reserved for the next library split.

### 2026-07-27 - MCP server library checkpoint

- Scope: extracted the shared MCP server implementation into
  `llama-agent-mcp-server` without changing daemon or host-config ownership.
  MCP server smokes and the daemon executable now link the library instead of
  compiling the same server implementation sources independently.
- Focused MCP/CLI/daemon build: 23/23 changed build steps completed with four
  build workers.
- Agent CTest: 19/19 passed with Cozo enabled.
- Direct MCP provider, stdio client, and HTTP client smokes passed. The direct
  stdio-server integration smoke was stopped after it produced no output for
  several minutes; it is not registered under the `agent` CTest label and is
  not counted as passed in this checkpoint.
- Compile database inspection shows each extracted MCP server implementation
  source compiling once per configuration.

### 2026-07-27 - Daemon protocol and Cozo storage library checkpoint

- Scope: extracted the reusable daemon JSONL protocol implementation into
  `llama-agent-daemon-protocol` and the Cozo data-store family into
  `llama-agent-storage-cozo`. The daemon protocol library is static so the
  temporary fake-daemon client smoke can stage its executable without DLL
  deployment. Cozo storage remains conditional on `LLAMA_MEMORY_COZO`.
- Focused Windows/MSVC Debug build: 91/91 steps passed with four build
  workers. This included the runtime-support library, daemon JSONL protocol
  smokes, daemon client fixture, Cozo data-store CTest, and Cozo seed tool.
- Agent CTest: first run exposed a stale daemon-client smoke binary because it
  was not part of the focused target build. After rebuilding that consumer,
  the daemon-client CTest passed; the complete 19-test agent label was then
  rerun and passed.
- Compile database inspection shows the extracted daemon JSONL and Cozo data
  sources compile once per configuration. The test and seed consumers now
  link the Cozo library instead of compiling a second copy of its sources.

### 2026-07-27 - MCP stdio bounded-read regression checkpoint

- Scope: removed the remaining blocking `FILE*` stdout-read fallback from the
  MCP stdio client. The async/no-wait subprocess configuration now uses the
  incremental `subprocess_read_stdout()` path for both bounded and unbounded
  request reads. Windows binary-mode setup remains inside the `_WIN32` guard;
  framing and timeout behavior are platform-neutral.
- Build: `llama-agent-mcp-stdio-client-smoke` rebuilt successfully with four
  workers in the Cozo-enabled Windows/MSVC Debug tree.
- Focused checks: direct stdio client smoke passed; CLI/MCP and stdio client
  CTests passed 2/2.
- Agent CTest: 20/20 passed. The new
  `llama-agent-mcp-stdio-client-ctest` covers normal framed reads, malformed
  server diagnostics, and termination of a hanging server within its request
  deadline.
- Registered inventory: 64 total CTest cases, with 20 under the `agent` label.
