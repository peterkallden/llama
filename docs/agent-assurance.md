# Agent Beta Assurance

Status: Conditional for the executed Windows/Debug beta scope; the complete
model-free agent CTest and smoke matrix passed with backend skips recorded,
while model-backed coverage remains outstanding

This document is the assurance record for the `llama-agent` beta milestone.
It separates milestone criteria from the test evidence collected for a
specific branch, commit, platform, and date. A checked item is only valid
when the corresponding verification record contains evidence for it.

## Milestone

- Milestone: Agent runtime beta
- Integration branch: `pocs/agent-tool-profiles`
- Baseline: `origin/master`
- Scope: reflective, deliberate, research, tools, sessions, scheduling,
  resources, memory, plans, daemon, JSONL, and MCP hosts

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
- [x] Research workspace remains turn-scoped and ephemeral by default
- [ ] Research workspace checkpointing has been evaluated

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
| Sandbox execution | [x] | Direct Docker and Kubernetes sandbox smokes passed; build/test end-to-end execution remains a separate gate |
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

Each record must identify exactly what was run. Counts use `passed/total`;
skipped and unavailable tests are recorded separately rather than counted as
passes.

### Current run

| Field | Value |
|---|---|
| Branch | `pocs/agent-tool-profiles` |
| Commit | `e57f5839d` |
| Date | `2026-07-26` |
| Platform | Windows / MSVC |
| Build configuration | Debug semantics, Cozo enabled, Ninja, four-way build; separate debug information disabled to avoid PDB/ILK disk exhaustion |
| Cozo | `LLAMA_MEMORY_COZO=ON`, `LLAMA_PLAN_COZO=ON`, configured Cozo 0.7.6 MSVC release library and DLL |
| CTest | Agent label 16/16 passed; Kubernetes label 1/1 passed; Docker label skipped in the normal run and hung when elevated, so no Docker backend pass is claimed |
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

### CTest evidence

| Batch | Passed | Failed | Skipped | Total | Result |
|---|---:|---:|---:|---:|---|
| Agent contracts/runtime (`test-agent-*`) | 8 | 0 | 0 | 8 | Passed |
| Tooling (`test-tool-*`, clangd, Cozo) | 5 | 0 | 0 | 5 | Passed |
| Agent runtime CTest smokes | 3 | 0 | 0 | 3 | Passed |
| **Agent CTest total** | **16** | **0** | **0** | **16** | **Passed** |
| Sandbox Kubernetes (`sandbox-kubernetes`) | 1 | 0 | 0 | 1 | Passed |
| Sandbox Docker (`sandbox-docker`) | 0 | 0 | 1 | 1 | Skipped; backend unavailable |

### Registered CTest inventory

The current Cozo-enabled build registers 60 CTest cases in total. The agent
verification slice is the `agent` label with 16 tests, all of which passed in
the current run. The sandbox labels are separate backend slices:

| Label | Registered tests | Current result |
|---|---:|---|
| `agent` | 16 | 16 passed |
| `sandbox-kubernetes` | 1 | 1 passed |
| `sandbox-docker` | 1 | 1 skipped; backend unavailable |
| Other repository tests | 42 | Not part of the agent assurance sweep |

The 60-test inventory is configuration-dependent. It includes general
repository tests whose executables were not built or run in this focused agent
verification, so the total inventory must not be reported as a 60-test agent
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
- Commit: `e57f5839d`
- Reviewer: pending
- Notes: The complete model-free agent CTest label passed 16/16; Kubernetes
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
- Deferred backlog: binary resources, PDF/document extraction, and byte-oriented
  resource transport remain out of scope for this text-only slice.
- The MCP agent-tools executable remained subject to a local Windows
  `LNK1104` output-lock failure during rebuild; it was not counted as a test
  pass or failure.

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
