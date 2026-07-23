# Agent Beta Assurance

Status: Conditional for the executed Windows/Debug beta scope; the latest
focused tool-name and sandbox verification passed, while the complete smoke
matrix and model-backed resident run remain outstanding

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
| Tool-specific smoke coverage | [ ] | The focused migration smoke set passed; the complete smoke executable set and model-backed resident coverage were not rerun |

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
| Commit | `d55ff9d7b` |
| Date | `2026-07-23` |
| Platform | Windows / MSVC |
| Build configuration | Debug |
| CTest | 10/11 passed, 1 skipped; the Docker CTest was skipped by the CTest environment, while the Kubernetes CTest passed |
| Focused smokes | 8/8 passed: affected tool-name smokes plus direct Docker and Kubernetes backend smokes |
| Decision | Conditional assurance; follow-up required |

### CTest evidence

| Batch | Passed | Failed | Skipped | Total | Result |
|---|---:|---:|---:|---:|---|
| Agent contracts/runtime (`test-agent-*`) | 4 | 0 | 0 | 4 | Passed |
| Tooling (`test-tool-*`, clangd, Cozo) | 4 | 0 | 0 | 4 | Passed |
| Sandbox (`llama-agent-sandbox-*-ctest`) | 2 | 0 | 1 | 3 | Conditional; Docker skipped |
| **Focused agent CTest total** | **10** | **0** | **1** | **11** | **Conditional** |

Commands and full output should be retained in the task handoff or CI log;
this file records the summarized result and the commit it belongs to.

### Smoke evidence

| Group | Passed | Failed | Total | Result |
|---|---:|---:|---:|---|
| Affected research/resource/MCP/daemon smokes | 6 | 0 | 6 | Passed |
| Docker backend smoke | 1 | 0 | 1 | Passed |
| Kubernetes backend smoke | 1 | 0 | 1 | Passed |
| **Focused smoke set** | **8** | **0** | **8** | Passed |
| Complete smoke executable set | — | — | 30 | Not rerun |

## Known limitations

- Linux and model-backed Qwen/Nomic validation require their respective
  environments and are not implied by a Windows smoke run.
- The recorded model-free smoke groups are the executable groupings used by
  the current `pocs/agent/CMakeLists.txt`; there is no separate executable
  called “beta aggregate” in this assurance record.
- The resident model-backed smoke was not executed in this focused run because
  it requires an explicit `--model` GGUF path.
- Developer workspace tools are now in scope and have bounded adapter coverage;
  semantic indexing, end-to-end build/test execution and the complete smoke
  matrix remain separate gates.
- The direct Docker and Kubernetes backend smokes passed. CTest still reports
  Docker as skipped when the test process cannot access the Docker daemon.
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
including symbol/reference fallback and grouped test-failure diagnostics. The
direct Docker and Kubernetes sandbox smokes passed; CTest Docker remains
environment-skipped when daemon access is unavailable. Data-analysis execution classes are
represented by host configuration, while full operation-manager integration and
semantic project indexing remain separate verification gates.

## Exit decision

The milestone can be marked `Passed` only when all required gates are either
verified or explicitly accepted as out of scope, no open P0/P1 defect remains,
and the verification record identifies the exact branch, commit, platform,
date, and test counts.

### Decision record

- Decision: Conditional beta assurance for Windows/Debug model-free scope
- Date: 2026-07-23
- Commit: `d55ff9d7b`
- Reviewer: pending
- Notes: The focused verification for the canonical namespaced repository tools
  passed: 10/11 focused CTests passed with one Docker skip, and 8/8 affected
  smokes passed including direct Docker and Kubernetes backend execution. The
  complete smoke matrix, Linux, model-backed Qwen/Nomic execution, semantic
  indexing, checkpointing and long-running stability remain outside this run.

## Assurance history

Add a dated entry for every meaningful verification run. Do not overwrite a
previous result when the branch or test configuration changes.

### 2026-07-21 — Windows Debug assurance run

- Branch: `feature/llama-agent`
- Commit: `6747c18f7`
- CTest: 11/11 agent tests passed after enabling `LLAMA_BUILD_TESTS=ON`.
- Smokes: runtime 8/8, MCP 7/7, daemon 7/7, resource 1/1.
- Overall model-free smoke result: 23/23 passed.
- Not run: resident model-backed smoke, Linux build, Qwen/Nomic model runs,
  long-running stability.
