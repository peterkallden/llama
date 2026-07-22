# Agent Beta Assurance

Status: Verified for the executed Windows/Debug beta scope

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
- [ ] Developer workspace tools are implemented

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

## Verification record

Each record must identify exactly what was run. Counts use `passed/total`;
skipped and unavailable tests are recorded separately rather than counted as
passes.

### Current run

| Field | Value |
|---|---|
| Branch | `feature/llama-agent` |
| Commit | `6747c18f7` |
| Date | `2026-07-21` |
| Platform | Windows / MSVC |
| Build configuration | Debug |
| CTest | 11/11 agent tests passed |
| Agent smokes | 23/23 model-free smokes passed |
| Decision | Conditional beta assurance |

### CTest evidence

| Batch | Passed | Failed | Skipped | Total | Result |
|---|---:|---:|---:|---:|---|
| Agent contracts/runtime (`test-agent-*`) | 4 | 0 | 0 | 4 | Passed |
| Memory (`test-memory-*`) | 2 | 0 | 0 | 2 | Passed |
| Plan (`test-plan-*`) | 2 | 0 | 0 | 2 | Passed |
| Reflection/tooling (`test-reflection-*`, `test-tool-*`) | 3 | 0 | 0 | 3 | Passed |
| **Agent CTest total** | **11** | **0** | **0** | **11** | **Passed** |

Commands and full output should be retained in the task handoff or CI log;
this file records the summarized result and the commit it belongs to.

### Smoke evidence

| Group | Passed | Failed | Total | Result |
|---|---:|---:|---:|---|
| Runtime | 8 | 0 | 8 | Passed |
| MCP | 7 | 0 | 7 | Passed |
| Daemon | 7 | 0 | 7 | Passed |
| Resource | 1 | 0 | 1 | Passed |

## Known limitations

- Linux and model-backed Qwen/Nomic validation require their respective
  environments and are not implied by a Windows smoke run.
- The recorded model-free smoke groups are the executable groupings used by
  the current `pocs/agent/CMakeLists.txt`; there is no separate executable
  called “beta aggregate” in this assurance record.
- The resident model-backed smoke was built but not executed in this run
  because it requires an explicit `--model` GGUF path.
- Developer workspace tools are outside the current minimal beta scope.
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
session. The Docker sandbox backend is covered by a real container smoke; the
Kubernetes backend and broader network scopes remain future scope. Developer
and data-analysis execution classes are represented by host configuration, but
full operation-manager integration remains a separate verification gate.

## Exit decision

The milestone can be marked `Passed` only when all required gates are either
verified or explicitly accepted as out of scope, no open P0/P1 defect remains,
and the verification record identifies the exact branch, commit, platform,
date, and test counts.

### Decision record

- Decision: Conditional beta assurance for Windows/Debug model-free scope
- Date: 2026-07-21
- Commit: `6747c18f7`
- Reviewer: pending
- Notes: 11/11 agent CTests and 23/23 model-free agent smokes passed. Linux,
  model-backed Qwen/Nomic, developer workspace tools, checkpointing and
  long-running stability remain outside this run.

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
