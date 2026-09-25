# FlyDelta session bootstrap

Use this file when starting a new development session for FlyDelta. The
current branch implementation is authoritative. Older prompts, plans,
conversation summaries and documentation are context only and must be
rechecked against the current callers and tests.

## Mission

Work on the host-controlled FlyDelta sideband without weakening these
boundaries:

```text
generation != evidence
evidence != learning update
learning update != promotion
promotion != activation
```

FlyDelta algorithms own proposals and bounded search policy. The model host
owns physical execution. The host verifier owns behavioral truth. The daemon
owns references, persistence and scheduling. The registry/controller owns
lifecycle transitions. The base GGUF is never edited.

## Branch and commit discipline

Work on the currently selected development branch. This workstream targets
the active workspace branch; do not switch to or synchronize with
`feature/llama-agent` unless the project owner explicitly requests that
integration step. Preserve unrelated dirty files and artifacts.

Group related changes into several coherent sweeps. After the grouped sweeps
have passed their relevant verification and the documentation gate below,
create a local commit on the current work branch. Do not push, merge, rebase
or integrate into `feature/llama-agent` as part of normal sweep completion.
The project owner decides when the integration branch becomes the active work
target.

## Long-running build and poll routine

When a build is likely to take more than a few minutes, do not busy-wait in
the foreground. Start it with the project-approved elevated build environment
and writable cache configuration, wait briefly, then inspect actual progress
and estimate the remaining time. If completion is still several minutes away,
schedule one one-shot poll for the estimated completion time plus a small
measured margin. A fixed delay such as twenty minutes is not the default; it is
only appropriate when the measured estimate supports it. Re-check progress
when the estimate changes and replace the poll rather than leaving a stale
one. The poll must
carry the next concrete actions: inspect the final build result, run the
relevant tests serially, inspect tracing, perform the correction pass and
continue toward the local commit.

Keep this routine portable: repository documentation must not contain a local
home directory, machine-specific hardware, or a hard-coded cache path. The
environment supplies those values. Compilation may use the configured build
thread count; GPU/CPU functional tests remain serial and must use the full
agent/server-context flow. Do not create duplicate polls when one is already
active; update or reuse the existing one.

## Read before proposing changes

Read these files in order:

```text
docs/agent/agent-flydelta.md
docs/agent/agent-model-adaptation.md
docs/agent/README.md
docs/agent/agent-daemon-usage.md
docs/examples/agent-host-config-flydelta-full.json
docs/examples/agent-flydelta-dataset-question-suite.json
docs/examples/agent-flydelta-dataset-question-suite-incremental.json
```

Then inspect the current implementation, production daemon callers and
relevant tests/smokes. Start with the FlyDelta evaluator, worker, job,
candidate-lifecycle, promotion, review-store, sideband-controller and daemon
runtime/protocol files.

## Required investigation method

For every claimed gap, trace the complete chain:

```text
contract
  -> implementation
  -> registration
  -> production invocation
  -> durable persistence
  -> consumer or lifecycle decision
```

Also check the worker, daemon, test and smoke callers. A type, validator,
callback, capability bit, unit test or documentation statement is not proof
that a production capability is complete.

Treat verification evidence by type:

```text
CTest
  -> contract, validation and invariant evidence

model-free smoke
  -> functional daemon/queue/worker/persistence wiring without a model claim

model smoke
  -> functional model-host, batch, generation and host-verification behavior
```

Every functional smoke must emit tracing sufficient to follow the bounded
flow from job/request through worker and host callback to durable result and,
where applicable, `next_action`, review or lifecycle consumption. A passing
exit code without that trace is not enough to claim functional wiring. Model-
free and model-backed smokes must remain clearly named and must not be used as
substitutes for one another.

Before implementation, produce a short status map using only:

```text
IMPLEMENTED
PARTIAL
MISSING
LEGACY/REPLACED
```

The map must include exact entrypoints, production callers, durable reports or
state, consumers/decisions, and tests/smokes. Do not create a new host,
runtime, queue, evaluator, registry or GPU path when an existing seam can be
extended. Do not perform cleanup unless it is the first real blocker.

## Current pre-canary focus

The intended chain is:

```text
candidate artifact
  -> counterfactual reports -> PromotionSummary
  -> EvaluationRunner -> EvaluationReport
  -> durable explicit review
  -> stage_canary
```

These are parallel evidence gates: PromotionSummary is derived from trusted
counterfactual reports, while EvaluationReport comes from bounded intended,
holdout, retention and agent-regression fixtures. Evaluation does not equal
review. Review is authority. Canary is not active.

Keep these invariants visible:

```text
UNKNOWN/NEUTRAL != HELPED
margin != host truth
search_rank != evidence_rank
augmentation != evidence
concept synthesis != evidence
graft != promotion
artifact creation != activation
candidate != canary
canary != active
```

The current stage-canary daemon path is `IMPLEMENTED`: the daemon uses the
existing atomic review-store seam, and
`llama-agent-daemon-flydelta-admin-smoke` covers the model-free admin sequence
through traced queue, worker, persistence, review and canary replay.

## End-of-sweep documentation gate

At the end of each grouped development sweep, before calling the sweep done:

1. Compare changed contracts, callers, persistence and consumers with
   `docs/agent/agent-flydelta.md` and the daemon guide.
2. Update status labels, entrypoints, configs, fixtures, commands and known
   limitations when the implementation changed.
3. Re-check that every documented production link is actually registered,
   invoked, persisted and consumed.
4. Run relevant CTests for contracts, then the functional model-free and/or
   model-backed smokes. Inspect their traces, not only their exit codes.
5. Record the verification date, commit, test category and traces in the FlyDelta maintenance
   log. If no documentation change is needed, record that review and why.
6. When the grouped sweeps are ready to checkpoint, create one local commit on
   the current work branch. Keep branch synchronization as a separate,
   explicitly requested operation.

Do this once at the end of a related sweep, not after every micro-edit. Never
leave a documentation claim stronger than the evidence produced by the
sweep.

## Recurring clean-architecture gate

Do a separate clean-architecture pass when an area is being tied together,
when several seams have crossed from contract work into production wiring, or
when a feature is approaching an evidence-level checkpoint. Do not run this
after every contiguous sweep by default. Before starting the pass, record a
short decision: `defer` when the work is still local and the chain has not
changed, or `run` when ownership, scope, persistence, concurrency or lifecycle
boundaries may have shifted.

When it is `run`, re-trace the completed area as:

```text
contract -> implementation -> registration -> invocation
         -> durable persistence -> consumer/lifecycle decision
```

Check specifically for duplicate orchestration, authority derived from model
output, fixed local-environment assumptions, queue state standing in for
durable evidence, and a next action that can advance after persistence has
failed. Correct findings in the existing seam, update the status map and
maintenance log, then perform the normal CTest/model-free/model-smoke gate.
The next pass is needs-based again; it is not automatically required for the
next micro-sweep.
