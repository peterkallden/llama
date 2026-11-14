# Prior-work continuation

A paused turn can yield a successor plan through the normal model plan selector.
The predecessor remains in the existing plan store. A successor owns its own
turn identity, execution authority and budgets.

## Checkpoint and method identity

`prior-work.v2` references the predecessor ID, version and evidence fingerprint.
Method metadata references its saved outline and last phase review. Canonical
`origin_phase_id` values survive blueprint instantiation; instance IDs are not
parsed to infer their origin. Older v1 checkpoints remain readable as historical
context, without automatic phase transfer.

After the model chooses prior work, an identical eligible phased-method route
is retained from the current authorized route catalog. An unavailable method or
changed revision requires ordinary route selection and prevents automatic phase
inheritance. This rule does not authorize an old workflow or tool envelope.

The runtime restores compatible `satisfied` and `not_needed` decisions before
planning the first successor slice. The planner receives the unresolved phase
frontier. Blocked phases require a new assessment. Reopening a prerequisite
invalidates dependent completion. Unsupported or changed source evidence cannot
silently close a phase.

## Evidence and current validity

The shared lineage resolver verifies namespace, project, session, source plan
version and evidence fingerprint. It rejects cycles, missing references and
lineages deeper than 32 plans. Evidence references identify their actual owning
plan, version and observation, including across repeated continuations.

Planning, reasoning, final drafting and reflection use a transient read view of
validated predecessor reads. Original completed-step provenance accompanies the
observations for existing context and review contracts. The view is never
scheduled or persisted as a successor execution journal. Current-turn evidence
retains presentation priority. Model operations still apply to the real plan.

`prior_evidence_is_current` is a host-only callback. The first implementation
supports repository/workspace text reads with an intact saved SHA-256. It checks
canonical paths within the approved repository, regular-file identity, the
64 MiB source bound and current content hash. It follows no path outside that
root. Journal integrity and current content validity are separate checks.

Other observation types remain historical unless a host validity implementation
is supplied. In particular, a saved Git log, external search or runtime snapshot
is not automatically certified current. The reviewer may inspect or reacquire
missing evidence through ordinary authorized planning. No source-specific
semantic routing or keyword table is introduced.

Inherited source references and phase decisions remain ordinary observations in
the existing SQLite/Cozo plan codecs. On re-entry the runtime resolves references
again and reopens inherited decisions when their source validity changes.
Reserved checkpoint, outline and phase-review observations require host-authorized
record operations. They cannot be manufactured by a model record operation.

## Review continuity

A successor restores unresolved checklist items with their original issue IDs.
Previously reviewed excerpts and their approval fingerprints are not inherited;
evidence attachments on open issues are cleared until the new review grounds
its resolution in the current evidence view. This keeps historical needs
addressable without treating prior approval as approval of new text.

Reflection uses short host-issued phase handles. Exact predecessor phase IDs
can also be mapped through canonical phase keys from a compatible validated
checkpoint. Truncated or invented IDs are not guessed. A downstream verdict
cannot close before its dependencies; such a verdict remains provisional while
its tool evidence is retained for the next assessment.

## Verification

The following commands are prepared, not evidence of a passed build or test:

```sh
ctest --test-dir build-agent-sanitize --output-on-failure \
  -R '^(test-agent-method-phases|test-plan-context|test-agent-flexible-workflow|test-plan-sqlite-continuation|test-plan-cozo)$'
build-agent-sanitize/bin/llama-agent-prior-work-smoke
```

The model-free smoke uses the real runtime, normal plan store and deterministic
planner/reviewer stubs. It checks that completed inspection leads to analysis,
that inherited source evidence reaches the planner and reviewer, and that no
completed source read is executed again. It always prints runtime trace entries.
SQLite and Cozo contracts additionally round-trip checkpoint references and
resolve an ancestor through a second successor after reopening storage.

Use the Luna UI scenarios in `pocs/agent/smoke/comprehension/luna-ui-scenarios.md`
for live verification. Successful source-code contracts do not prove that Luna
will produce a substantive graph description. Keep that result separate.

Default budget values are unchanged. The daemon must allocate enough execution
rounds for both its configured reflection rounds and `1 + max_plan_revisions`,
including reflective mode. Measure phase transitions, repairs and repeated source
reads after verifying reuse before changing the default development budgets.

For phased methods, host phase and response reviews use the existing bounded
control-observation allowance rather than consuming source-evidence slots.
They remain host-authorized records. Continuation planning receives the latest
reasoning separately before the prefix-bounded journal projection, so a growing
journal cannot hide the model's most recent unresolved work. Quoted values in
tool projections are literal stored values; explicit truncation markers describe
omitted coverage. An inadequate saved proposal can require a new sourced
correction rather than repeated reads of the same unchanged result.
