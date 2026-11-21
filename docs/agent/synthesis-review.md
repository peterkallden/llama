# Synthesis, bounded completion and response review

Research remains the owner of its workspace, evidence assessment, coverage,
stop reason and `partial_answer_available`. The shared synthesis contract is a
projection of those existing facts plus plan/workflow execution obligations.
It is not a second research controller or another evidence scorer. Deliberate
and ordinary planning use the same projection without depending on a research
workspace. Providers, inference backends and source-selection semantics do not
change.

## Host completion contract

`common_agent_synthesis_state` separates work completion from evidence quality.
Work can be pending, complete or bounded-complete. Evidence can be unknown,
sufficient, partial or insufficient. The host additionally supplies
`mandatory_remaining`, `partial_allowed`, the original research stop reason,
the continuation resolution reason, plan identity/version and evidence
fingerprint. Synthesis and final review receive the same contract as mandatory
context under the existing context allocator; optional observations give way
first if context is limited.

Continuation resolution distinguishes `transitioned`, `not_applicable` and
`failed`. Only actual transitions consume the existing global transition
budget. A host-only `workflow_continuation_resolution` observation retains why
no further route is applicable, scoped to the workflow revision, route segment,
terminal state and workflow evidence fingerprint. For example, the declared
alternative may have reached `max_uses`, or its candidate frontier may be
unavailable. New source evidence invalidates that closure. Restored terminal
plans without a current closure return to the existing resolver before
synthesis. An applicable mandatory continuation blocked by global budget still
fails; it is not converted into permission to answer.

A bounded partial answer is permissible only after mandatory work is closed
and the existing research evidence policy permits partial synthesis. Ordinary
workflow execution can also retain a declared unavailable terminal as a
bounded disposition without assigning it a research coverage score. Lack of
full text must be disclosed; it is not by itself grounds to reject a correctly
qualified partial answer. Missing URLs, failed requests and unread full text
remain different facts.

## Review checklist

The existing review verdict and assurance action remain authoritative. JSON
reviews may additionally provide up to four `issues`, four `advisory_notes`
and four `resolved_issue_ids`. Each issue has a description and correction,
optional host-known evidence IDs and an optional `previous_issue_id` when
retaining an earlier blocker. The host assigns new issue identities and checks
bounds, references and consistency; the model judges whether the defect was
resolved. No wording dictionary or textual similarity heuristic performs that
judgment.

Every next review receives the previous checklist, the complete revised draft,
the current evidence and the completion contract. Structured reviews must
account for every earlier blocker as resolved or retained. Acceptance cannot
retain blockers; response revision must have a blocking correction, rather than
only advisory style preferences. Contract errors use the existing bounded
regeneration path. The host never changes a rejected verdict into acceptance.

The compact review DSL retains `guidance` and adds:

```text
reflect revise
assurance_action revise_response
issue description="Missing limitation" correction="Disclose unread full text" previous_issue_id="review-1-1"
advisory "A shorter heading would also work"
```

An accepting structured DSL review can use `resolved "review-1-1"`. Older JSON
and DSL reviews using only `revision_guidance` remain supported; rejected
guidance becomes a conservative blocking checklist. An older accepting verdict
remains a semantic acceptance. Review history and its draft/evidence
fingerprints are stored with existing `response_review` observations, so restart
does not discard the unresolved checklist. Changed evidence is explicitly
available for reassessment. No budget is increased.

## Unapproved draft delivery

After response review stops with a complete draft, runtime can return a separate
optional `unverified_draft`. It contains the text, `review_status: unverified`,
stop reason, plan reference, draft/evidence fingerprints and unresolved issues.
The ordinary `response` remains empty, the turn remains unsuccessful and the
plan is not completed. Cancellation, transport errors and incomplete generated
text do not automatically become review drafts. Legacy clients can ignore the
optional field and retain their existing failure behavior.

The session host retains display-only draft artifacts in the existing scoped
resource store with purpose `agent.context.review_draft.v1`. Conversation
loading and compaction do not read that purpose. Chat history displays it with
a permanent **Ej godkänt utkast** warning and outstanding defects, including
after reload. This is different from an accepted partial report. An accepted
response supersedes an older displayed draft for the same turn. A terminal SSE
failure notification does not abort the command response before its draft
payload can arrive; cancellation still cancels the request.

## Verification

Contract coverage is in `test-agent-synthesis-state`, `test-reflection-json`,
`test-agent-context-management`, the workflow recovery contracts and web session
tests. It covers pending mandatory work, stale closure, checklist identity and
resolution, legacy responses and draft exclusion from model history.

The standalone model-free `llama-agent-synthesis-review-smoke` runs deterministic
reviewers through the real runtime. It exercises a corrected accepted partial
report and exhausted review with a separate unapproved draft, and prints runtime
traces. It needs no model or GPU.

For live validation, restart the configured daemon/web launcher with the Luna
HTTP profile and tracing enabled, then create a new chat. Ask OpenAlex for urban
green-space/mental-health papers, request full text and another candidate if the
first is blocked or has no URL. Verify two distinct host-observed lookup IDs,
the exact access limitations, bounded continuation closure and the shared
completion contract in synthesis/review traces. Record separately whether the
result is an accepted partial report or an unapproved draft. The latter must
remain visibly marked after chat reload. A displayed draft proves delivery,
not that model review succeeded. Live PDF access may remain blocked; successful
full-text analysis is a separate verification case.

No compilation, contract execution or model verification is implied by the
presence of these fixtures. Build/test authorization remains separate.
