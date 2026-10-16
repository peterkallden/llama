# Workflow grounding

Implementation status: implemented and partially verified. The validation
record below separates direct model-free tests and MCP tool checks from the
still-pending live workflow and model-smoke evidence. The earlier Eurostat
run22 trace predates this implementation and is not counted as validation.

## Validation record (2026-10-16)

- A CPU/Cozo build completed with `BUILD_TESTING=OFF`. The directly invoked
  binaries `test-agent-workflow-grounding`, `test-agent-bootstrap`,
  `test-agent-route-compiler`, `test-plan-cozo`,
  `test-agent-runtime-contract`, `test-plan-json`, and `test-plan-store` all
  passed. These were direct executable runs, not a registered CTest run.
- The connected Codex MCP tool-audit server exposed the expected working tree
  and commit `d7e310583`. Repository search/read and dataset inspection,
  schema, sample, query (projection without a predicate), and statistics calls
  succeeded. The inspected Eurostat materialization contained one observation:
  Sweden (`geo=SE`), 2020, `PER_KM2`, value `25.2`; grouped statistics returned
  the same value. This checks the dataset/MCP tools against existing materialized
  data, not execution of the Eurostat workflow graph.
- An OpenAlex MCP chain also succeeded: `openalex.listWorks` returned a bounded
  result materialized as a dataset, the selected work ID was passed to
  `openalex.getWork`, and the returned host resource was inspectable. This
  verifies tool-to-tool identifier use and resource publication, not automatic
  workflow transition binding.
- The attached MCP catalog did not expose `eurostat.*` operations or an agent
  session/workflow invocation endpoint. Consequently, it could not exercise
  route selection, `preferred_thinking_mode`, verification gates, continuation
  handoff, or declared-claim enforcement through the live workflow runtime.
  No Qwen model smoke was run. Do not treat the MCP tool checks above as
  end-to-end workflow or model validation.

## Execution preference

`preferred_thinking_mode` is optional workflow metadata, accepting `deliberate`
or `research`. It is considered for automatic thinking selection after route
selection, on resume and on a declared handoff. An explicit requested mode and
host escalation/review limits take precedence. The driver preserves an already
higher mode and does not expand tool or plan-revision budgets. The trace records
the preferred and resolved modes and denial reasons. Research remains subject
to its existing execution envelope and evidence-acquisition budgets.

Old packages without the field retain their previous behavior. Workflows with
verification or declared-claim contracts require package `schema_version: 2`;
the current importer accepts schema 1 and 2. Export chooses schema 2 when these
contracts are present, so an older schema-1-only importer rejects rather than
silently dropping mandatory gates. The historical starter filename is retained;
its package version is now v24.

## Verification contract

`verification_checks` is a workflow-level bounded list (maximum 32), with each
check attached to a stable tool transition through `transition`. The check has
an `id`, `phase`, `operation`, `left`, optional `right`, and `required` (default
true). `before_answer` conditions need no tool transition. Available operations
are `required`, `equal`, `member`, `count`, `status`, `dimension_membership`,
and `coverage`.

```json
{
  "preferred_thinking_mode": "deliberate",
  "require_declared_claims": true,
  "verification_checks": [
    {
      "id": "geography-defined",
      "transition": "request-observations",
      "phase": "before_tool",
      "operation": "required",
      "left": {"source": "arguments", "pointer": "/geo"},
      "clarification_question": "Which geographic area should the statistics cover?"
    }
  ]
}
```

Operands select typed JSON through RFC-6901 pointers. Sources are `arguments`
(materialized current arguments), `result` (the full current successful output),
`literal` (the `value` member), `observation` (successful named completed
transition), `transition_arguments` (host-retained actual arguments of that
completed transition), and `handoff` (the host snapshot from a declared link).
`observation` and `transition_arguments` require `transition`; they never search
an earlier workflow segment by a same-named transition. Handoff snapshots bind
the source fingerprint and target workflow/revision.

Setting `dataset_rows: true` resolves a dataset URI through the host's bounded
reader. It checks the current descriptor, resource read authority and source
fingerprint before reading all rows, with a 10,000-row bound. Derived datasets
retain the first parent's source provenance, and every parent is authorized
through the existing lineage (maximum depth 16 / 64 visited nodes). Partial results,
unavailable readers and unknown authority fail to `unknown`. Reads use the
existing data store and do not expose a new model tool or network operation.

Before-checks run after argument materialization and schema validation, before
execution. After-checks consume the full host output before the next workflow
transition; compact model observations do not define the verifier's input.
The successful tool observation remains stored if verification fails. Required
checks must produce `passed`; `failed` and `unknown` stop dependent progress.
A missing required argument can return its configured clarification question
and leave the step incomplete for repair on a later turn.

Proof observations record phase, check/step/revision, status, evidence references,
reason and an input fingerprint. They use existing plan persistence. Unchanged
revalidation results are not stored repeatedly. Completed guarded steps are
rechecked against current evidence before synthesis, including on resume.
Only host-authorized operations may record grounding evidence or complete a
guarded step; reflection cannot bypass a failed gate by naming a successful
transport observation or replacing its transition identity.

## Eurostat and dataset interpretation

The SDMX normalizer's rows retain constraint regions, include/exclude flags,
dimension codes and constraint provenance. `dimension_membership` checks tuples
against these grouped regions; it does not merge them into independent global
code lists. Excluded regions are respected. Unsupported key exclusions,
ranges or varying unbound dimensions remain unknown. At most 256 requested
tuples are expanded. This is bounded verification, not a general SDMX solver.

`dimension_mapping` maps argument/result field names to normalized dimension
names. Eurostat uses `geo -> geo` and `time -> TIME_PERIOD` before retrieval.
After JSON-stat materialization, coverage compares result `geo`/`time` against
the retained request, checks complete requested tuples and non-null values,
then checks returned `unit`, `freq`, `geo` and `time` against SDMX constraints.
Status policy in the starter permits only absent/null or empty flags.

Regional titles do not exclude national aggregates. The saved run22 response
from `demo_r_d3dens` contains SE/2020/PER_KM2/A and 25.2; its title alone was
not grounds for rejecting it. Catalogue discovery still narrows candidates
by concept before metadata inspection. Metadata verifies code support, not
whether an inferred indicator definition exactly matches a user's intent.

## Declared numerical claims

`require_declared_claims` makes the existing draft generation return an answer
plus 1..16 declarations. The requirement is retained in host plan observations
through subsequent declared analysis handoffs. No additional extraction call
is introduced. The selected model-facing DSL/JSON format is used.

The host exposes short evidence handles (`E1`, etc.), values, units, dimensions
and status. Full dataset/resource references and fingerprints remain on the
host. A maximum of 16 handles and 8 KiB of host handle data are prepared.

```text
answer: Sweden had 25.2 persons per square kilometre in 2020.
claim: E1
value: 25.2
unit: Persons per square kilometre
dimension: freq A
dimension: geo SE
dimension: time 2020
```

For JSON the equivalent object has `answer` and `claims`, where each claim
contains `handle`, `value`, `unit`, and a `dimensions` object. Repeat `answer`
lines for a multiline DSL answer and repeat `claim` blocks for multiple claims.
Dimension names or codes containing spaces may be quoted.
The host verifies exact numeric value, unit, dimension codes and usable status
against the referenced cell. It retains dataset URI, cell location, source
resource and fingerprint in existing plan/research evidence. Declared claims
populate the existing research verifier's claim context; deliberate uses the
same numeric check without starting research.

V1 handles require a materialized numeric `value` column with explicit `unit`
or `unit_label`. A transformed result must retain those and the required
dimensions to be admitted. Missing measurement provenance or flagged values
remain unsupported rather than deriving units or calculations from model text.
Draft revision uses the existing bounded iteration/reflection budgets.
Only the declarations are checked: arbitrary claims in prose are not thereby
certified, and a correctly quoted data point does not prove a causal explanation.

## Remaining end-to-end validation

The direct model-free test executables listed above pass, but they do not replace
a daemon-level workflow run. Still to verify through a workflow-capable runtime
are route/mode selection and resume, missing-argument clarification, a successful
tool result followed by a failed required postcondition, stale-source
revalidation, guarded plan mutation, continuation/budget exhaustion, and declared
claims across a handoff. The current MCP tool-audit connection exposes tools but
not the agent workflow invocation surface, so it cannot perform those checks.

The next full validation should use the daemon's actual workflow entrypoint and
full trace, then run Qwen with GPU auto-fit. Include supported SE/2020 data and
negative cases for absent codes, wrong unit/status, and unsupported declarations.
Record model input/output, host verification observations, route/mode changes,
handoffs, and final answer separately. Until that run completes, workflow-level
model behavior remains unverified; the direct tests and MCP tool checks above
must not be summarized as model E2E success.
