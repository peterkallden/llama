# Agent workflows

This page describes runtime workflows (not GitHub Actions workflows).

## Contracts

- A **blueprint** describes a task goal, constraints, assumptions and success
  criteria. It may be useful without a bound executable workflow.
- A **workflow** describes a host-approved method and its execution authority.
  Definitions are stored as plan-store records and referenced by immutable
  workflow identity and revision.
- A **procedure** is reusable methodology or guidance; it does not itself grant
  tool access or define an executable path.
- A **domain adapter** binds a workflow to request-specific host state and may
  materialize a task plan. Dataset inventory selection is the first such
  adapter; other domains should add adapters without adding domain branches to
  route compilation.

Workflow definitions declare semantic `required_capabilities`, optional
`optional_capabilities`, and host-state `required_context`. Required
capabilities gate route eligibility. Optional capabilities are included only
when the host can resolve them in the active tool view; those resolved tools
are part of the route envelope and its fingerprint from route selection.
Context requirements are checked separately and do not grant tool authority.

`allowed_tools` remains accepted for v1 package compatibility. When semantic
capabilities are also present, `allowed_tools` can only narrow the resolved
tool set. New workflow packages should prefer capabilities.

## Current implementation boundary

The route compiler evaluates workflow capabilities and host-context
requirements generically. The starter library covers these host-tool domains:

- Dataset analysis and quality, including `data.query`, `data.filter`,
  `data.aggregate`, `data.transform`, and the descriptive/statistics tools.
- Multi-dataset joins through a separate route that is eligible only when the
  host reports at least two scoped datasets. Join inputs must resolve to two
  distinct host dataset identities.
- Repository inspection, with read-only search/read capabilities.
- Repository change and regression repair, with workspace patching plus
  sandbox-backed build/test capabilities.
- Resource/document analysis, including host-resolved resource reads and
  structured document tables.
- Web research, gated by the host's network policy and limited to the safe
  search/fetch tools.

Dataset analysis/quality/join, resource/document analysis and web research
declare `artifact.export` as an optional capability. The route receives it only
when the selected tool profile exposes the host-backed exporter. Export remains
an explicit user-requested action; it is not an automatic terminal step.

The dataset A* adapter accepts every data operation above when explicit
host-bound operation arguments are available. It does not invent predicates,
aggregate measures, transforms, or join keys. If those semantic arguments are
not already bound, materialization yields to the ordinary planner, which can
select among all operations allowed by the route envelope. Join materialization
adds separate host-resolved left/right dataset selections and schema checks.
In this case the selector does not persist the blueprint's generic template as
a task. It defers plan creation to the ordinary planner and carries the selected
route binding into that new plan, while the route-scoped tool view remains the
authority for both model-visible tools and execution. The plan also retains the
selected workflow definition and blueprint ancestry so route revalidation and
workflow continuation remain available after the fallback.

The resource/document workflow keeps table discovery and materialization
host-owned. After successful tool results, its A* continuation may append a
host-bound data operation using a typed reference to the completed
`document.table` dataset output. It never chooses table names or operation
arguments. Without a host-bound operation, ordinary planning continues within
the same route and envelope.

Repository and web workflows currently use the common route compiler and
execution envelope, with ordinary planning for request-specific paths.

## Next adapter seams

When adding another domain, keep its domain-specific context resolution,
transition graph and materialization in its adapter. The common route compiler
should only validate declared capability/context requirements, resolve tools
from the active host view, and produce the execution envelope. Tool execution
remains host-validated regardless of how a candidate path was proposed.
