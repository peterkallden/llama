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
- OpenAPI onboarding, which fetches a JSON specification into a host resource,
  admits anonymous read-only operations, and exposes those operations on the
  next turn. The route's optional `openapi.read` capability resolves against
  the newly registered provider so the same workflow can continue safely.

The starter blueprint, procedure and workflow for this path are imported from
[`agent-bootstrap-workflows-v1.json`](../examples/agent-bootstrap-workflows-v1.json).
The blueprint binds `workflow://openapi/resource-onboarding`; the workflow
requires `web.fetch` and `openapi.connect`, and adds available read-only API
operations through an optional capability after admission. The model receives
the fetched resource handle from `web.fetch`; the host owns operation discovery,
policy filtering and the next-turn tool view.

OpenAPI HTTP failures retain retryability and a status-specific failure class:
invalid request/input (400/405/415/422), host credentials or access policy
(401/403), missing resource (404/410), timeout/conflict/size-limit, and
transient service/rate-limit failures (408/425/429/5xx). Retry guidance is
host-derived from `Retry-After` (delta seconds or HTTP date), standard
`RateLimit`/`RateLimit-Reset`, and common `X-RateLimit-*` / `X-Rate-Limit-*`
headers. The runtime reports the observed delay and headers; it does not
automatically replay a request or let the model choose a retry time.

An imported package without a `project_id` is host-wide within its namespace:
its procedures use global memory scope, and its blueprints/workflows use global
template scope. Supplying a `project_id` installs the package in that project
instead. Global template visibility does not make task plans global; instantiated
plans still use the caller's requested turn/session/project scope. The plan
scope matcher for active task plans remains exact.

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
