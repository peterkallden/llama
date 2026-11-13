# Comprehension refresh: local Luna UI test

Use `~/scripts/llama-agent-web.sh` with Luna and tracing. First rebuild daemon,
web adapter and dependent libraries, and import the current Bootstrap JSON into
the configured plan store. Verify blueprint revision, project scope and graph
write policy. Restarting alone does not import Bootstrap. Use a separate test
project, without clearing an existing database. Record source commit and dirty
state, model profile, project scope, bootstrap import and trace locations.

## Standalone code entity

In a fresh chat ask:

> Use comprehension refresh to inspect `common/agent/context-graph/context-graph.h`.
> Choose one useful concept and propose one short graph description containing
> purpose, invariant, dependencies and change hazard. Cite current source and
> revision, mark uncertainty, and do not create a relation merely to create a
> node. Report the saved node ID and whether the description was reused.

Verify blueprint selection and source reads in trace, then `graph.find` and
`graph.describe.propose`, policy decision, scope and publication receipt. Inspect
the graph: a standalone node must exist with zero edges, evidence references,
source revision and proposed status. Record actual tool calls and counts.

## New session and daemon restart

Start a new chat in the same project and ask:

> Find the graph note for the context graph header from the previous session.
> Give its purpose, source revision and references. Check the current source
> before treating the note as current.

Verify the same node is found and the source was checked. Replay the same
proposal with current graph revision and identical evidence and confirm reuse.
Restart the daemon with the normal script, reconnect, and repeat the lookup in a
new session. Confirm persistence, scope, revision and evidence.

## Another domain

Using an existing authorized research artifact, document or case, request one
sourced comprehension note with explicit uncertainty. Verify source identity,
proposal status and absence of invented relations. If no suitable source is
available, record a skip rather than inventing source material.

## Acceptance

Pass only if Luna selects the method, actually saves a standalone node, and a
later session finds it after daemon restart. Keep prompts, UI observations,
trace references, node and graph revision, call counts and limitations. Report
separately if post-verification continuation was not exercised. Tool-call limits
are not proof of a host-enforced entity frontier or traversal-depth limit.
