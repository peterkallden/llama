# Agent memory

Agent memory records remain in the existing scoped memory stores. Conversation
turns and checkpoints are resource records, not memory records; this keeps
session history, reusable memory, and source resources under their existing
retention and authority rules.

## Temporal facts

Every record keeps `created_at` as its registration time. Optional versioned
metadata carries:

- `agent.temporal.v1.observed_at`: when the observation was made;
- `agent.temporal.v1.valid_from`: inclusive start of the fact's validity;
- `agent.temporal.v1.valid_to`: exclusive end of its validity.

The interval is `[valid_from, valid_to)`. Either endpoint may be unknown; the
store does not invent dates. If both are set, `valid_from` must be less than
`valid_to`. Older records without these keys continue to load and are reported
with unknown validity. SQLite, Cozo, and in-memory backends retain the fields
through their existing metadata serialization.

Supersession uses the existing relation graph with `relation="supersedes"`:
the new record is `from`, and the replaced record is `to`. Both endpoints must
have identical scope identity, and cycles are rejected. Relation reads receive
the same scoped query as memory search, so a relation cannot disclose or hide a
record outside the caller's namespace/session/project/turn.

`memory.search` accepts optional `as_of` (Unix seconds) and `history`:

- default/current mode excludes records not yet valid, expired at that time,
  or superseded by a relation already present at that time;
- `history: true` retains expired and superseded records for inspection, but
  still excludes facts whose `valid_from` is after `as_of`;
- recency ranking uses `observed_at` when present, otherwise `created_at`.

Temporal filtering happens before final ranking. `as_of` defaults to the
host's current time; it is not a model-controlled memory scope. New memory
records and corrections still pass through the existing proposal and admission
policy—these fields do not bypass approval.

## Conversation context

The session host writes accepted user turns, completed assistant turns, and
immutable checkpoint revisions through the existing resource store. Internal
`agent.context.*` records are scoped to namespace/session and excluded from
ordinary resource candidates. A persistent resource backend supports restore
after daemon restart; in-memory storage is transient. Reset begins a new
conversation epoch without deleting older durable records.

Checkpoint summaries are orientation only. They refer to immutable source
turns and are validated against source handles and a content fingerprint.
Verbatim user instructions remain host-retained evidence and are rendered
exactly when needed; generated summaries cannot grant them new authority.
