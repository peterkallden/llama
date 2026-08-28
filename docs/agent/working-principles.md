# Agent implementation working principles

When a component has reached a stable implementation, revisit the surrounding
history and remove superseded aliases, duplicate paths and temporary
compatibility layers. Do this only after the replacement contract is explicit,
the ownership boundary is understood, and focused positive and negative tests
prove the new behavior.

Contracts remain the authority during cleanup. Removing an old path is a
deliberate architectural decision, not permission to weaken validation or
silently reinterpret old data. If an old representation is encountered, the
runtime should fail with a bounded diagnostic unless a migration has been
explicitly designed and tested.

For dataset references this means one canonical `dataset://` identity. The
host dataset registry and the source-resource authority determine whether a
reference is usable; URI aliases must not create a second execution model.
