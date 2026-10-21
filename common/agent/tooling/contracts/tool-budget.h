#pragma once

#include <cstddef>

// Host configuration uses zero for the default, never for a one-call turn.
// Explicit positive limits remain authoritative at every execution boundary.
inline size_t common_agent_resolve_tool_round_limit(size_t configured) {
    return configured > 0 ? configured : 16;
}
