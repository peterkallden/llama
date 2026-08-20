#pragma once

#include "agent-generation.h"

#include <cstddef>

// Run a host-bounded structured generation sequence. The accept callback owns
// schema parsing and may update its caller-owned parsed result. A failed or
// rejected attempt is never combined with the next attempt; the final
// generation result is returned for normal status/error reporting.
template <typename Generate, typename Accept>
auto common_agent_bounded_structured_regeneration(
        Generate && generate,
        Accept && accept,
        size_t max_regenerations = 1) {
    for (size_t attempt = 0; ; ++attempt) {
        auto generation = generate(attempt != 0);
        if (common_agent_generation_succeeded(generation) && accept(generation)) {
            return generation;
        }
        // Regeneration is only valid for a completed generation whose
        // structured payload was rejected. Host cancellation and backend
        // failures must propagate immediately and must not consume another
        // inference attempt.
        if (!common_agent_generation_succeeded(generation) ||
                attempt >= max_regenerations) {
            return generation;
        }
    }
}
