#pragma once

#include <cstddef>
#include <vector>

// A soft target influences the share, while demand prevents unused space from
// being reserved. Values are in tokens; callers own conversion and rendering.
struct common_agent_context_section_demand {
    size_t target_tokens = 0;
    size_t demand_tokens = 0;
};

struct common_agent_context_allocation {
    bool mandatory_fits = false;
    size_t available_tokens = 0;
    std::vector<size_t> section_tokens;
};

common_agent_context_allocation common_agent_allocate_context(
    size_t capacity_tokens, size_t mandatory_tokens, size_t reserved_tokens,
    const std::vector<common_agent_context_section_demand> & sections);
