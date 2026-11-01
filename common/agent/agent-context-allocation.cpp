#include "agent-context-allocation.h"

#include <algorithm>
#include <limits>

common_agent_context_allocation common_agent_allocate_context(
        size_t capacity, size_t mandatory, size_t reserved,
        const std::vector<common_agent_context_section_demand> & sections) {
    common_agent_context_allocation out;
    out.section_tokens.resize(sections.size());
    if (mandatory > capacity || reserved > capacity - mandatory) return out;
    out.mandatory_fits = true;
    out.available_tokens = capacity - mandatory - reserved;
    size_t remaining = out.available_tokens;
    // Repeated weighted passes let sections exceed their targets and share
    // capacity abandoned by sections with less content than requested.
    while (remaining != 0) {
        size_t weight_sum = 0;
        for (size_t i = 0; i < sections.size(); ++i) {
            if (out.section_tokens[i] >= sections[i].demand_tokens) continue;
            const size_t weight = std::max<size_t>(1, sections[i].target_tokens);
            weight_sum = weight > std::numeric_limits<size_t>::max() - weight_sum
                ? std::numeric_limits<size_t>::max() : weight_sum + weight;
        }
        if (weight_sum == 0) break;
        const size_t pass_remaining = remaining;
        for (size_t i = 0; i < sections.size() && remaining != 0; ++i) {
            if (out.section_tokens[i] >= sections[i].demand_tokens) continue;
            const size_t weight = std::max<size_t>(1, sections[i].target_tokens);
            const auto fraction = static_cast<long double>(weight) / weight_sum;
            const size_t share = std::max<size_t>(1, static_cast<size_t>(pass_remaining * fraction));
            const size_t grant = std::min({share, remaining,
                sections[i].demand_tokens - out.section_tokens[i]});
            out.section_tokens[i] += grant;
            remaining -= grant;
        }
    }
    return out;
}
