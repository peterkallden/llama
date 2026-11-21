#include "agent/agent-context-allocation.h"
#include "agent/agent-resource-chunk-budget.h"

#include <cstdlib>
#include <iostream>
#include <limits>

static void check(bool condition, const char * message) {
    if (!condition) { std::cerr << message << '\n'; std::exit(1); }
}

class chunk_budget_inference final : public common_agent_inference {
public:
    size_t capacity = 0;
    bool generate(const common_agent_generation_request &, common_agent_generation_result &) override { return false; }
    size_t context_capacity_tokens() const override { return capacity; }
};

int main() {
    const auto balanced = common_agent_allocate_context(100, 20, 10, {{10, 100}, {20, 100}});
    check(balanced.mandatory_fits && balanced.available_tokens == 70, "incorrect available capacity");
    check(balanced.section_tokens[0] + balanced.section_tokens[1] == 70 &&
        balanced.section_tokens[1] > balanced.section_tokens[0], "targets did not influence sharing");
    const auto surplus = common_agent_allocate_context(100, 20, 10, {{10, 2}, {20, 100}});
    check(surplus.section_tokens[0] == 2 && surplus.section_tokens[1] == 68,
        "unused allocation was not redistributed beyond the target");
    const auto underfilled = common_agent_allocate_context(100, 20, 10, {{10, 2}, {20, 3}});
    check(underfilled.section_tokens[0] == 2 && underfilled.section_tokens[1] == 3,
        "allocation exceeded demand");
    const auto no_room = common_agent_allocate_context(100, 91, 10, {{1, 1}});
    check(!no_room.mandatory_fits && no_room.section_tokens[0] == 0,
        "mandatory content or reserve was silently cut");
    const auto overflow = common_agent_allocate_context(100, 1, std::numeric_limits<size_t>::max(), {{1, 1}});
    check(!overflow.mandatory_fits, "reserve arithmetic overflowed");

    chunk_budget_inference inference;
    common_agent_context_budget_config budgets;
    common_runtime_resource_chunk_policy chunk;
    std::string error;
    check(common_agent_resolve_resource_chunk_policy(inference, budgets, "Read this document", chunk, error) &&
        chunk.max_bytes == 4096, "unknown context did not use a bounded fallback");
    inference.capacity = 2048;
    check(common_agent_resolve_resource_chunk_policy(inference, budgets, "Read this document", chunk, error) &&
        chunk.max_bytes < 2048 && chunk.overlap_bytes < chunk.max_bytes, "small context was not respected");
    inference.capacity = 32768;
    check(common_agent_resolve_resource_chunk_policy(inference, budgets, "Read this document", chunk, error) &&
        chunk.max_bytes > 4096 && chunk.max_bytes <= 8192, "large context was not allocated dynamically");
    budgets.resource_chunk_max_bytes = 1024;
    check(common_agent_resolve_resource_chunk_policy(inference, budgets, "Read this document", chunk, error) &&
        chunk.max_bytes == 1024, "explicit maximum was not respected");
    inference.capacity = 128;
    check(!common_agent_resolve_resource_chunk_policy(inference, budgets, "Read this document", chunk, error),
        "insufficient mandatory context was accepted");
}
