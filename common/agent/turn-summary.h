#pragma once

#include <cstddef>
#include <string>
#include <vector>

// Compact orchestration metadata for client and agent handoffs.
// This is not a reasoning trace.
struct common_agent_turn_summary {
    std::string mode;
    std::string status;
    std::string objective;
    std::vector<std::string> phases;
    std::vector<std::string> tools_used;
    size_t plan_revisions = 0;
    size_t sources = 0;
    size_t evidence_items = 0;
    size_t unresolved_items = 0;
    bool verified = false;
    std::string stop_reason;
    std::vector<std::string> unresolved;
};
