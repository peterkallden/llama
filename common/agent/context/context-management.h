#pragma once

#include <cstddef>

struct common_agent_context_management_config {
    bool enabled = true;
    bool persist_session = true;
    bool semantic_compaction = true;
    size_t recent_turns = 2;
    int summary_max_tokens = 512;
    size_t max_summary_calls_per_turn = 1;
    size_t max_history_chars = 32768;
};
