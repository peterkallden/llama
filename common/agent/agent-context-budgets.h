#pragma once

#include "agent/agent-working-state.h"

#include <cstddef>

// Character budgets used when runtime state is rendered into model prompts.
// These are soft context budgets: the host may tune them for the selected
// model, while individual renderers still apply their own safety limits.
struct common_agent_context_budget_config {
    size_t plan_chars = 2048;
    size_t step_chars = 1400;
    size_t tool_observation_chars = 4096;
    size_t input_resources_chars = 2048;
    size_t deliberate_input_resources_chars = 1200;
    // Controller-owned bounded resource views. These are byte budgets, while
    // the renderer budgets above remain character-oriented prompt budgets.
    size_t resource_chunk_max_bytes = 4096;
    size_t resource_chunk_overlap_bytes = 256;

    size_t memory_chars = 4096;
    size_t memory_per_item_chars = 1200;
    size_t overlay_chars = 1600;
    size_t overlay_per_item_chars = 240;
    size_t deliberate_memory_chars = 900;
    size_t deliberate_memory_per_item_chars = 300;
    size_t deliberate_overlay_chars = 700;
    size_t deliberate_overlay_per_item_chars = 220;

    // Bounds for the host-owned working-state projection carried by
    // continuation checkpoints. These are deliberately separate from the
    // prompt section budgets above because a checkpoint is execution state,
    // not a second prompt renderer.
    common_agent_working_state_limits working_state;
};
