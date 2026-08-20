#pragma once

#include <cstddef>
#include <limits>

// This is a per-inference measurement, not a second runtime state store.  The
// host supplies the model context limit and reserves; renderers supply the
// bounded input estimate before inference starts.
struct common_agent_context_budget {
    size_t context_limit_tokens = 0;
    size_t estimated_input_tokens = 0;
    size_t reserved_output_tokens = 0;
    size_t reserved_tool_tokens = 0;
    size_t safety_margin_tokens = 0;
    size_t soft_limit_percent = 80;
    size_t hard_limit_percent = 90;
};

enum class common_agent_context_pressure {
    normal,
    compact_recommended,
    compact_required,
    continuation_required,
};

struct common_agent_context_budget_evaluation {
    common_agent_context_pressure pressure = common_agent_context_pressure::normal;
    bool valid = true;
    size_t reserved_tokens = 0;
    size_t available_input_tokens = 0;
};

inline common_agent_context_budget_evaluation evaluate_common_agent_context_pressure(
        const common_agent_context_budget & budget) {
    common_agent_context_budget_evaluation evaluation;
    if (budget.context_limit_tokens == 0 ||
            budget.soft_limit_percent > budget.hard_limit_percent ||
            budget.hard_limit_percent > 100) {
        evaluation.valid = false;
        evaluation.pressure = common_agent_context_pressure::continuation_required;
        return evaluation;
    }

    if (budget.reserved_output_tokens > std::numeric_limits<size_t>::max() - budget.reserved_tool_tokens) {
        evaluation.valid = false;
        evaluation.pressure = common_agent_context_pressure::continuation_required;
        return evaluation;
    }
    evaluation.reserved_tokens = budget.reserved_output_tokens + budget.reserved_tool_tokens;
    if (evaluation.reserved_tokens > std::numeric_limits<size_t>::max() - budget.safety_margin_tokens) {
        evaluation.valid = false;
        evaluation.pressure = common_agent_context_pressure::continuation_required;
        return evaluation;
    }
    evaluation.reserved_tokens += budget.safety_margin_tokens;
    if (evaluation.reserved_tokens >= budget.context_limit_tokens) {
        evaluation.pressure = common_agent_context_pressure::continuation_required;
        return evaluation;
    }

    evaluation.available_input_tokens = budget.context_limit_tokens - evaluation.reserved_tokens;
    if (budget.estimated_input_tokens >= evaluation.available_input_tokens) {
        evaluation.pressure = common_agent_context_pressure::continuation_required;
        return evaluation;
    }

    const size_t soft_limit = evaluation.available_input_tokens * budget.soft_limit_percent / 100;
    const size_t hard_limit = evaluation.available_input_tokens * budget.hard_limit_percent / 100;
    evaluation.pressure = budget.estimated_input_tokens >= hard_limit
        ? common_agent_context_pressure::compact_required
        : budget.estimated_input_tokens >= soft_limit
            ? common_agent_context_pressure::compact_recommended
            : common_agent_context_pressure::normal;
    return evaluation;
}
