#include "agent-runtime-context.h"

#include <nlohmann/json.hpp>

bool common_agent_runtime_transition_cognitive_mode(
        common_agent_runtime_turn_context & context,
        common_agent_cognitive_mode to,
        const std::string & reason,
        const std::string & source,
        common_runtime_trace_stage trace_stage) {
    auto & state = context.request.cognitive_state;
    if (state.mode == to) return true;

    const auto from = state.mode;
    const bool allowed = common_agent_cognitive_transition(state, to, reason);
    const auto detail = nlohmann::ordered_json{
        {"type", "cognitive_mode_transition"},
        {"from", common_agent_cognitive_mode_name(from)},
        {"to", common_agent_cognitive_mode_name(to)},
        {"reason", reason},
        {"source", source},
        {"allowed", allowed},
    }.dump();
    context.emit_trace(
        trace_stage,
        allowed ? common_runtime_trace_kind::updated : common_runtime_trace_kind::failed,
        detail,
        context.outer_plan == nullptr ? std::string() : context.outer_plan->id,
        {});
    return allowed;
}
