#pragma once

#include "plan/plan-types.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

// A bounded, host-owned projection of active execution state. It is working
// context for compaction and continuation, not a replacement for the plan or
// the authoritative resource store.
struct common_agent_working_state {
    std::string goal;
    std::string current_phase;
    std::vector<std::string> completed_steps;
    std::string active_step;
    std::vector<std::string> remaining_steps;
    std::vector<std::string> decisions;
    std::vector<std::string> constraints;
    std::vector<std::string> open_questions;
    std::vector<common_runtime_resource_ref> resource_refs;
    std::vector<std::string> chunk_status;
    std::vector<std::string> tool_results;
    std::string continuation_action;
};

inline std::string common_agent_working_state_bounded(
        const std::string & value,
        size_t max_chars) {
    return value.size() <= max_chars ? value : value.substr(0, max_chars);
}

inline void common_agent_working_state_add(
        std::vector<std::string> & values,
        const std::string & value,
        size_t & remaining_chars) {
    if (value.empty() || remaining_chars == 0) {
        return;
    }
    const size_t take = std::min(remaining_chars, value.size());
    values.push_back(value.substr(0, take));
    remaining_chars -= take;
}

inline common_agent_working_state make_common_agent_working_state(
        const common_plan_state & plan,
        size_t max_chars = 8192) {
    common_agent_working_state state;
    size_t remaining_chars = max_chars;
    state.goal = common_agent_working_state_bounded(plan.goal, std::min<size_t>(1024, remaining_chars));
    remaining_chars = remaining_chars > state.goal.size() ? remaining_chars - state.goal.size() : 0;

    const common_plan_step * active_step = nullptr;
    for (const auto & step : plan.steps) {
        if (step.status == common_plan_step_status::active ||
                (plan.active_step_id && step.id == *plan.active_step_id)) {
            active_step = &step;
        }
    }
    if (active_step != nullptr) {
        state.active_step = active_step->id + ": " + active_step->title;
        state.current_phase = common_plan_step_effective_mode(*active_step) == common_plan_step_mode::tool
            ? "tool"
            : common_plan_step_effective_mode(*active_step) == common_plan_step_mode::reasoning
                ? "reasoning"
                : "synthesis";
    } else {
        state.current_phase = plan.status == common_plan_status::completed ? "completed" : "planning";
    }

    for (const auto & step : plan.steps) {
        const std::string label = step.id + ": " + step.title;
        if (step.status == common_plan_step_status::completed) {
            common_agent_working_state_add(state.completed_steps, label, remaining_chars);
        } else if (step.status == common_plan_step_status::pending ||
                step.status == common_plan_step_status::active) {
            if (!active_step || step.id != active_step->id) {
                common_agent_working_state_add(state.remaining_steps, label, remaining_chars);
            }
        }
        if (step.result_summary) {
            common_agent_working_state_add(
                state.tool_results,
                step.id + ": " + *step.result_summary,
                remaining_chars);
        }
    }
    for (const auto & constraint : plan.constraints) {
        common_agent_working_state_add(
            state.constraints,
            constraint.id + ": " + constraint.description,
            remaining_chars);
    }
    for (const auto & assumption : plan.assumptions) {
        if (!assumption.valid) {
            common_agent_working_state_add(
                state.open_questions,
                assumption.id + ": " + assumption.statement,
                remaining_chars);
        }
    }
    for (const auto & observation : plan.observations) {
        for (const auto & resource : observation.resource_refs) {
            if (resource.uri.empty() || std::any_of(
                    state.resource_refs.begin(), state.resource_refs.end(),
                    [&](const auto & existing) { return existing.uri == resource.uri; })) {
                continue;
            }
            state.resource_refs.push_back(resource);
        }
        if (observation.source == "resource_chunk" && observation.resource_refs.size() == 1) {
            const auto & lineage = observation.resource_refs.front().lineage;
            common_agent_working_state_add(
                state.chunk_status,
                lineage.parent_uri + "[" + std::to_string(lineage.chunk_index) + "/" +
                    std::to_string(lineage.chunk_count) + "]",
                remaining_chars);
        }
    }
    state.continuation_action = plan.next_action.value_or("resume active plan step");
    return state;
}

inline std::string render_common_agent_working_state(
        const common_agent_working_state & state,
        size_t max_chars = 8192) {
    std::string rendered;
    auto append = [&](const std::string & key, const std::string & value) {
        if (value.empty() || rendered.size() >= max_chars) return;
        rendered += key + "=" + value + "\n";
    };
    append("goal", state.goal);
    append("current_phase", state.current_phase);
    append("active_step", state.active_step);
    append("continuation_action", state.continuation_action);
    for (const auto & value : state.completed_steps) append("completed_step", value);
    for (const auto & value : state.remaining_steps) append("remaining_step", value);
    for (const auto & value : state.decisions) append("decision", value);
    for (const auto & value : state.constraints) append("constraint", value);
    for (const auto & value : state.open_questions) append("open_question", value);
    for (const auto & value : state.chunk_status) append("chunk", value);
    for (const auto & value : state.tool_results) append("tool_result", value);
    for (const auto & value : state.resource_refs) append("resource_ref", value.uri);
    if (rendered.size() > max_chars) rendered.resize(max_chars);
    return rendered;
}
