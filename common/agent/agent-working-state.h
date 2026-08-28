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
    std::vector<common_agent_dataset_ref> dataset_refs;
    std::vector<std::string> chunk_status;
    std::vector<std::string> tool_results;
    std::string continuation_action;
};

struct common_agent_working_state_limits {
    size_t max_total_chars = 8192;
    size_t max_value_chars = 1024;
    size_t max_completed_steps = 64;
    size_t max_remaining_steps = 64;
    size_t max_constraints = 32;
    size_t max_open_questions = 32;
    size_t max_resource_refs = 32;
    size_t max_dataset_refs = 32;
    size_t max_chunk_status = 64;
    size_t max_tool_results = 32;
};

inline std::string common_agent_working_state_bounded(
        const std::string & value,
        size_t max_chars) {
    return value.size() <= max_chars ? value : value.substr(0, max_chars);
}

inline void common_agent_working_state_add(
        std::vector<std::string> & values,
        const std::string & value,
        size_t & remaining_chars,
        size_t max_value_chars = 1024) {
    if (value.empty() || remaining_chars == 0) {
        return;
    }
    const size_t take = std::min({remaining_chars, max_value_chars, value.size()});
    values.push_back(value.substr(0, take));
    remaining_chars -= take;
}

inline common_agent_working_state make_common_agent_working_state(
        const common_plan_state & plan,
        const common_agent_working_state_limits & limits) {
    common_agent_working_state state;
    size_t remaining_chars = limits.max_total_chars;
    state.goal = common_agent_working_state_bounded(
        plan.goal, std::min(limits.max_value_chars, remaining_chars));
    remaining_chars = remaining_chars > state.goal.size() ? remaining_chars - state.goal.size() : 0;

    const common_plan_step * active_step = nullptr;
    for (const auto & step : plan.steps) {
        if (step.status == common_plan_step_status::active ||
                (plan.active_step_id && step.id == *plan.active_step_id)) {
            active_step = &step;
        }
    }
    if (active_step != nullptr) {
        state.active_step = common_agent_working_state_bounded(
            active_step->id + ": " + active_step->title,
            std::min(limits.max_value_chars, remaining_chars));
        remaining_chars = remaining_chars > state.active_step.size()
            ? remaining_chars - state.active_step.size() : 0;
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
            if (state.completed_steps.size() < limits.max_completed_steps) {
                common_agent_working_state_add(
                    state.completed_steps, label, remaining_chars, limits.max_value_chars);
            }
        } else if (step.status == common_plan_step_status::pending ||
                step.status == common_plan_step_status::active) {
            if ((!active_step || step.id != active_step->id) &&
                    state.remaining_steps.size() < limits.max_remaining_steps) {
                common_agent_working_state_add(
                    state.remaining_steps, label, remaining_chars, limits.max_value_chars);
            }
        }
        if (step.result_summary && state.tool_results.size() < limits.max_tool_results) {
            common_agent_working_state_add(
                state.tool_results,
                step.id + ": " + *step.result_summary,
                remaining_chars,
                limits.max_value_chars);
        }
    }
    for (const auto & constraint : plan.constraints) {
        if (state.constraints.size() >= limits.max_constraints) break;
        common_agent_working_state_add(
            state.constraints, constraint.id + ": " + constraint.description,
            remaining_chars, limits.max_value_chars);
    }
    for (const auto & assumption : plan.assumptions) {
        if (!assumption.valid && state.open_questions.size() < limits.max_open_questions) {
            common_agent_working_state_add(
                state.open_questions,
                assumption.id + ": " + assumption.statement,
                remaining_chars,
                limits.max_value_chars);
        }
    }
    for (const auto & observation : plan.observations) {
        for (const auto & resource : observation.resource_refs) {
            if (state.resource_refs.size() >= limits.max_resource_refs || resource.uri.empty() || std::any_of(
                    state.resource_refs.begin(), state.resource_refs.end(),
                    [&](const auto & existing) { return existing.uri == resource.uri; })) {
                continue;
            }
            state.resource_refs.push_back(resource);
        }
        for (const auto & dataset : observation.dataset_refs) {
            if (state.dataset_refs.size() >= limits.max_dataset_refs || dataset.uri.empty() || std::any_of(
                    state.dataset_refs.begin(), state.dataset_refs.end(),
                    [&](const auto & existing) { return existing.uri == dataset.uri; })) {
                continue;
            }
            state.dataset_refs.push_back(dataset);
        }
        if (observation.source == "resource_chunk" && observation.resource_refs.size() == 1) {
            if (state.chunk_status.size() >= limits.max_chunk_status) continue;
            const auto & lineage = observation.resource_refs.front().lineage;
            common_agent_working_state_add(
                state.chunk_status,
                lineage.parent_uri + "[" + std::to_string(lineage.chunk_index) + "/" +
                    std::to_string(lineage.chunk_count) + "];status=completed;observation=" +
                    observation.id,
                remaining_chars,
                limits.max_value_chars);
        }
    }
    state.continuation_action = common_agent_working_state_bounded(
        plan.next_action.value_or("resume active plan step"),
        std::min(limits.max_value_chars, remaining_chars));
    return state;
}

inline common_agent_working_state make_common_agent_working_state(
        const common_plan_state & plan,
        size_t max_chars = 8192) {
    common_agent_working_state_limits limits;
    limits.max_total_chars = max_chars;
    return make_common_agent_working_state(plan, limits);
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
    for (const auto & value : state.dataset_refs) append("dataset_ref", value.uri);
    if (rendered.size() > max_chars) rendered.resize(max_chars);
    return rendered;
}
