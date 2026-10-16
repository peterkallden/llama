#include "plan-workflow-state.h"

#include <algorithm>
#include <cmath>
#include <nlohmann/json.hpp>

bool common_plan_workflow_observation_is_usable(const common_plan_observation & observation) {
    if (!std::isfinite(observation.confidence) || observation.confidence <= 0.0f) return false;
    const auto value = nlohmann::ordered_json::parse(observation.summary, nullptr, false);
    if (!value.is_object() && !value.is_array()) return false;
    if (value.is_object() && value.contains("ok") && value["ok"] != true) return false;
    if (value.is_object() && value.contains("isError") && value["isError"] != false) return false;
    return true;
}

const common_plan_observation * common_plan_workflow_observation(
        const common_plan_state & plan, const std::string & source_transition_id,
        bool require_dataset) {
    const size_t start = plan.route_binding
        ? std::min(plan.route_binding->workflow_step_start_index, plan.steps.size()) : 0;
    for (size_t index = plan.steps.size(); index > start; --index) {
        const auto & step = plan.steps[index - 1];
        if (step.status != common_plan_step_status::completed || !step.tool_call) continue;
        const bool has_alias = step.semantic_alias &&
            step.semantic_alias->rfind("workflow-transition:", 0) == 0;
        if (!source_transition_id.empty() && !has_alias) continue;
        const auto aliases = has_alias
            ? step.semantic_alias->substr(std::string("workflow-transition:").size()) : std::string{};
        const auto comma = aliases.rfind(',');
        const auto last = aliases.substr(comma == std::string::npos ? 0 : comma + 1);
        std::string choice;
        if (!source_transition_id.empty() && last != source_transition_id) {
            size_t begin = 0;
            while (begin < aliases.size()) {
                const auto end = aliases.find(',', begin);
                const auto token = aliases.substr(begin,
                    end == std::string::npos ? std::string::npos : end - begin);
                const auto prefix = source_transition_id + "=";
                if (token.rfind(prefix, 0) == 0) choice = token.substr(prefix.size());
                if (end == std::string::npos) break;
                begin = end + 1;
            }
            if (choice.empty()) continue;
        }
        const auto prefix = "tool:" + step.id + ":";
        const common_plan_observation * tool_observation = nullptr;
        for (auto observed = plan.observations.rbegin(); observed != plan.observations.rend(); ++observed) {
            if (observed->id.rfind(prefix, 0) == 0 &&
                    observed->source == step.tool_call->name &&
                    common_plan_workflow_observation_is_usable(*observed)) {
                tool_observation = &*observed;
                break;
            }
        }
        if (!tool_observation) continue;
        if (choice.empty()) {
            if (!require_dataset || !tool_observation->dataset_refs.empty()) return tool_observation;
            continue;
        }
        if (require_dataset) continue;
        const auto choice_id = "workflow_choice:" + step.id + ":" + source_transition_id;
        for (auto observed = plan.observations.rbegin(); observed != plan.observations.rend(); ++observed) {
            if (observed->id != choice_id || observed->source != "workflow_choice" ||
                    !common_plan_workflow_observation_is_usable(*observed) ||
                    std::find(observed->evidence_ids.begin(), observed->evidence_ids.end(),
                        tool_observation->id) == observed->evidence_ids.end()) continue;
            const auto value = nlohmann::ordered_json::parse(observed->summary, nullptr, false);
            if (value.is_object() && value.contains("transition_id") &&
                    value["transition_id"] == source_transition_id && value.contains("choice") &&
                    value["choice"] == choice && value.contains("verified_against_host_candidates") &&
                    value["verified_against_host_candidates"] == true) return &*observed;
        }
    }
    return nullptr;
}

std::string common_plan_workflow_completed_state(
        const common_plan_state & plan,
        const common_plan_workflow_definition & definition) {
    const size_t start_index = plan.route_binding
        ? std::min(plan.route_binding->workflow_step_start_index, plan.steps.size()) : 0;
    if (definition.transitions.empty()) {
        const bool has_completed_tool = std::any_of(
            plan.steps.begin() + start_index, plan.steps.end(), [](const auto & step) {
                return step.status == common_plan_step_status::completed && step.tool_call.has_value();
            });
        return has_completed_tool && !definition.terminal_states.empty()
            ? definition.terminal_states.front() : std::string{};
    }

    std::string state = definition.start_state;
    for (size_t index = start_index; index < plan.steps.size(); ++index) {
        const auto & step = plan.steps[index];
        if (step.status != common_plan_step_status::completed || !step.semantic_alias ||
                step.semantic_alias->rfind("workflow-transition:", 0) != 0) continue;
        const auto aliases = step.semantic_alias->substr(std::string("workflow-transition:").size());
        size_t begin = 0;
        while (begin <= aliases.size()) {
            const auto end = aliases.find(',', begin);
            auto token = aliases.substr(begin,
                end == std::string::npos ? std::string::npos : end - begin);
            std::string choice_id;
            const auto choice_at = token.find('=');
            if (choice_at != std::string::npos) {
                choice_id = token.substr(choice_at + 1);
                token.resize(choice_at);
            }
            const auto transition = std::find_if(definition.transitions.begin(),
                definition.transitions.end(), [&](const auto & candidate) {
                    return candidate.id == token && candidate.from_state == state;
                });
            if (transition != definition.transitions.end()) {
                if (transition->kind == common_plan_workflow_transition_kind::model_choice &&
                        !transition->choices.empty()) {
                    const auto choice = std::find_if(transition->choices.begin(),
                        transition->choices.end(), [&](const auto & candidate) {
                            return candidate.id == choice_id;
                        });
                    if (choice != transition->choices.end()) state = choice->to_state;
                } else if (transition->kind != common_plan_workflow_transition_kind::model_choice ||
                        !choice_id.empty()) {
                    // Candidate-backed choices are host-checked before the
                    // resulting alias is persisted on the selected tool step.
                    state = transition->to_state;
                }
            }
            if (end == std::string::npos) break;
            begin = end + 1;
        }
    }
    return state;
}

bool common_plan_workflow_is_terminal(
        const common_plan_state & plan,
        const common_plan_workflow_definition & definition) {
    const auto state = common_plan_workflow_completed_state(plan, definition);
    return !state.empty() && std::find(definition.terminal_states.begin(),
        definition.terminal_states.end(), state) != definition.terminal_states.end();
}
