#include "plan-workflow-state.h"

#include <algorithm>

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
