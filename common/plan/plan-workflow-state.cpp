#include "plan-workflow-state.h"

#include <algorithm>
#include <cmath>
#include <nlohmann/json.hpp>

namespace {
// Both unavailable tool outcomes and model-selected terminal outcomes retain
// the same host-verified entry frontier. Never reuse another route's handoff.
void attach_entry_handoff(const common_plan_state & plan, nlohmann::ordered_json & payload) {
    if (!plan.route_binding || !plan.workflow_definition ||
            plan.route_binding->continuation_source_fingerprint.empty()) return;
    for (auto observed = plan.observations.rbegin(); observed != plan.observations.rend(); ++observed) {
        if (observed->source != "workflow_handoff") continue;
        const auto handoff = nlohmann::ordered_json::parse(observed->summary, nullptr, false);
        if (!handoff.is_object() || !handoff.contains("value") ||
                handoff.value("source_fingerprint", nlohmann::ordered_json{}) != plan.route_binding->continuation_source_fingerprint ||
                handoff.value("target_revision", nlohmann::ordered_json{}) != plan.workflow_definition->workflow_revision ||
                observed->id != "workflow_handoff:" + plan.route_binding->continuation_source_fingerprint +
                    ":" + plan.workflow_definition->workflow_ref) continue;
        payload["handoff"] = handoff;
        payload["handoff_observation_id"] = observed->id;
        return;
    }
}

const common_plan_workflow_transition * declared_failure_transition(const common_plan_state & plan,
        const common_plan_step & step, const std::string & observation_id) {
    if (!plan.workflow_definition || !plan.route_binding || step.status != common_plan_step_status::failed ||
            !step.tool_call || !step.semantic_alias || step.semantic_alias->rfind("workflow-transition:", 0) != 0) return nullptr;
    const auto start = std::min(plan.route_binding->workflow_step_start_index, plan.steps.size());
    if (std::none_of(plan.steps.begin() + start, plan.steps.end(), [&](const auto & item) { return item.id == step.id; })) return nullptr;
    const auto comma = step.semantic_alias->rfind(',');
    const auto transition_id = step.semantic_alias->substr(comma == std::string::npos
        ? std::string("workflow-transition:").size() : comma + 1);
    const auto & definition = *plan.workflow_definition;
    const auto transition = std::find_if(definition.transitions.begin(), definition.transitions.end(), [&](const auto & item) {
        return item.id == transition_id && item.kind == common_plan_workflow_transition_kind::tool &&
            item.tool_name == step.tool_call->name && !item.failure_terminal_state.empty() &&
            std::find(definition.terminal_states.begin(), definition.terminal_states.end(), item.failure_terminal_state) != definition.terminal_states.end();
    });
    if (transition == definition.transitions.end()) return nullptr;
    const std::string prefix = "tool:" + step.id + ":";
    const auto latest = std::find_if(plan.observations.rbegin(), plan.observations.rend(), [&](const auto & item) {
        return item.id.rfind(prefix, 0) == 0 && item.source == step.tool_call->name;
    });
    if (latest == plan.observations.rend() || latest->id != observation_id) return nullptr;
    const auto value = nlohmann::ordered_json::parse(latest->summary, nullptr, false);
    if (!value.is_object() || !value.contains("failure") || !value["failure"].is_object()) return nullptr;
    const auto & failure = value["failure"];
    if (!failure.contains("http_status") || !failure["http_status"].is_number_integer() ||
            !failure.contains("step_id") || !failure["step_id"].is_string() || failure["step_id"] != step.id ||
            !failure.contains("tool") || !failure["tool"].is_string() || failure["tool"] != step.tool_call->name) return nullptr;
    const int status = failure["http_status"].get<int>();
    return std::find(transition->handled_http_statuses.begin(), transition->handled_http_statuses.end(), status) !=
        transition->handled_http_statuses.end() ? &*transition : nullptr;
}

bool terminal_choice_valid(const common_plan_state & plan,
        const common_plan_workflow_definition & definition, const std::string & state,
        const std::string & transition_id, const std::string & choice_id,
        const std::string & source_step_id, const std::string & source_observation_id) {
    const size_t start = plan.route_binding ? std::min(plan.route_binding->workflow_step_start_index, plan.steps.size()) : 0;
    const auto step = std::find_if(plan.steps.begin() + start, plan.steps.end(), [&](const auto & item) {
        return item.id == source_step_id && item.status == common_plan_step_status::completed && item.tool_call;
    });
    if (step == plan.steps.end() || source_observation_id.rfind("tool:" + source_step_id + ":", 0) != 0) return false;
    const auto observation = std::find_if(plan.observations.begin(), plan.observations.end(), [&](const auto & item) {
        return item.id == source_observation_id && item.source == step->tool_call->name;
    });
    if (observation == plan.observations.end() || !common_plan_workflow_observation_is_usable(*observation)) return false;
    const auto transition = std::find_if(definition.transitions.begin(), definition.transitions.end(), [&](const auto & item) {
        return item.id == transition_id && item.from_state == state && item.kind == common_plan_workflow_transition_kind::model_choice;
    });
    if (transition == definition.transitions.end()) return false;
    const auto choice = std::find_if(transition->choices.begin(), transition->choices.end(), [&](const auto & item) {
        return item.id == choice_id && std::find(definition.terminal_states.begin(), definition.terminal_states.end(), item.to_state) != definition.terminal_states.end();
    });
    return choice != transition->choices.end() && common_plan_workflow_choice_available(*observation, *choice);
}
}

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
        bool require_dataset, const std::string & tool_name) {
    const size_t start = plan.route_binding
        ? std::min(plan.route_binding->workflow_step_start_index, plan.steps.size()) : 0;
    for (size_t index = plan.steps.size(); index > start; --index) {
        const auto & step = plan.steps[index - 1];
        if (step.status != common_plan_step_status::completed || !step.tool_call ||
                (!tool_name.empty() && step.tool_call->name != tool_name)) continue;
        const bool has_alias = step.semantic_alias &&
            step.semantic_alias->rfind("workflow-transition:", 0) == 0;
        const bool direct_alias = step.semantic_alias && *step.semantic_alias == source_transition_id;
        if (!source_transition_id.empty() && !has_alias && !direct_alias) continue;
        const auto aliases = has_alias
            ? step.semantic_alias->substr(std::string("workflow-transition:").size()) : std::string{};
        const auto comma = aliases.rfind(',');
        const auto last = aliases.substr(comma == std::string::npos ? 0 : comma + 1);
        std::string choice;
        if (!source_transition_id.empty() && last != source_transition_id && !direct_alias) {
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
        const bool handled_failure = common_plan_workflow_failure_handled(plan, step);
        if ((step.status != common_plan_step_status::completed && !handled_failure) || !step.semantic_alias ||
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
                } else if (handled_failure && transition->kind == common_plan_workflow_transition_kind::tool) {
                    state = transition->failure_terminal_state;
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
        for (const auto & recorded : plan.observations) {
            if (recorded.source != "workflow_terminal_choice" || recorded.evidence_ids.size() != 1) continue;
            const auto value = nlohmann::ordered_json::parse(recorded.summary, nullptr, false);
            if (!value.is_object() || !value.contains("source_step_id") || !value["source_step_id"].is_string() ||
                    value["source_step_id"] != step.id || !value.contains("host_validated") ||
                    value["host_validated"] != true || !value.contains("transition_id") ||
                    !value["transition_id"].is_string() || !value.contains("choice_id") ||
                    !value["choice_id"].is_string()) continue;
            const auto transition_id = value.value("transition_id", std::string{});
            const auto choice_id = value.value("choice_id", std::string{});
            if (!terminal_choice_valid(plan, definition, state, transition_id, choice_id, step.id, recorded.evidence_ids.front())) continue;
            const auto & transition = *std::find_if(definition.transitions.begin(), definition.transitions.end(),
                [&](const auto & item) { return item.id == transition_id && item.from_state == state; });
            const auto & choice = *std::find_if(transition.choices.begin(), transition.choices.end(),
                [&](const auto & item) { return item.id == choice_id; });
            state = choice.to_state;
            break;
        }
    }
    return state;
}

bool common_plan_workflow_handled_failure_observation(const common_plan_state & plan,
        const std::string & step_id, const std::string & failure_observation_id,
        common_plan_observation & result, std::string & error) {
    const auto step = std::find_if(plan.steps.begin(), plan.steps.end(), [&](const auto & item) { return item.id == step_id; });
    const auto * transition = step == plan.steps.end() ? nullptr : declared_failure_transition(plan, *step, failure_observation_id);
    if (!transition) { error = "workflow failure is not an explicitly declared outcome"; return false; }
    auto state = common_plan_workflow_completed_state(plan, *plan.workflow_definition);
    const auto aliases = step->semantic_alias->substr(std::string("workflow-transition:").size());
    size_t offset = 0;
    while (true) {
        const auto end = aliases.find(',', offset);
        const auto token = aliases.substr(offset, end == std::string::npos ? std::string::npos : end - offset);
        if (end == std::string::npos) {
            if (token != transition->id || state != transition->from_state) {
                error = "workflow failure does not match the active transition"; return false;
            }
            break;
        }
        const auto equals = token.find('=');
        if (equals == std::string::npos || equals == 0 || equals + 1 == token.size()) {
            error = "workflow failure has an invalid preceding choice"; return false;
        }
        const auto prior = std::find_if(plan.workflow_definition->transitions.begin(), plan.workflow_definition->transitions.end(),
            [&](const auto & item) { return item.id == token.substr(0, equals) && item.from_state == state &&
                item.kind == common_plan_workflow_transition_kind::model_choice; });
        if (prior == plan.workflow_definition->transitions.end()) {
            error = "workflow failure has an invalid choice transition"; return false;
        }
        if (prior->choices.empty()) state = prior->to_state;
        else {
            const auto choice = std::find_if(prior->choices.begin(), prior->choices.end(),
                [&](const auto & item) { return item.id == token.substr(equals + 1); });
            const auto * source = common_plan_workflow_observation(plan, {});
            if (choice == prior->choices.end() || !source || !common_plan_workflow_choice_available(*source, *choice)) {
                error = "workflow failure has an ungrounded preceding choice"; return false;
            }
            state = choice->to_state;
        }
        offset = end + 1;
    }
    nlohmann::ordered_json payload{{"step_id", step_id},
        {"transition_id", transition->id}, {"terminal_state", transition->failure_terminal_state}};
    // Preserve the host-validated entry evidence when this workflow was
    // reached from a candidate collection. A declared recovery link can then
    // choose another observed candidate without searching again or asking the
    // model to reconstruct an identity. The handoff is already bounded and
    // persisted as host evidence on this plan.
    attach_entry_handoff(plan, payload);
    result = {};
    result.id = "workflow_handled_failure:" + step_id + ":" + failure_observation_id;
    result.source = "workflow_handled_failure";
    result.summary = payload.dump();
    result.confidence = 1.0f;
    result.evidence_ids = {failure_observation_id};
    error.clear(); return true;
}

bool common_plan_workflow_failure_handled(const common_plan_state & plan, const common_plan_step & step) {
    if (step.status != common_plan_step_status::failed) return false;
    for (auto recorded = plan.observations.rbegin(); recorded != plan.observations.rend(); ++recorded) {
        if (recorded->source != "workflow_handled_failure" || recorded->evidence_ids.size() != 1) continue;
        const auto value = nlohmann::ordered_json::parse(recorded->summary, nullptr, false);
        if (!value.is_object() || !value.contains("step_id") || value["step_id"] != step.id ||
                !value.contains("transition_id") || !value["transition_id"].is_string() ||
                !value.contains("terminal_state") || !value["terminal_state"].is_string()) continue;
        const auto * transition = declared_failure_transition(plan, step, recorded->evidence_ids.front());
        if (transition && value["transition_id"] == transition->id && value["terminal_state"] == transition->failure_terminal_state) return true;
        // After a host-authorized route transition, the old definition is no
        // longer active. Its recorded continuation is the durable proof that
        // this exact terminal outcome was accepted before the transition.
        if (!plan.route_binding || !step.tool_call) continue;
        const auto position = std::find_if(plan.steps.begin(), plan.steps.end(),
            [&](const auto & item) { return item.id == step.id; });
        if (position == plan.steps.end() || static_cast<size_t>(position - plan.steps.begin()) >=
                plan.route_binding->workflow_step_start_index) continue;
        const auto latest = std::find_if(plan.observations.rbegin(), plan.observations.rend(), [&](const auto & item) {
            return item.source == step.tool_call->name && item.id.rfind("tool:" + step.id + ":", 0) == 0;
        });
        if (latest == plan.observations.rend() || latest->id != recorded->evidence_ids.front()) continue;
        const auto failure = nlohmann::ordered_json::parse(latest->summary, nullptr, false);
        if (!failure.is_object() || !failure.contains("failure") || !failure["failure"].is_object() ||
                !failure["failure"].contains("step_id") || !failure["failure"]["step_id"].is_string() ||
                !failure["failure"].contains("tool") || !failure["failure"]["tool"].is_string() ||
                failure["failure"].value("step_id", std::string{}) != step.id ||
                failure["failure"].value("tool", std::string{}) != step.tool_call->name) continue;
        const auto transition_id = value["transition_id"].get<std::string>();
        const auto terminal_state = value["terminal_state"].get<std::string>();
        const auto prefix = transition_id + "@" + terminal_state + "->";
        if (std::any_of(plan.workflow_continuation_history.begin(), plan.workflow_continuation_history.end(),
                [&](const auto & history) {
                    return history.source_observation_id == recorded->id &&
                        history.source_terminal_state == terminal_state &&
                        (history.link_id.rfind(prefix, 0) == 0 ||
                         history.link_id.rfind("@" + terminal_state + "->", 0) == 0);
                })) return true;
    }
    return false;
}

bool common_plan_workflow_terminal_choice_observation(
        const common_plan_state & plan, const std::string & transition_id,
        const std::string & choice_id, const std::string & source_step_id,
        const std::string & source_observation_id, common_plan_observation & result,
        std::string & error) {
    if (!plan.workflow_definition || !plan.route_binding || !terminal_choice_valid(plan, *plan.workflow_definition,
            common_plan_workflow_completed_state(plan, *plan.workflow_definition), transition_id, choice_id,
            source_step_id, source_observation_id)) {
        error = "terminal workflow choice is not grounded in the active route observation"; return false;
    }
    result = {};
    result.id = "workflow_terminal_choice:" + source_step_id + ":" + transition_id;
    result.source = "workflow_terminal_choice";
    const auto & transition = *std::find_if(plan.workflow_definition->transitions.begin(),
        plan.workflow_definition->transitions.end(), [&](const auto & item) { return item.id == transition_id; });
    const auto & choice = *std::find_if(transition.choices.begin(), transition.choices.end(),
        [&](const auto & item) { return item.id == choice_id; });
    nlohmann::ordered_json payload{{"transition_id", transition_id}, {"choice_id", choice_id},
        {"source_step_id", source_step_id}, {"host_validated", true}, {"terminal_state", choice.to_state},
        {"choice_description", choice.description}};
    attach_entry_handoff(plan, payload);
    result.summary = payload.dump();
    result.confidence = 1.0f;
    result.evidence_ids = {source_observation_id};
    const auto source = std::find_if(plan.observations.begin(), plan.observations.end(),
        [&](const auto & observation) { return observation.id == source_observation_id; });
    if (source != plan.observations.end()) {
        result.resource_refs = source->resource_refs;
        result.dataset_refs = source->dataset_refs;
    }
    error.clear(); return true;
}

bool common_plan_workflow_is_terminal(
        const common_plan_state & plan,
        const common_plan_workflow_definition & definition) {
    const auto state = common_plan_workflow_completed_state(plan, definition);
    return !state.empty() && std::find(definition.terminal_states.begin(),
        definition.terminal_states.end(), state) != definition.terminal_states.end();
}

bool common_plan_workflow_choice_available(const common_plan_observation & observation,
        const common_plan_workflow_transition::choice & choice) {
    if (!common_plan_workflow_observation_is_usable(observation)) return false;
    const auto value = nlohmann::ordered_json::parse(observation.summary, nullptr, false);
    return std::all_of(choice.required_observation_json_pointers.begin(),
        choice.required_observation_json_pointers.end(), [&](const auto & path) {
            if (path.empty() || path.front() != '/') return false;
            try {
                const nlohmann::ordered_json::json_pointer pointer(path);
                return value.contains(pointer) && !value.at(pointer).is_null() &&
                    (!value.at(pointer).is_string() || !value.at(pointer).get<std::string>().empty());
            } catch (const std::exception &) { return false; }
        });
}
