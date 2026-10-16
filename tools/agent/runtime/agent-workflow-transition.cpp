#include "agent-workflow-transition.h"
#include "agent/tooling/contracts/schema-contract.h"
#include "agent/artifact/artifact-contract.h"
#include "plan/plan-workflow-state.h"

#include <algorithm>
#include <limits>
#include <nlohmann/json.hpp>
#include <queue>
#include <set>
#include <unordered_map>

using json = nlohmann::ordered_json;

namespace {

bool terminal(const common_plan_workflow_definition & definition, const std::string & state) {
    return std::find(definition.terminal_states.begin(), definition.terminal_states.end(), state) !=
        definition.terminal_states.end();
}

bool replace_previous(json & value, const std::string & step_id, const std::string & choice) {
    if (value.is_array()) {
        for (auto & item : value) if (!replace_previous(item, step_id, choice)) return false;
        return true;
    }
    if (value.is_string() && !choice.empty()) {
        auto text = value.get<std::string>();
        size_t offset = 0;
        while ((offset = text.find("$choice", offset)) != std::string::npos) {
            text.replace(offset, std::string("$choice").size(), choice);
            offset += choice.size();
        }
        value = std::move(text);
        return true;
    }
    if (!value.is_object()) return true;
    if (value.value("$from_step", std::string{}) == "$previous") value["$from_step"] = step_id;
    if (value.is_object() && value.value("$choice", false)) {
        value = choice;
        return true;
    }
    for (auto & item : value.items()) if (!replace_previous(item.value(), step_id, choice)) return false;
    return true;
}

bool candidate_ids(const common_plan_observation & observation,
        const std::string & pointer, const std::string & id_pointer,
        std::set<std::string> & ids, std::string & error) {
    ids.clear();
    const auto value = json::parse(observation.summary, nullptr, false);
    if (!value.is_object() && !value.is_array()) { error = "workflow choice observation is not JSON"; return false; }
    if (!pointer.empty()) {
        const json::json_pointer location(pointer);
        if (!value.contains(location)) { error = "workflow choice candidate pointer is absent"; return false; }
        const auto & selected = value.at(location);
        if (!selected.is_array()) { error = "workflow choice candidates are not an array"; return false; }
        std::string columns_pointer = pointer;
        const auto last_separator = columns_pointer.rfind('/');
        columns_pointer = last_separator == std::string::npos || last_separator == 0
            ? std::string{} : columns_pointer.substr(0, last_separator);
        if (!columns_pointer.empty()) columns_pointer += "/columns";
        const json::json_pointer columns_location(columns_pointer);
        const json * columns = !columns_pointer.empty() && value.contains(columns_location)
            ? &value.at(columns_location) : nullptr;
        const size_t id_column = [&]() {
            if (columns == nullptr || !columns->is_array() || id_pointer.size() < 2 ||
                    id_pointer.front() != '/' || id_pointer.find('/', 1) != std::string::npos) {
                return std::numeric_limits<size_t>::max();
            }
            const std::string field = id_pointer.substr(1);
            for (size_t index = 0; index < columns->size(); ++index) {
                if ((*columns)[index].is_string() && (*columns)[index].get<std::string>() == field) return index;
            }
            return std::numeric_limits<size_t>::max();
        }();
        for (const auto & candidate : selected) {
            const json * identity = nullptr;
            if (candidate.is_object()) {
                try { identity = &candidate.at(json::json_pointer(id_pointer)); }
                catch (const std::exception &) {}
            } else if (candidate.is_array() && id_column < candidate.size()) {
                identity = &candidate[id_column];
            }
            if (identity && identity->is_string() && !identity->get<std::string>().empty()) {
                ids.insert(identity->get<std::string>());
            }
        }
    }
    if (ids.empty()) { error = "workflow choice has no candidate ids"; return false; }
    return true;
}

bool json_pointer_is_present(const json & value, const std::string & pointer) {
    if (pointer.empty() || pointer.front() != '/') return false;
    const json::json_pointer location(pointer);
    if (!value.contains(location)) return false;
    const auto & selected = value.at(location);
    return selected.is_string() && !selected.get<std::string>().empty();
}

const common_plan_observation * binding_observation(
        const common_plan_state & plan,
        const common_plan_observation & current,
        const std::string & source_transition_id) {
    if (source_transition_id.empty() || source_transition_id == "$previous") {
        return common_plan_workflow_observation_is_usable(current) ? &current : nullptr;
    }
    // A model-choice transition has no tool step of its own. Its verified
    // candidate is persisted as a workflow_choice observation after the
    // immediately following tool succeeds, so later graph transitions must
    // resolve explicit bindings from that host-validated observation.
    return common_plan_workflow_observation(plan, source_transition_id);
}

bool resolve_argument_bindings(json & arguments,
        const common_plan_state & plan,
        const common_plan_workflow_transition & transition,
        const common_plan_observation & observation,
        std::string & error) {
    if (transition.argument_bindings.empty()) return true;
    const auto value = json::parse(observation.summary, nullptr, false);
    if (!value.is_object() && !value.is_array()) {
        error = "workflow argument binding observation is not structured JSON";
        return false;
    }
    for (const auto & binding : transition.argument_bindings) {
        const auto * source_observation = binding_observation(
            plan, observation, binding.source_transition_id);
        if (source_observation == nullptr) {
            if (binding.required) {
                error = "required workflow source transition has no verified observation: " +
                    binding.source_transition_id;
                return false;
            }
            continue;
        }
        const auto source_value_json = json::parse(source_observation->summary, nullptr, false);
        if (!source_value_json.is_object() && !source_value_json.is_array()) {
            if (binding.required) { error = "workflow binding source observation is not JSON"; return false; }
            continue;
        }
        const json::json_pointer source(binding.source_json_pointer);
        const json::json_pointer target(binding.target_json_pointer);
        if (!source_value_json.contains(source)) {
            if (binding.required) {
                error = "required workflow argument source is absent: " + binding.source_json_pointer;
                return false;
            }
            continue;
        }
        const auto & source_value = source_value_json.at(source);
        if (source_value.is_null() || (source_value.is_string() && source_value.get<std::string>().empty())) {
            if (binding.required) {
                error = "required workflow argument source must not be null or an empty string";
                return false;
            }
            continue;
        }
        arguments[target] = source_value;
    }
    return true;
}

common_plan_step make_tool_step(const common_plan_state & plan,
        const common_plan_workflow_transition & transition,
        const std::string & completed_step_id,
        const std::string & choice,
        const common_plan_observation & observation,
        std::string & error) {
    const auto arguments = json::parse(transition.arguments_template_json, nullptr, false);
    if (!arguments.is_object()) { error = "workflow transition arguments are not an object"; return {}; }
    auto resolved = arguments;
    if (!replace_previous(resolved, completed_step_id, choice)) { error = "workflow transition arguments are invalid"; return {}; }
    if (!resolve_argument_bindings(resolved, plan, transition, observation, error)) return {};
    common_plan_step step;
    step.id = plan.id + ":workflow:" + transition.id + ":" + std::to_string(plan.version);
    step.title = transition.tool_name;
    step.objective = "Execute host-approved workflow transition " + transition.id + ".";
    step.intended_contribution = step.objective;
    step.status = common_plan_step_status::pending;
    step.optional = transition.optional;
    step.mode = common_plan_step_mode::tool;
    step.selected_tool = transition.tool_name;
    step.tool_call = common_plan_tool_call{transition.tool_name, resolved.dump()};
    step.semantic_alias = "workflow-transition:" + transition.id;
    return step;
}

} // namespace

const common_plan_workflow_transition * common_agent_workflow_next_transition(
        const common_plan_workflow_definition & definition,
        const std::string & state) {
    if (state.empty() || terminal(definition, state)) return nullptr;
    // Dijkstra is A* with a zero admissible heuristic.  It keeps V1 graph
    // selection deterministic while the domain supplies no richer heuristic.
    struct node { float cost; std::string state; std::string first; };
    const auto compare = [](const node & a, const node & b) {
        return a.cost == b.cost ? a.first > b.first : a.cost > b.cost;
    };
    std::priority_queue<node, std::vector<node>, decltype(compare)> queue(compare);
    std::unordered_map<std::string, float> best;
    queue.push({0.0f, state, {}}); best[state] = 0.0f;
    while (!queue.empty()) {
        const auto current = queue.top(); queue.pop();
        if (current.cost > best[current.state] + 1e-6f) continue;
        if (terminal(definition, current.state) && !current.first.empty()) {
            return &*std::find_if(definition.transitions.begin(), definition.transitions.end(),
                [&](const auto & transition) { return transition.id == current.first; });
        }
        for (const auto & transition : definition.transitions) {
            if (transition.from_state != current.state) continue;
            const auto next_cost = current.cost + transition.cost;
            const auto prior = best.find(transition.to_state);
            if (prior != best.end() && next_cost >= prior->second - 1e-6f) continue;
            best[transition.to_state] = next_cost;
            queue.push({next_cost, transition.to_state,
                current.first.empty() ? transition.id : current.first});
        }
    }
    return nullptr;
}

bool common_agent_workflow_static_tool_arguments(
        const common_plan_workflow_transition & transition,
        const common_chat_tool & tool,
        std::string & normalized_arguments_json,
        std::string & error) {
    normalized_arguments_json.clear();
    error.clear();
    if (transition.kind != common_plan_workflow_transition_kind::tool ||
            transition.tool_name != tool.name || !transition.argument_bindings.empty()) {
        return false;
    }
    const auto arguments = json::parse(transition.arguments_template_json, nullptr, false);
    if (!arguments.is_object()) {
        error = "workflow transition arguments are not an object";
        return false;
    }
    std::string validation_error;
    if (common_schema_normalize_and_validate_object(
            arguments.dump(), tool.parameters, normalized_arguments_json, validation_error)) {
        return !transition.model_supplies_arguments;
    }

    // A partial static template can still be completed by the model, but its
    // provided values must independently satisfy the tool's schema. Remove
    // only `required` for this check so malformed fixed values cannot be
    // silently replaced by model-generated arguments.
    auto partial_schema = json::parse(tool.parameters, nullptr, false);
    if (partial_schema.is_object()) {
        partial_schema.erase("required");
        std::string partial_normalized;
        std::string partial_error;
        if (!common_schema_normalize_and_validate_object(
                arguments.dump(), partial_schema.dump(), partial_normalized, partial_error)) {
            error = "workflow transition static arguments violate " + tool.name + " schema: " + partial_error;
            return false;
        }
        normalized_arguments_json = std::move(partial_normalized);
        return false;
    }
    error = "workflow transition static arguments violate " + tool.name + " schema: " + validation_error;
    normalized_arguments_json.clear();
    return false;
}

common_agent_workflow_continuation_provider make_agent_workflow_graph_continuation(
        common_agent_route_candidate route,
        common_agent_workflow_choice_selector select_choice) {
    return [route = std::move(route), select_choice = std::move(select_choice)](
            const common_plan_state & plan, const std::string & completed_step_id,
            const common_plan_observation & observation,
            std::vector<common_plan_step> & steps, std::string & error) {
        steps.clear(); error.clear();
        if (!route.workflow_definition || route.workflow_definition->transitions.empty() ||
                !plan.workflow_definition || !plan.route_binding ||
                plan.route_binding->route_id != route.id) return true;
        const auto & definition = *route.workflow_definition;
        std::string state = common_plan_workflow_completed_state(plan, definition);
        const auto * next = common_agent_workflow_next_transition(definition, state);
        if (next == nullptr) return true;
        if ((next->kind == common_plan_workflow_transition_kind::model_choice ||
                !next->argument_bindings.empty()) && !common_plan_workflow_observation_is_usable(observation)) {
            error = "workflow continuation requires a successful structured host observation";
            return false;
        }
        std::string choice;
        std::string aliases;
        if (next->kind == common_plan_workflow_transition_kind::model_choice) {
            if (!select_choice) { error = "workflow graph needs a model choice selector"; return false; }
            std::set<std::string> ids;
            if (!next->choices.empty()) {
                const auto observed = json::parse(observation.summary, nullptr, false);
                for (const auto & option : next->choices) {
                    const bool available = std::all_of(option.required_observation_json_pointers.begin(),
                        option.required_observation_json_pointers.end(), [&](const auto & pointer) {
                            return json_pointer_is_present(observed, pointer);
                        });
                    if (available) ids.insert(option.id);
                }
                if (ids.empty()) { error = "workflow has no available fixed choices for this observation"; return false; }
            } else if (!candidate_ids(observation, next->candidate_json_pointer,
                    next->candidate_id_json_pointer, ids, error)) {
                return false;
            }
            if (!select_choice(observation, *next, choice, error)) return false;
            if (!ids.count(choice)) { error = "workflow choice selected an id outside the host observation"; return false; }
            if (!next->choices.empty()) {
                const auto selected = std::find_if(next->choices.begin(), next->choices.end(),
                    [&](const auto & option) { return option.id == choice; });
                if (selected == next->choices.end()) { error = "workflow fixed choice was not declared"; return false; }
                state = selected->to_state;
                aliases = next->id + "=" + choice + ",";
            } else {
                state = next->to_state;
                // Persist the selected host-validated candidate in the step
                // alias as well as the workflow_choice observation. State
                // replay needs the choice token to advance candidate-backed
                // model_choice transitions after resume.
                aliases = next->id + "=" + choice + ",";
            }
            next = common_agent_workflow_next_transition(definition, state);
            if (next == nullptr) return true;
        }
        if (next->kind == common_plan_workflow_transition_kind::artifact_compose) {
            common_agent_artifact_profile profile;
            if (!common_agent_artifact_profile_from_json(next->artifact_profile_json, profile, error)) {
                if (error.empty()) error = "artifact profile was not host-materialized";
                return false;
            }
            const auto section = std::find_if(profile.sections.begin(), profile.sections.end(), [&](const auto & candidate) {
                return candidate.id == next->artifact_section_id;
            });
            if (section == profile.sections.end()) { error = "artifact transition references an unknown profile section"; return false; }
            common_plan_step step;
            step.id = plan.id + ":artifact:" + profile.id + ":" + section->id + ":" + std::to_string(plan.version);
            step.title = section->title;
            step.objective = section->purpose;
            step.intended_contribution = section->purpose;
            step.status = common_plan_step_status::pending;
            step.mode = common_plan_step_mode::artifact_compose;
            step.depends_on.push_back(completed_step_id);
            step.required_evidence = section->required_observation_sources;
            step.semantic_alias = "workflow-transition:" + next->id;
            step.artifact_binding = common_plan_artifact_binding{next->id,profile.id,profile.revision,section->id,"snapshot:"+plan.id};
            steps.push_back(std::move(step));
            error.clear();
            return true;
        }
        if (next->kind != common_plan_workflow_transition_kind::tool ||
                std::find(route.resolved_tools.begin(), route.resolved_tools.end(), next->tool_name) == route.resolved_tools.end()) {
            error = "workflow graph proposed a tool outside the selected route";
            return false;
        }
        auto step = make_tool_step(plan, *next, completed_step_id, choice, observation, error);
        if (!error.empty()) return false;
        step.depends_on.push_back(completed_step_id);
        step.semantic_alias = "workflow-transition:" + aliases + next->id;
        steps.push_back(std::move(step));
        return true;
    };
}
