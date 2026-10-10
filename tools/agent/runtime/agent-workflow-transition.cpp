#include "agent-workflow-transition.h"

#include <algorithm>
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

std::string state_after_completed_transitions(
        const common_plan_state & plan,
        const common_plan_workflow_definition & definition) {
    std::string state = definition.start_state;
    for (const auto & step : plan.steps) {
        if (step.status != common_plan_step_status::completed || !step.semantic_alias ||
                step.semantic_alias->rfind("workflow-transition:", 0) != 0) continue;
        const auto ids = step.semantic_alias->substr(std::string("workflow-transition:").size());
        size_t begin = 0;
        while (begin <= ids.size()) {
            const auto end = ids.find(',', begin);
            const auto id = ids.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
            const auto found = std::find_if(definition.transitions.begin(), definition.transitions.end(),
                [&](const auto & transition) { return transition.id == id && transition.from_state == state; });
            if (found != definition.transitions.end()) state = found->to_state;
            if (end == std::string::npos) break;
            begin = end + 1;
        }
    }
    return state;
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
        const std::string & pointer, std::set<std::string> & ids, std::string & error) {
    ids.clear();
    const auto value = json::parse(observation.summary, nullptr, false);
    if (!value.is_object() && !value.is_array()) { error = "workflow choice observation is not JSON"; return false; }
    const json::json_pointer location(pointer);
    if (!value.contains(location)) { error = "workflow choice candidate pointer is absent"; return false; }
    const auto & selected = value.at(location);
    if (!selected.is_array()) { error = "workflow choice candidates are not an array"; return false; }
    for (const auto & candidate : selected) {
        if (candidate.is_object() && candidate.contains("id") && candidate["id"].is_string()) {
            ids.insert(candidate["id"].get<std::string>());
        }
    }
    if (ids.empty()) { error = "workflow choice has no candidate ids"; return false; }
    return true;
}

common_plan_step make_tool_step(const common_plan_state & plan,
        const common_plan_workflow_transition & transition,
        const std::string & completed_step_id,
        const std::string & choice,
        std::string & error) {
    const auto arguments = json::parse(transition.arguments_template_json, nullptr, false);
    if (!arguments.is_object()) { error = "workflow transition arguments are not an object"; return {}; }
    auto resolved = arguments;
    if (!replace_previous(resolved, completed_step_id, choice)) { error = "workflow transition arguments are invalid"; return {}; }
    common_plan_step step;
    step.id = plan.id + ":workflow:" + transition.id + ":" + std::to_string(plan.version);
    step.title = transition.tool_name;
    step.objective = "Execute host-approved workflow transition " + transition.id + ".";
    step.intended_contribution = step.objective;
    step.status = common_plan_step_status::pending;
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
        std::string state = state_after_completed_transitions(plan, definition);
        const auto * next = common_agent_workflow_next_transition(definition, state);
        if (next == nullptr) return true;
        std::string choice;
        std::string aliases;
        if (next->kind == common_plan_workflow_transition_kind::model_choice) {
            if (!select_choice) { error = "workflow graph needs a model choice selector"; return false; }
            std::set<std::string> ids;
            if (!candidate_ids(observation, next->candidate_json_pointer, ids, error)) return false;
            if (!select_choice(observation, *next, choice, error)) return false;
            if (!ids.count(choice)) { error = "workflow choice selected an id outside the host observation"; return false; }
            state = next->to_state;
            aliases = next->id + ",";
            next = common_agent_workflow_next_transition(definition, state);
            if (next == nullptr) return true;
        }
        if (next->kind != common_plan_workflow_transition_kind::tool ||
                std::find(route.resolved_tools.begin(), route.resolved_tools.end(), next->tool_name) == route.resolved_tools.end()) {
            error = "workflow graph proposed a tool outside the selected route";
            return false;
        }
        auto step = make_tool_step(plan, *next, completed_step_id, choice, error);
        if (!error.empty()) return false;
        step.depends_on.push_back(completed_step_id);
        step.semantic_alias = "workflow-transition:" + aliases + next->id;
        steps.push_back(std::move(step));
        return true;
    };
}
