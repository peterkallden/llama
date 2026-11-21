#include "agent/agent-package-json.h"
#include "memory/memory-in-memory.h"
#include "plan/plan-in-memory.h"
#include "plan/plan-workflow-state.h"
#include "tools/agent/runtime/agent-workflow-transition.h"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

static bool smoke_source(const common_plan_workflow_definition & definition, const std::string & tool) {
    const auto entry = std::find_if(definition.transitions.begin(), definition.transitions.end(),
        [&](const auto & value) { return value.from_state == definition.start_state && value.tool_name == tool; });
    if (entry == definition.transitions.end()) return false;
    common_agent_route_candidate route;
    route.id = "hypothesis-smoke";
    route.workflow_definition = definition;
    route.resolved_tools = {tool};
    common_plan_state plan;
    plan.id = "hypothesis-smoke";
    plan.route_binding = common_plan_route_binding{};
    plan.route_binding->route_id = route.id;
    plan.workflow_definition = definition;
    common_plan_step first{"support", "Search for support", "Inspect candidate evidence"};
    first.status = common_plan_step_status::completed;
    first.tool_call = common_plan_tool_call{tool, R"({"query":"hypothesis support"})"};
    first.semantic_alias = "workflow-transition:" + entry->id;
    plan.steps.push_back(first);
    common_plan_observation observation{"tool:support:result", tool,
        R"({"ok":true,"result":{"items":[{"title":"fixture supporting perspective"}]}})", 1.0f};
    plan.observations.push_back(observation);
    bool challenge_requested = false;
    const auto continuation = make_agent_workflow_graph_continuation(route, {},
        [&](const common_plan_state &, const common_plan_observation & prior,
            const common_plan_workflow_transition & transition, const std::string &,
            std::string & arguments, std::string & error) {
            challenge_requested = transition.tool_name == tool &&
                transition.model_instruction.find("contradict") != std::string::npos &&
                prior.summary.find("supporting perspective") != std::string::npos;
            arguments = R"({"query":"hypothesis alternative evidence"})";
            error.clear();
            return challenge_requested;
        });
    common_agent_workflow_continuation steps;
    std::string error;
    if (!continuation(plan, first.id, observation, steps, error) || !challenge_requested || steps.size() != 1 ||
            !steps.front().tool_call || steps.front().tool_call->name != tool) {
        std::cerr << "challenge step failed for " << tool << ": " << error << '\n';
        return false;
    }
    steps.front().status = common_plan_step_status::completed;
    plan.steps.push_back(steps.front());
    return common_plan_workflow_is_terminal(plan, definition);
}

int main(int argc, char ** argv) {
    if (argc != 2) { std::cerr << "usage: agent-hypothesis-workflow-smoke PACKAGE.json\n"; return 2; }
    std::ifstream input(argv[1]);
    if (!input) { std::cerr << "cannot open workflow package\n"; return 1; }
    std::stringstream text; text << input.rdbuf();
    common_agent_bootstrap_package package; std::string error;
    if (!common_agent_package_parse_json(text.str(), package, error)) { std::cerr << error << '\n'; return 1; }
    const auto workflow = std::find_if(package.workflows.begin(), package.workflows.end(), [](const auto & value) {
        return value.definition.workflow_ref == "workflow://research/hypothesis-inquiry";
    });
    const auto blueprint = std::find_if(package.blueprints.begin(), package.blueprints.end(), [](const auto & value) {
        return value.id == "hypothesis-inquiry-v1";
    });
    if (workflow == package.workflows.end() || blueprint == package.blueprints.end()) return 1;
    common_memory_in_memory_store memory;
    common_plan_in_memory_store plans;
    if (!memory.open("", error) || !plans.open("", error)) return 1;
    common_agent_bootstrap_config config;
    config.namespace_id = "smoke";
    config.session_id = "hypothesis";
    common_agent_bootstrap_result installed;
    if (!common_agent_install_bootstrap_package(memory, plans, config, package,
            [](const std::string &, std::vector<float> & embedding, std::string & failure) {
                embedding = {1.0f, 0.0f}; failure.clear(); return true;
            }, installed, error)) { std::cerr << error << '\n'; return 1; }
    const auto id = std::find_if(installed.installed_blueprint_ids.begin(), installed.installed_blueprint_ids.end(),
        [](const auto & value) { return value.find("hypothesis-inquiry-v1") != std::string::npos; });
    if (id == installed.installed_blueprint_ids.end()) return 1;
    common_blueprint_candidate candidate;
    candidate.logical_id = *id; candidate.persisted_id = *id;
    common_agent_request request;
    request.prompt = "Test this hypothesis against supporting and opposing evidence.";
    common_agent_scope scope;
    scope.namespace_id = "smoke"; scope.session_id = "hypothesis";
    for (const auto & tool : {"web.search", "resource.read", "dataset.sample", "data.query", "openalex.listWorks", "eurostat.getData"}) {
        const std::string capability = std::string(tool).find('.') != std::string::npos &&
            (std::string(tool).find("openalex.") == 0 || std::string(tool).find("eurostat.") == 0)
            ? "openapi.read" : tool;
        common_agent_runtime_tooling tooling;
        tooling.profile_tools_active = true;
        tooling.capabilities = {capability};
        tooling.capability_tools[capability] = {tool};
        tooling.tools.push_back({tool, tool, R"({"type":"object"})"});
        common_agent_route_catalog catalog;
        if (!compile_agent_route_catalog(request, plans, scope, {candidate}, tooling, "", catalog, error)) {
            std::cerr << "route compile failed for " << tool << ": " << error << '\n'; return 1;
        }
        const auto selected = std::find_if(catalog.candidates.begin(), catalog.candidates.end(), [](const auto & route) {
            return route.workflow && route.workflow->workflow_ref == "workflow://research/hypothesis-inquiry";
        });
        if (selected == catalog.candidates.end() || !selected->workflow_definition ||
                std::find(selected->resolved_tools.begin(), selected->resolved_tools.end(), tool) == selected->resolved_tools.end()) {
            std::cerr << "hypothesis route not eligible for " << tool << '\n'; return 1;
        }
        if (!smoke_source(workflow->definition, tool)) return 1;
        std::cout << "hypothesis_workflow=" << tool << " bootstrap_route=ok support_to_challenge=ok\n";
    }
    return 0;
}
