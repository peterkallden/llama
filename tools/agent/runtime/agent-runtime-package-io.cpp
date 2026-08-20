#include "agent-runtime-package-io.h"

#include "agent/agent-package-json.h"
#include "tools/agent/cli/agent-cli-scope.h"

#include <fstream>
#include <sstream>

namespace {

std::string make_bootstrap_prefix(const common_agent_scope & scope) {
    return "bootstrap:" + scope.namespace_id + ":" +
        (scope.project_id.empty() ? "session:" + scope.session_id : "project:" + scope.project_id) + ":";
}

} // namespace

bool parse_plan_scope(const std::string & value, common_plan_scope & scope) {
    if (value == "turn")    { scope = common_plan_scope::turn; return true; }
    if (value == "session") { scope = common_plan_scope::session; return true; }
    if (value == "project") { scope = common_plan_scope::project; return true; }
    if (value == "global")  { scope = common_plan_scope::global; return true; }
    return false;
}

bool load_bootstrap_file(const std::string & path, common_agent_bootstrap_package & package, std::string & error) {
    std::ifstream input(path);
    if (!input) {
        error = "could not open bootstrap file: " + path;
        return false;
    }
    std::stringstream text;
    text << input.rdbuf();
    return common_agent_package_parse_json(text.str(), package, error);
}

bool export_agent_package(
        common_memory_store & memory_store,
        common_plan_store & plan_store,
        const common_agent_scope & scope,
        const std::string & output_path,
        std::string & error) {
    if (!common_cli_supports_bootstrap_package_scope(scope)) {
        error = "--agent-export currently supports only session- or project-scoped bootstrap packages";
        return false;
    }
    common_memory_query query;
    query.scope = scope.project_id.empty() ? common_memory_scope::session : common_memory_scope::project;
    query.namespace_id = scope.namespace_id;
    query.session_id = scope.session_id;
    query.project_id = scope.project_id;
    query.turn_id = scope.turn_id;
    query.global_opt_in = false;
    const auto memories = memory_store.list(query, error);
    if (!error.empty()) {
        return false;
    }
    const auto plans = plan_store.list(error);
    if (!error.empty()) {
        return false;
    }

    const std::string prefix = make_bootstrap_prefix(scope);
    common_agent_bootstrap_package package;
    package.name = "agent-export";
    package.version = "v1";

    const std::string procedure_prefix = prefix + "procedure:";
    for (const auto & memory : memories) {
        if (memory.kind != common_memory_kind::procedure || memory.id.rfind(procedure_prefix, 0) != 0) {
            continue;
        }
        package.procedures.push_back({
            memory.id.substr(procedure_prefix.size()),
            memory.content,
            memory.summary,
            memory.importance,
            memory.confidence,
        });
    }

    const std::string blueprint_prefix = prefix + "blueprint:";
    for (const auto & plan : plans) {
        if (plan.kind != common_plan_kind::blueprint || plan.id.rfind(blueprint_prefix, 0) != 0) {
            continue;
        }
        common_agent_bootstrap_blueprint blueprint;
        blueprint.id = plan.id.substr(blueprint_prefix.size());
        blueprint.purpose = plan.purpose;
        blueprint.goal = plan.goal;
        blueprint.success_criteria = plan.success_criteria;
        blueprint.required_capabilities = plan.required_capabilities;
        blueprint.steps = plan.steps;
        blueprint.constraints = plan.constraints;
        blueprint.assumptions = plan.assumptions;
        blueprint.next_action = plan.next_action;
        package.blueprints.push_back(std::move(blueprint));
    }

    std::string text;
    if (!common_agent_package_to_json(package, text, error)) {
        return false;
    }
    std::ofstream file(output_path, std::ios::binary | std::ios::trunc);
    if (!file) {
        error = "cannot open --agent-export path";
        return false;
    }
    file << text;
    if (!file) {
        error = "failed to write --agent-export package";
        return false;
    }

    error.clear();
    return true;
}
