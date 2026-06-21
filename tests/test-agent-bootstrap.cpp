#include "agent/agent-bootstrap.h"
#include "memory/memory-in-memory.h"
#include "plan/plan-blueprint.h"
#include "plan/plan-in-memory.h"

#include <cassert>

int main() {
    common_memory_in_memory_store memory;
    common_plan_in_memory_store plans;
    std::string error;
    assert(memory.open("", error));
    assert(plans.open("", error));

    common_agent_bootstrap_config config;
    config.namespace_id = "local";
    config.session_id = "session-a";
    config.project_id = "project-a";
    config.now = 42;
    auto embed = [](const std::string & text, std::vector<float> & out, std::string & error) {
        error.clear();
        out = {(float) text.size(), 1.0f};
        return true;
    };

    common_agent_bootstrap_result first;
    assert(common_agent_install_default_bootstrap(memory, plans, config, embed, first, error));
    assert(first.installed_memory_ids.size() == 4);
    assert(first.installed_blueprint_ids.size() == 2);
    const auto procedure = memory.get(first.installed_memory_ids.front(), error);
    assert(procedure && procedure->kind == common_memory_kind::procedure);
    assert(procedure->scope == common_memory_scope::project);
    assert(procedure->metadata.at("origin") == "bootstrap");
    const auto blueprint = plans.get(first.installed_blueprint_ids.front(), error);
    assert(blueprint && blueprint->kind == common_plan_kind::blueprint);
    assert(!blueprint->steps.empty() && !blueprint->steps.front().tool_call);

    common_plan_state instance;
    assert(common_plan_instantiate_blueprint(*blueprint, "instance-a", config.session_id, instance, error, common_plan_scope::project, 43));
    assert(instance.kind == common_plan_kind::task);
    assert(instance.derived_from_plan_id && *instance.derived_from_plan_id == blueprint->id);

    common_agent_bootstrap_result second;
    assert(common_agent_install_default_bootstrap(memory, plans, config, embed, second, error));
    assert(second.installed_memory_ids.empty() && second.installed_blueprint_ids.empty());
    assert(second.existing_memory_ids.size() == 4 && second.existing_blueprint_ids.size() == 2);

    common_agent_bootstrap_package custom;
    custom.name = "custom";
    custom.version = "v1";
    custom.procedures.push_back({"review-loop", "Review the implementation against its acceptance criteria before completion.", "review-loop"});
    common_agent_bootstrap_blueprint custom_blueprint;
    custom_blueprint.id = "review";
    custom_blueprint.goal = "Review a change";
    custom_blueprint.success_criteria = "Review findings are evidence-backed.";
    custom_blueprint.steps.push_back({"inspect", "Inspect", "Inspect the change."});
    custom.blueprints.push_back(custom_blueprint);
    common_agent_bootstrap_result custom_result;
    assert(common_agent_install_bootstrap_package(memory, plans, config, custom, embed, custom_result, error));
    assert(custom_result.installed_memory_ids.size() == 1 && custom_result.installed_blueprint_ids.size() == 1);

    common_agent_bootstrap_config invalid;
    invalid.namespace_id = "local";
    common_agent_bootstrap_result invalid_result;
    assert(!common_agent_install_default_bootstrap(memory, plans, invalid, embed, invalid_result, error));
    return 0;
}
