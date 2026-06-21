#include "agent/agent-bootstrap.h"
#include "agent/agent-package-json.h"
#include "agent/blueprint-selector.h"
#include "memory/memory-in-memory.h"
#include "plan/plan-blueprint.h"
#include "plan/plan-in-memory.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

class fixed_selector final : public common_blueprint_selector {
public:
    int calls = 0;
    common_blueprint_selection select(const common_agent_request &, const std::vector<common_blueprint_candidate> &, std::string & error) override {
        ++calls; error.clear();
        common_blueprint_selection selection;
        selection.decision = common_blueprint_selection_decision::instantiate;
        selection.logical_id = "repository-change";
        selection.confidence = 0.9f;
        return selection;
    }
};

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
    // Package export must list bootstrap procedures in their package scope,
    // which is project here rather than the normal session retrieval default.
    common_memory_query bootstrap_query;
    bootstrap_query.scope = common_memory_scope::project;
    bootstrap_query.namespace_id = config.namespace_id;
    bootstrap_query.project_id = config.project_id;
    const auto bootstrap_procedures = memory.list(bootstrap_query, error);
    assert(error.empty() && bootstrap_procedures.size() == first.installed_memory_ids.size());
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

    std::string package_json;
    const auto default_package = common_agent_default_bootstrap_package();
    assert(common_agent_package_to_json(default_package, package_json, error));
    common_agent_bootstrap_package parsed_package;
    assert(common_agent_package_parse_json(package_json, parsed_package, error));
    assert(parsed_package.procedures.size() == default_package.procedures.size());
    assert(parsed_package.blueprints.size() == default_package.blueprints.size());
    assert(common_agent_package_parse_json(R"({"schema_version":1,"name":"forward-compatible","version":"v1","procedures":[],"blueprints":[],"future_section":{"ignored":true}})", parsed_package, error));

    fixed_selector selector;
    common_blueprint_selection_config selection_config;
    selection_config.task_plan_id = "selected-instance";
    selection_config.session_id = config.session_id;
    selection_config.scope = common_plan_scope::project;
    selection_config.now = 44;
    common_blueprint_selection_result selection_result;
    common_agent_request selection_request;
    assert(common_agent_select_and_instantiate_blueprint(plans, selection_request, selector,
        {{"repository-change", first.installed_blueprint_ids.front(), "repository work"}}, selection_config, selection_result, error));
    assert(selection_result.outcome == common_blueprint_selection_outcome::instantiated && selector.calls == 1);
    assert(common_agent_select_and_instantiate_blueprint(plans, selection_request, selector,
        {{"repository-change", first.installed_blueprint_ids.front(), "repository work"}}, selection_config, selection_result, error));
    assert(selection_result.outcome == common_blueprint_selection_outcome::resumed && selector.calls == 1);

    common_agent_bootstrap_package custom;
    custom.name = "custom";
    custom.version = "v1";
    custom.procedures.push_back({"review-loop", "Review the implementation against its acceptance criteria before completion.", "review-loop"});
    common_agent_bootstrap_blueprint custom_blueprint;
    custom_blueprint.id = "review";
    custom_blueprint.goal = "Review a change";
    custom_blueprint.success_criteria = "Review findings are evidence-backed.";
    custom_blueprint.steps.push_back({"inspect", "Inspect", "Inspect the change."});
    custom_blueprint.constraints.push_back({"evidence", "Use inspection evidence.", true});
    custom_blueprint.assumptions.push_back({"repository", "The requested change is in the current repository.", 0.8f, true, {}});
    custom.blueprints.push_back(custom_blueprint);
    assert(common_agent_package_to_json(custom, package_json, error));
    assert(common_agent_package_parse_json(package_json, parsed_package, error));
    assert(parsed_package.blueprints.size() == 1);
    assert(parsed_package.blueprints.front().constraints.size() == 1);
    assert(parsed_package.blueprints.front().assumptions.size() == 1);
    assert(parsed_package.blueprints.front().assumptions.front().valid);
    common_agent_bootstrap_result custom_result;
    assert(common_agent_install_bootstrap_package(memory, plans, config, custom, embed, custom_result, error));
    assert(custom_result.installed_memory_ids.size() == 1 && custom_result.installed_blueprint_ids.size() == 1);

    common_plan_state incompatible = *blueprint;
    incompatible.id = "incompatible-task";
    assert(plans.create(incompatible, error));
    selection_config.task_plan_id = incompatible.id;
    assert(common_agent_select_and_instantiate_blueprint(plans, selection_request, selector,
        {{"repository-change", first.installed_blueprint_ids.front(), "repository work"}}, selection_config, selection_result, error));
    assert(selection_result.outcome == common_blueprint_selection_outcome::failed_safely);

    common_explicit_blueprint_selector explicit_selector("repository-change");
    selection_config.task_plan_id = "explicit-instance";
    assert(common_agent_select_and_instantiate_blueprint(plans, selection_request, explicit_selector,
        {{"repository-change", first.installed_blueprint_ids.front(), "repository work"}}, selection_config, selection_result, error));
    assert(selection_result.outcome == common_blueprint_selection_outcome::instantiated);

    common_agent_bootstrap_config invalid;
    invalid.namespace_id = "local";
    common_agent_bootstrap_result invalid_result;
    assert(!common_agent_install_default_bootstrap(memory, plans, invalid, embed, invalid_result, error));
    return 0;
}
