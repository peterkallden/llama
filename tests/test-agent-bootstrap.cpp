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
    std::vector<common_blueprint_candidate> seen;
    common_blueprint_selection select(const common_agent_request &, const std::vector<common_blueprint_candidate> & candidates, std::string & error) override {
        ++calls; error.clear();
        seen = candidates;
        common_blueprint_selection selection;
        selection.decision = common_blueprint_selection_decision::instantiate;
        selection.logical_id = "repository-change";
        selection.confidence = 0.9f;
        return selection;
    }
};

class declining_selector final : public common_blueprint_selector {
public:
    common_blueprint_selection select(const common_agent_request &, const std::vector<common_blueprint_candidate> &, std::string & error) override {
        error.clear();
        return {common_blueprint_selection_decision::none, std::nullopt, 0.0f, "model declined"};
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
    assert(blueprint->purpose == "Safely modify a repository while preserving intended behavior.");
    assert(blueprint->constraints.size() == 2 && blueprint->assumptions.size() == 1);
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
    selection_request.namespace_id = config.namespace_id;
    selection_request.session_id = config.session_id;
    selection_request.project_id = config.project_id;
    assert(common_agent_select_and_instantiate_blueprint(plans, selection_request, selector,
        {{"repository-change", first.installed_blueprint_ids.front(), "repository work",
          "Safely modify the repository.", "Implement the requested change.", "The change is verified.",
          {{"scope", "Keep the change bounded.", true}},
          {{"workspace", "A controlled workspace is available.", 0.9f, true, {}}},
          {"inspect affected code", "run focused tests"}}}, selection_config, selection_result, error));
    assert(selection_result.outcome == common_blueprint_selection_outcome::instantiated && selector.calls == 1);
    assert(selector.seen.size() == 1 && selector.seen.front().purpose == "Safely modify the repository.");
    assert(selector.seen.front().constraints.size() == 1 && selector.seen.front().assumptions.size() == 1);
    assert(selector.seen.front().contributions.size() == 2);
    selection_config.task_plan_id = "selected-instance";
    assert(common_agent_select_and_instantiate_blueprint(plans, selection_request, selector, {}, selection_config, selection_result, error));
    assert(selection_result.outcome == common_blueprint_selection_outcome::resumed && selector.calls == 1);
    const auto selected_instance = plans.get(selection_config.task_plan_id, error);
    assert(selected_instance && selected_instance->namespace_id == config.namespace_id && selected_instance->project_id == config.project_id);
    assert(common_plan_scope_matches(*selected_instance, common_plan_scope::project,
        config.namespace_id, config.session_id, config.project_id, {}));

    common_plan_state invalid_assumption = *blueprint;
    invalid_assumption.id = "invalid-assumption-blueprint";
    invalid_assumption.assumptions.push_back({"workspace", "A controlled workspace is available.", 0.1f, false, {}});
    assert(plans.create(invalid_assumption, error));
    selection_config.task_plan_id = "invalid-assumption-instance";
    assert(common_agent_select_and_instantiate_blueprint(plans, selection_request, selector,
        {{"repository-change", invalid_assumption.id, "repository work"}}, selection_config, selection_result, error));
    assert(selection_result.outcome == common_blueprint_selection_outcome::declined && selector.calls == 1);
    assert(selection_result.candidate_count == 1 && selection_result.eligible_count == 0 && selection_result.rejections.size() == 1);
    common_plan_state capability_blueprint = *blueprint;
    capability_blueprint.id = "capability-blueprint";
    capability_blueprint.required_capabilities = {"development.build"};
    assert(plans.create(capability_blueprint, error));
    selection_config.task_plan_id = "missing-capability-instance";
    selection_config.capabilities_resolved = true;
    selection_config.available_capabilities = {"workspace.read"};
    assert(common_agent_select_and_instantiate_blueprint(plans, selection_request, selector,
        {{"capability", capability_blueprint.id, "build work", "Build", "Build", "Build succeeds.", {}, {}, {}, {"development.build"}}},
        selection_config, selection_result, error));
    assert(selection_result.outcome == common_blueprint_selection_outcome::declined && selector.calls == 1);
    assert(selection_result.rejections.size() == 1 && selection_result.rejections.front().reason == "required host capability is unavailable");
    common_plan_state blocked_blueprint = *blueprint;
    blocked_blueprint.id = "blocked-blueprint";
    blocked_blueprint.constraints = {{"host-write", "requires host-approved writes", true}};
    assert(plans.create(blocked_blueprint, error));
    selection_config.capabilities_resolved = false;
    selection_config.available_capabilities.clear();
    selection_config.blocked_constraint_ids = {"host-write"};
    assert(common_agent_select_and_instantiate_blueprint(plans, selection_request, selector,
        {{"blocked", blocked_blueprint.id, "blocked work"}}, selection_config, selection_result, error));
    assert(selection_result.outcome == common_blueprint_selection_outcome::declined &&
        selection_result.rejections.size() == 1 &&
        selection_result.rejections.front().reason == "hard constraint conflicts with host policy");
    selection_config.blocked_constraint_ids.clear();
    selection_config.capabilities_resolved = false;
    selection_config.available_capabilities.clear();
    selection_config.task_plan_id = "selected-instance";
    assert(common_agent_select_and_instantiate_blueprint(plans, selection_request, selector,
        {{"repository-change", first.installed_blueprint_ids.front(), "repository work"}}, selection_config, selection_result, error));
    assert(selection_result.outcome == common_blueprint_selection_outcome::resumed && selector.calls == 1);

    declining_selector declining;
    selection_config.task_plan_id = "fallback-instance";
    selection_request.prompt = "Diagnose unexpected behavior in an agent regression.";
    assert(common_agent_select_and_instantiate_blueprint(plans, selection_request, declining,
        {{"repository-change", first.installed_blueprint_ids.front(), "Implement or modify code in a repository."},
         {"agent-regression", first.installed_blueprint_ids.back(), "Diagnose unexpected behavior at the plan, memory, tool, or reflection boundary."}},
        selection_config, selection_result, error));
    assert(selection_result.outcome == common_blueprint_selection_outcome::instantiated);
    assert(selection_result.logical_id && *selection_result.logical_id == "agent-regression");
    assert(selection_result.reason.rfind("native keyword fallback", 0) == 0);

    common_plan_state build_blueprint = *blueprint;
    build_blueprint.id = "build-repair-blueprint";
    build_blueprint.purpose = "Diagnose and repair repository build failures.";
    build_blueprint.goal = "Repair the failing build.";
    build_blueprint.success_criteria = "The affected build succeeds.";
    assert(plans.create(build_blueprint, error));
    common_plan_state explanation_blueprint = *blueprint;
    explanation_blueprint.id = "architecture-explanation-blueprint";
    explanation_blueprint.purpose = "Explain repository architecture and runtime boundaries.";
    explanation_blueprint.goal = "Produce an architecture explanation.";
    explanation_blueprint.success_criteria = "The explanation is evidence-backed.";
    assert(plans.create(explanation_blueprint, error));
    selection_request.prompt = "Fix the repository build failure and verify the result.";
    common_memory_policy_pack selection_policy;
    selection_policy.purpose = "Repair a repository build failure.";
    selection_policy.goal = "Restore a passing build.";
    selection_policy.success_criteria = "The affected build succeeds.";
    selection_request.policy_pack = selection_policy;
    selection_config.task_plan_id = "ranked-build-instance";
    assert(common_agent_select_and_instantiate_blueprint(plans, selection_request, declining,
        {{"build-repair", build_blueprint.id, "Repository work",
          build_blueprint.purpose, build_blueprint.goal, build_blueprint.success_criteria},
         {"architecture", explanation_blueprint.id, "Repository work",
          explanation_blueprint.purpose, explanation_blueprint.goal, explanation_blueprint.success_criteria}},
        selection_config, selection_result, error));
    assert(selection_result.outcome == common_blueprint_selection_outcome::instantiated);
    assert(selection_result.logical_id && *selection_result.logical_id == "build-repair");
    assert(selection_result.reason.rfind("native keyword fallback", 0) == 0);
    selection_request.policy_pack.reset();

    common_agent_bootstrap_package custom;
    custom.name = "custom";
    custom.version = "v1";
    custom.procedures.push_back({"review-loop", "Review the implementation against its acceptance criteria before completion.", "review-loop"});
    common_agent_bootstrap_blueprint custom_blueprint;
    custom_blueprint.id = "review";
    custom_blueprint.goal = "Review a change";
    custom_blueprint.success_criteria = "Review findings are evidence-backed.";
    custom_blueprint.required_capabilities = {"workspace.read"};
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
    assert(parsed_package.blueprints.front().required_capabilities.size() == 1);
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
