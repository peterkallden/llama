#include "agent/agent-bootstrap.h"
#include "agent/agent-package-json.h"
#include "agent/learning/blueprint-selector.h"
#include "memory/memory-in-memory.h"
#include "plan/plan-blueprint.h"
#include "plan/plan-in-memory.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <algorithm>
#include <cassert>
#include <fstream>
#include <sstream>

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

class low_confidence_selector final : public common_blueprint_selector {
public:
    common_blueprint_selection select(const common_agent_request &, const std::vector<common_blueprint_candidate> &, std::string & error) override {
        error.clear();
        return {common_blueprint_selection_decision::instantiate, "repository-change", 0.1f, "model selected below threshold"};
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
    assert(first.installed_blueprint_ids.size() == 4);
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
    assert(blueprint->source_revision == "default@v1");
    assert(blueprint->purpose == "Safely modify a repository while preserving intended behavior.");
    assert(blueprint->constraints.size() == 2 && blueprint->assumptions.size() == 1);
    assert(!blueprint->steps.empty() && !blueprint->steps.front().tool_call);

    common_plan_state instance;
    assert(common_plan_instantiate_blueprint(*blueprint, "instance-a", config.session_id, instance, error, common_plan_scope::project, 43));
    assert(instance.kind == common_plan_kind::task);
    assert(instance.derived_from_plan_id && *instance.derived_from_plan_id == blueprint->id);
    assert(instance.source_revision == blueprint->source_revision);

    common_plan_state cyclic = *blueprint;
    cyclic.id = "cyclic-blueprint";
    cyclic.steps[0].depends_on = {cyclic.steps[1].id};
    cyclic.steps[1].depends_on = {cyclic.steps[0].id};
    assert(!common_plan_validate_blueprint(cyclic, {}, error));
    assert(error.find("cycle") != std::string::npos);
    assert(!common_plan_instantiate_blueprint(cyclic, "cyclic-instance", config.session_id, instance, error));

    common_plan_state invalid_blueprint = *blueprint;
    invalid_blueprint.id = "invalid-blueprint";
    invalid_blueprint.assumptions.front().valid = false;
    assert(!common_plan_instantiate_blueprint(invalid_blueprint, "invalid-instance", config.session_id, instance, error));
    assert(error.find("assumption") != std::string::npos);

    common_agent_bootstrap_result second;
    assert(common_agent_install_default_bootstrap(memory, plans, config, embed, second, error));
    assert(second.installed_memory_ids.empty() && second.installed_blueprint_ids.empty());
    assert(second.existing_memory_ids.size() == 4 && second.existing_blueprint_ids.size() == 4);

    std::string package_json;
    const auto default_package = common_agent_default_bootstrap_package();
    assert(default_package.blueprints.size() == 4);
    assert(default_package.blueprints[2].id == "dataset-inspect-summarize");
    assert(default_package.blueprints[3].id == "openapi-paged-retrieval");
    assert(common_agent_package_to_json(default_package, package_json, error));
    common_agent_bootstrap_package parsed_package;
    assert(common_agent_package_parse_json(package_json, parsed_package, error));
    assert(parsed_package.procedures.size() == default_package.procedures.size());
    assert(parsed_package.blueprints.size() == default_package.blueprints.size());
    auto versioned_package = default_package;
    versioned_package.blueprints.front().source_revision = "repository-work@v3";
    assert(common_agent_package_to_json(versioned_package, package_json, error));
    assert(common_agent_package_parse_json(package_json, parsed_package, error));
    assert(parsed_package.blueprints.front().source_revision == "repository-work@v3");
    versioned_package.blueprints.front().source_revision.assign(129, 'x');
    assert(!common_agent_package_to_json(versioned_package, package_json, error));
    assert(common_agent_package_parse_json(R"({"schema_version":1,"name":"forward-compatible","version":"v1","procedures":[],"blueprints":[],"future_section":{"ignored":true}})", parsed_package, error));

    // The import format is the portable starter-library boundary.  Keep this
    // fixture deliberately outside the built-in package so operators can
    // install a versioned code/data library without changing native defaults.
    const std::string starter_package_json = R"json(
{
  "schema_version": 1,
  "name": "agent-workflows-starter",
  "version": "v1",
  "procedures": [
    {"id":"code-orientation-v1","summary":"code orientation","content":"Before modifying repository code, identify the relevant contract, implementation, direct callers, tests and execution seam. Prefer semantic symbol/reference diagnostics when available; otherwise use bounded repository search and reads.","importance":0.9,"confidence":1.0},
    {"id":"verification-ladder-v1","summary":"verification ladder","content":"Verify at the narrowest authoritative level first: contract or unit test, then functional smoke, then broader affected tests. Do not claim model-backed verification unless that run completed.","importance":0.9,"confidence":1.0},
    {"id":"data-understanding-v1","summary":"data understanding","content":"Before analysing an unfamiliar dataset, establish its identity and the fields needed by the question. Inspect schema or sample only as far as necessary; a host-trusted schema may satisfy this requirement.","importance":0.85,"confidence":1.0}
  ],
  "blueprints": [
    {
      "id":"repository-investigation-v1",
      "source_revision":"agent-workflows-starter@v1",
      "selection_description":"Investigate a repository question without modifying the workspace.",
      "purpose":"Establish repository evidence before proposing a change.",
      "goal":"Answer a repository investigation from inspected contracts, callers and tests.",
      "success_criteria":"Relevant implementation, dependencies and evidence are identified without modifying the workspace.",
      "required_capabilities":["tool.repository"],
      "next_action":"orient",
      "constraints":[{"id":"read-only","description":"Do not modify the workspace during investigation.","hard":true}],
      "assumptions":[{"id":"workspace","statement":"A controlled repository workspace is available.","confidence":0.9}],
      "steps":[
        {"id":"orient","title":"Orient","objective":"Identify the relevant repository area and available evidence."},
        {"id":"trace","title":"Trace contract","objective":"Inspect the contract, implementation, direct callers and relevant tests.","depends_on":["orient"]},
        {"id":"establish-evidence","title":"Establish evidence","objective":"Distinguish observed behavior from unverified inference.","depends_on":["trace"]},
        {"id":"answer","title":"Answer","objective":"Answer from the established evidence.","depends_on":["establish-evidence"]}
      ]
    },
    {
      "id":"dataset-analysis-v1",
      "source_revision":"agent-workflows-starter@v1",
      "selection_description":"Analyse a dataset through a host-resolved workflow and verified result.",
      "purpose":"Answer a bounded analytical question without guessing dataset identity or tool arguments.",
      "goal":"Analyse a resolved dataset and explain the verified result.",
      "success_criteria":"Dataset identity, sufficient data understanding, semantic operation and result verification are established before answering.",
      "required_capabilities":["tool.dataset","workflow.dataset"],
      "next_action":"resolve-dataset",
      "constraints":[{"id":"host-tool-contract","description":"Use host-resolved dataset contracts; do not invent tool names or arguments.","hard":true},{"id":"evidence-before-answer","description":"Do not present an unverified tool result as a completed answer.","hard":true}],
      "assumptions":[{"id":"dataset-access","statement":"The host exposes an applicable dataset operation.","confidence":0.8}],
      "steps":[
        {"id":"resolve-dataset","title":"Resolve dataset","objective":"Resolve the dataset through host-owned resource and dataset authority."},
        {"id":"understand-data","title":"Establish data understanding","objective":"Establish the fields needed by the question, using trusted schema or bounded inspection.","depends_on":["resolve-dataset"]},
        {"id":"choose-operation","title":"Choose operation","objective":"Choose the smallest semantic operation that answers the question.","depends_on":["understand-data"]},
        {"id":"execute-analysis","title":"Execute analysis","objective":"Execute the host-validated operation and retain its result as evidence.","depends_on":["choose-operation"]},
        {"id":"verify-result","title":"Verify result","objective":"Check that the result matches the requested dataset and operation.","depends_on":["execute-analysis"]},
        {"id":"answer","title":"Answer","objective":"Explain the verified result and its material limitations.","depends_on":["verify-result"]}
      ]
    }
  ]
}
)json";
    common_agent_bootstrap_package starter_package;
    assert(common_agent_package_parse_json(starter_package_json, starter_package, error));
    assert(starter_package.name == "agent-workflows-starter" && starter_package.version == "v1");
    assert(starter_package.procedures.size() == 3 && starter_package.blueprints.size() == 2);
    assert(starter_package.blueprints[1].selection_description ==
        "Analyse a dataset through a host-resolved workflow and verified result.");
    assert(starter_package.blueprints[1].source_revision == "agent-workflows-starter@v1");

    common_memory_in_memory_store starter_memory;
    common_plan_in_memory_store starter_plans;
    assert(starter_memory.open("", error));
    assert(starter_plans.open("", error));
    common_agent_bootstrap_result starter_import;
    assert(common_agent_install_bootstrap_package(
        starter_memory, starter_plans, config, starter_package, embed,
        starter_import, error));
    assert(starter_import.installed_memory_ids.size() == 3);
    assert(starter_import.installed_blueprint_ids.size() == 2);
    const auto imported_dataset_blueprint = starter_plans.get(
        "bootstrap:local:project:project-a:blueprint:dataset-analysis-v1", error);
    assert(imported_dataset_blueprint);
    assert(imported_dataset_blueprint->source_revision == "agent-workflows-starter@v1");
    assert(imported_dataset_blueprint->required_capabilities.size() == 2);

    common_agent_bootstrap_result starter_repeat;
    assert(common_agent_install_bootstrap_package(
        starter_memory, starter_plans, config, starter_package, embed,
        starter_repeat, error));
    assert(starter_repeat.installed_memory_ids.empty() && starter_repeat.installed_blueprint_ids.empty());
    assert(starter_repeat.existing_memory_ids.size() == 3 && starter_repeat.existing_blueprint_ids.size() == 2);

    // The checked-in package is the operator-facing import artifact. Parse and
    // install that exact file rather than relying only on the smaller inline
    // fixture above.
    std::ifstream starter_file(std::string(LLAMA_AGENT_SOURCE_DIR) +
        "/docs/examples/agent-bootstrap-workflows-v1.json");
    assert(starter_file);
    std::stringstream starter_file_text;
    starter_file_text << starter_file.rdbuf();
    common_agent_bootstrap_package checked_in_starter;
    assert(common_agent_package_parse_json(starter_file_text.str(), checked_in_starter, error));
    assert(checked_in_starter.name == "agent-workflows-starter" &&
        checked_in_starter.version == "v2");
    assert(checked_in_starter.procedures.size() == 8 &&
        checked_in_starter.blueprints.size() == 7);
    assert(checked_in_starter.workflows.size() == 8);
    assert(checked_in_starter.workflows[0].definition.workflow_ref ==
        "workflow://dataset/analysis");
    assert(checked_in_starter.workflows[0].definition.required_capabilities ==
        std::vector<std::string>({"dataset.resolve", "dataset.inspect", "data.query",
            "data.filter", "data.aggregate", "data.transform", "statistics.describe",
            "statistics.outliers", "statistics.value_counts"}));
    const auto resource_workflow = std::find_if(checked_in_starter.workflows.begin(),
        checked_in_starter.workflows.end(), [](const auto & workflow) {
            return workflow.definition.workflow_ref == "workflow://resource/document-analysis";
        });
    assert(resource_workflow != checked_in_starter.workflows.end());
    assert(resource_workflow->definition.required_capabilities ==
        std::vector<std::string>({"resource.inspect", "resource.read", "document.inspect", "data.query"}));
    assert(resource_workflow->definition.optional_capabilities ==
        std::vector<std::string>({"data.filter", "data.aggregate", "data.transform",
            "statistics.describe", "statistics.outliers", "statistics.value_counts"}));
    assert(resource_workflow->definition.workflow_revision == "v2");
    assert(checked_in_starter.workflows[0].definition.required_context ==
        std::vector<std::string>({"context.dataset.available"}));
    assert(checked_in_starter.blueprints[3].workflow_bindings.size() == 3);
    assert(checked_in_starter.blueprints[3].id == "dataset-analysis-v1");
    assert(checked_in_starter.blueprints[3].source_revision ==
        "agent-workflows-starter@v1");

    common_memory_in_memory_store checked_in_memory;
    common_plan_in_memory_store checked_in_plans;
    assert(checked_in_memory.open("", error));
    assert(checked_in_plans.open("", error));
    common_agent_bootstrap_result checked_in_import;
    assert(common_agent_install_bootstrap_package(
        checked_in_memory, checked_in_plans, config, checked_in_starter, embed,
        checked_in_import, error));
    assert(checked_in_import.installed_memory_ids.size() == 8 &&
        checked_in_import.installed_blueprint_ids.size() == 7);
    assert(checked_in_import.installed_workflow_ids.size() == 8);
    const auto checked_in_dataset = checked_in_plans.get(
        "bootstrap:local:project:project-a:blueprint:dataset-analysis-v1", error);
    assert(checked_in_dataset &&
        checked_in_dataset->source_revision == "agent-workflows-starter@v1");
    assert(checked_in_dataset->selection_description ==
        "Analyse a dataset through a host-resolved workflow and verified result.");
    assert(checked_in_dataset->workflow_bindings.size() == 3);
    const auto checked_in_workflow = checked_in_plans.get(
        "bootstrap:local:project:project-a:workflow:dataset-analysis-path-v1", error);
    assert(checked_in_workflow && checked_in_workflow->kind == common_plan_kind::workflow &&
        checked_in_workflow->workflow_definition &&
        checked_in_workflow->workflow_definition->workflow_revision == "v1" &&
        checked_in_workflow->workflow_definition->required_capabilities ==
            checked_in_starter.workflows[0].definition.required_capabilities &&
        checked_in_workflow->workflow_definition->optional_capabilities ==
            checked_in_starter.workflows[0].definition.optional_capabilities &&
        checked_in_workflow->workflow_definition->required_context ==
            checked_in_starter.workflows[0].definition.required_context);
    std::string checked_in_round_trip;
    assert(common_agent_package_to_json(checked_in_starter, checked_in_round_trip, error));
    common_agent_bootstrap_package reparsed_starter;
    assert(common_agent_package_parse_json(checked_in_round_trip, reparsed_starter, error));
    assert(reparsed_starter.blueprints[3].selection_description ==
        checked_in_starter.blueprints[3].selection_description);
    assert(reparsed_starter.workflows.size() == 8 &&
        reparsed_starter.blueprints[3].workflow_bindings.size() == 3 &&
        reparsed_starter.workflows[0].definition.required_capabilities ==
            checked_in_starter.workflows[0].definition.required_capabilities &&
        reparsed_starter.workflows[5].definition.optional_capabilities ==
            checked_in_starter.workflows[5].definition.optional_capabilities &&
        reparsed_starter.workflows[0].definition.required_context ==
            checked_in_starter.workflows[0].definition.required_context);

    auto partial_package = checked_in_starter;
    partial_package.blueprints[3].workflow_bindings.push_back({"workflow://missing", "v1"});
    common_memory_in_memory_store partial_memory;
    common_plan_in_memory_store partial_plans;
    assert(partial_memory.open("", error));
    assert(partial_plans.open("", error));
    common_agent_bootstrap_result partial_import;
    assert(common_agent_install_bootstrap_package(
        partial_memory, partial_plans, config, partial_package, embed,
        partial_import, error));
    assert(partial_import.installed_blueprint_ids.size() == 6 &&
        partial_import.rejected_items.size() == 1 &&
        partial_import.rejected_items.front().id == "dataset-analysis-v1");

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

    // A host may opt a selected dataset blueprint into the bounded A* backend
    // before persistence. The callback receives host-owned task data, verifies
    // the proposal, and returns an ordinary task plan; no second plan store or
    // runtime path is introduced.
    common_explicit_blueprint_selector dataset_selector("dataset-inspect-summarize");
    common_blueprint_selection_config dataset_selection_config;
    dataset_selection_config.task_plan_id = "dataset-astar-instance";
    dataset_selection_config.session_id = config.session_id;
    dataset_selection_config.scope = common_plan_scope::project;
    dataset_selection_config.now = 45;
    dataset_selection_config.materialize_instance =
        [](const common_agent_request &, const common_plan_state & instance,
                common_plan_state & materialized, std::string & materialization_error) {
            materialization_error.clear();
            if (!instance.derived_from_plan_id ||
                    instance.derived_from_plan_id->find("dataset-inspect-summarize") == std::string::npos) {
                return common_blueprint_materialization_outcome::not_applicable;
            }
            materialized = instance;
            materialized.steps.clear();
            common_plan_step select;
            select.id = instance.id + ":host:select";
            select.title = "Select dataset";
            select.objective = "Select the host-resolved dataset.";
            select.mode = common_plan_step_mode::tool;
            select.status = common_plan_step_status::active;
            select.selected_tool = "dataset.select";
            select.tool_call = common_plan_tool_call{"dataset.select", R"({"dataset":"dataset://local/sales"})"};
            common_plan_step answer;
            answer.id = instance.id + ":host:answer";
            answer.title = "Answer";
            answer.objective = instance.goal;
            answer.mode = common_plan_step_mode::final_response;
            answer.depends_on = {select.id};
            materialized.steps = {select, answer};
            materialized.active_step_id = select.id;
            materialized.next_action = select.id;
            materialized.status = common_plan_status::active;
            return common_blueprint_materialization_outcome::applied;
        };
    dataset_selection_config.task_plan_id = "dataset-astar-instance";
    assert(common_agent_select_and_instantiate_blueprint(
        plans, selection_request, dataset_selector,
        {{"dataset-inspect-summarize", first.installed_blueprint_ids[2], "dataset workflow"}},
        dataset_selection_config, selection_result, error));
    assert(selection_result.outcome == common_blueprint_selection_outcome::instantiated);
    const auto dataset_instance = plans.get(dataset_selection_config.task_plan_id, error);
    assert(dataset_instance && dataset_instance->steps.size() == 2);
    assert(dataset_instance->steps[0].selected_tool &&
        *dataset_instance->steps[0].selected_tool == "dataset.select");
    assert(dataset_instance->steps[1].mode == common_plan_step_mode::final_response);
    assert(dataset_instance->derived_from_plan_id &&
        *dataset_instance->derived_from_plan_id == first.installed_blueprint_ids[2]);

    common_blueprint_selection_config rejected_selection_config = dataset_selection_config;
    rejected_selection_config.task_plan_id = "rejected-astar-instance";
    rejected_selection_config.materialize_instance =
        [](const common_agent_request &, const common_plan_state &,
                common_plan_state &, std::string & materialization_error) {
            materialization_error = "host Oracle rejected the proposal";
            return common_blueprint_materialization_outcome::failed_safely;
        };
    common_explicit_blueprint_selector rejected_selector("repository-change");
    assert(common_agent_select_and_instantiate_blueprint(
        plans, selection_request, rejected_selector,
        {{"repository-change", first.installed_blueprint_ids.front(), "repository work"}},
        rejected_selection_config, selection_result, error));
    assert(selection_result.outcome == common_blueprint_selection_outcome::failed_safely);
    assert(plans.get(rejected_selection_config.task_plan_id, error) == std::nullopt);

    selection_config.expected_source_revision = "default@v0";
    selection_config.task_plan_id = "stale-source-instance";
    assert(common_agent_select_and_instantiate_blueprint(plans, selection_request, selector,
        { {"repository-change", first.installed_blueprint_ids.front(), "repository work"} },
        selection_config, selection_result, error));
    assert(selection_result.outcome == common_blueprint_selection_outcome::declined &&
        selection_result.eligible_count == 0 && selection_result.rejections.size() == 1 &&
        selection_result.rejections.front().reason.find("stale") != std::string::npos);
    selection_config.expected_source_revision.clear();

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

    common_plan_state wrong_scope = *selected_instance;
    wrong_scope.id = "wrong-scope-instance";
    wrong_scope.project_id = "other-project";
    assert(plans.create(wrong_scope, error));
    selection_config.task_plan_id = wrong_scope.id;
    assert(common_agent_select_and_instantiate_blueprint(plans, selection_request, selector,
        {{"repository-change", first.installed_blueprint_ids.front(), "repository work"}}, selection_config, selection_result, error));
    assert(selection_result.outcome == common_blueprint_selection_outcome::failed_safely);

    declining_selector declining;
    low_confidence_selector low_confidence;
    selection_config.task_plan_id = "fallback-instance";
    selection_request.prompt = "Diagnose unexpected behavior in an agent regression.";
    assert(common_agent_select_and_instantiate_blueprint(plans, selection_request, low_confidence,
        {{"repository-change", first.installed_blueprint_ids.front(), "Implement or modify code in a repository."},
         {"agent-regression", first.installed_blueprint_ids.back(), "Diagnose unexpected behavior at the plan, memory, tool, or reflection boundary."}},
        selection_config, selection_result, error));
    assert(selection_result.outcome == common_blueprint_selection_outcome::instantiated);
    assert(selection_result.logical_id && *selection_result.logical_id == "agent-regression");
    assert(selection_result.reason.rfind("native keyword fallback", 0) == 0);

    // A single generic overlap such as "tool" must not make auto-plan replace
    // a data task with the unrelated regression blueprint.  When no blueprint
    // is semantically applicable, normal agent orchestration owns the turn.
    selection_config.task_plan_id = "single-keyword-fallback-instance";
    selection_request.prompt = "Use the data and statistics tool families to inspect a dataset.";
    assert(common_agent_select_and_instantiate_blueprint(plans, selection_request, low_confidence,
        {{"repository-change", first.installed_blueprint_ids.front(), "Implement or modify code in a repository."},
         {"agent-regression", first.installed_blueprint_ids.back(), "Diagnose unexpected behavior at the plan, memory, tool, or reflection boundary."}},
        selection_config, selection_result, error));
    assert(selection_result.outcome == common_blueprint_selection_outcome::declined);

    // An explicit model decline is semantically different from a low
    // confidence selection: it must preserve normal agent orchestration even
    // when a keyword fallback would otherwise find a matching blueprint.
    selection_config.task_plan_id = "explicit-decline-instance";
    selection_request.prompt = "Diagnose unexpected behavior in an agent regression.";
    assert(common_agent_select_and_instantiate_blueprint(plans, selection_request, declining,
        {{"repository-change", first.installed_blueprint_ids.front(), "Implement or modify code in a repository."},
         {"agent-regression", first.installed_blueprint_ids.back(), "Diagnose unexpected behavior at the plan, memory, tool, or reflection boundary."}},
        selection_config, selection_result, error));
    assert(selection_result.outcome == common_blueprint_selection_outcome::declined);

    selection_config.allow_keyword_fallback = false;
    selection_config.task_plan_id = "fallback-disabled-instance";
    assert(common_agent_select_and_instantiate_blueprint(plans, selection_request, declining,
        {{"repository-change", first.installed_blueprint_ids.front(), "repository work"}},
        selection_config, selection_result, error));
    assert(selection_result.outcome == common_blueprint_selection_outcome::declined);
    selection_config.allow_keyword_fallback = true;

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
    assert(common_agent_select_and_instantiate_blueprint(plans, selection_request, low_confidence,
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
