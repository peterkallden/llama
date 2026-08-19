#include "agent-plan-orchestration.h"

#include "../cli/agent-cli-selection.h"
#include "../tooling/agent-tool-provider.h"
#include "tools/agent/cli/agent-cli-scope.h"
#include "agent/agent-bootstrap.h"
#include "agent/learning/blueprint-selector.h"

#include <algorithm>
#include <cstdio>
#include <ctime>
#include <sstream>

namespace {

std::string make_bootstrap_prefix(const common_agent_scope & scope) {
    return "bootstrap:" + scope.namespace_id + ":" +
        (scope.project_id.empty() ? "session:" + scope.session_id : "project:" + scope.project_id) + ":";
}

common_agent_request make_orchestration_selection_request(
        const common_agent_orchestration_config & config,
        const common_agent_scope & scope) {
    common_agent_request request;
    request.prompt = config.prompt;
    common_agent_scope_apply(scope, request);
    return request;
}

} // namespace

common_agent_orchestration_config make_agent_orchestration_config(
        common_agent_orchestration_build_config config) {
    common_agent_orchestration_config result;
    result.prompt = std::move(config.prompt);
    result.agent_plan = std::move(config.agent_plan);
    result.agent_blueprint = std::move(config.agent_blueprint);
    result.agent_bootstrap = std::move(config.agent_bootstrap);
    result.agent_import = std::move(config.agent_import);
    result.agent_export = std::move(config.agent_export);
    return result;
}

bool maybe_install_agent_bootstrap(
    common_memory_store & memory_store,
    common_plan_store & plan_store,
    const common_agent_orchestration_config & config,
    const common_agent_bootstrap_runtime_config & runtime_config,
    const common_agent_scope & scope,
    std::string & current_plan_id,
    std::vector<common_blueprint_candidate> & installed_blueprint_candidates,
    std::string & error) {
    if (config.agent_bootstrap != "default" && config.agent_import.empty()) {
        error.clear();
        return true;
    }

    common_agent_bootstrap_config bootstrap_config;
    bootstrap_config.namespace_id = scope.namespace_id;
    bootstrap_config.session_id = scope.session_id;
    bootstrap_config.project_id = scope.project_id;
    bootstrap_config.now = std::time(nullptr);
    common_agent_bootstrap_result bootstrap_result;

    common_agent_bootstrap_package package;
    if (config.agent_import.empty()) {
        package = common_agent_default_bootstrap_package();
    } else if (!load_bootstrap_file(config.agent_import, package, error)) {
        error = "agent import failed: " + error;
        return false;
    }

    if (!common_agent_install_bootstrap_package(memory_store, plan_store, bootstrap_config, package, runtime_config.embed_procedure, bootstrap_result, error)) {
        error = "agent bootstrap failed: " + error;
        return false;
    }

    if (bootstrap_config.install_blueprints) {
        const std::string prefix = make_bootstrap_prefix(scope) + "blueprint:";
        for (const auto & blueprint : package.blueprints) {
            common_blueprint_candidate candidate;
            candidate.logical_id = blueprint.id;
            candidate.persisted_id = prefix + blueprint.id;
            candidate.description = blueprint.selection_description.empty() ? blueprint.goal : blueprint.selection_description;
            candidate.purpose = blueprint.purpose;
            candidate.goal = blueprint.goal;
            candidate.success_criteria = blueprint.success_criteria;
            candidate.required_capabilities = blueprint.required_capabilities;
            candidate.constraints = blueprint.constraints;
            candidate.assumptions = blueprint.assumptions;
            for (const auto & step : blueprint.steps) {
                const auto & contribution = step.intended_contribution.empty() ? step.objective : step.intended_contribution;
                if (!contribution.empty()) candidate.contributions.push_back(contribution);
            }
            installed_blueprint_candidates.push_back(std::move(candidate));
        }
    }

    fprintf(stderr, "agent bootstrap: procedures installed=%zu existing=%zu; blueprints installed=%zu existing=%zu\n",
        bootstrap_result.installed_memory_ids.size(), bootstrap_result.existing_memory_ids.size(),
        bootstrap_result.installed_blueprint_ids.size(), bootstrap_result.existing_blueprint_ids.size());

    if (!config.agent_blueprint.empty() && config.agent_blueprint != "auto") {
        common_explicit_blueprint_selector selector(config.agent_blueprint);
        common_blueprint_selection_config selection_config;
        selection_config.task_plan_id = current_plan_id;
        selection_config.session_id = scope.session_id;
        selection_config.scope = scope.plan_scope;
        selection_config.now = bootstrap_config.now;
        common_blueprint_selection_result selection;
        common_agent_request selection_request;
        selection_request.prompt = config.prompt;
        common_agent_scope_apply(scope, selection_request);
        if (!common_agent_select_and_instantiate_blueprint(plan_store, selection_request, selector, installed_blueprint_candidates, selection_config, selection, error)) {
            error = "agent blueprint selection failed: " + error;
            return false;
        }
        if (selection.outcome != common_blueprint_selection_outcome::instantiated &&
                selection.outcome != common_blueprint_selection_outcome::resumed) {
            error = "agent blueprint selection failed safely: " + selection.reason;
            return false;
        }
        fprintf(stderr, "agent blueprint %s: %s\n",
            selection.outcome == common_blueprint_selection_outcome::instantiated ? "instantiated" : "resumed",
            current_plan_id.c_str());
    }

    error.clear();
    return true;
}

bool maybe_export_agent_package(
    common_memory_store & memory_store,
    common_plan_store & plan_store,
    const common_agent_orchestration_config & config,
    const common_agent_scope & scope,
    bool & exported,
    std::string & error) {
    exported = false;
    if (config.agent_export.empty()) {
        error.clear();
        return true;
    }
    if (!export_agent_package(memory_store, plan_store, scope, config.agent_export, error)) {
        error = "agent export failed: " + error;
        return false;
    }
    fprintf(stderr, "agent export written: %s\n", config.agent_export.c_str());
    exported = true;
    error.clear();
    return true;
}

bool maybe_auto_select_plan(
    const common_agent_orchestration_runtime_context & context,
    std::string & error) {
    if (context.config.agent_plan != "auto" || !context.current_plan_id.empty()) {
        error.clear();
        return true;
    }

    const auto plans = context.plan_store.list(error);
    if (!error.empty()) {
        error = "failed to list plan candidates: " + error;
        return false;
    }

    std::vector<common_plan_state> candidates;
    for (const auto & plan : plans) {
        if (plan.kind != common_plan_kind::task ||
                (plan.status != common_plan_status::active && plan.status != common_plan_status::blocked)) {
            continue;
        }
        if (!common_plan_scope_matches(plan, context.scope.plan_scope, context.scope.namespace_id, context.scope.session_id, context.scope.project_id, context.scope.turn_id)) {
            continue;
        }
        candidates.push_back(plan);
    }

    std::sort(candidates.begin(), candidates.end(), [](const auto & lhs, const auto & rhs) {
        if (lhs.updated_at != rhs.updated_at) {
            return lhs.updated_at > rhs.updated_at;
        }
        return lhs.id < rhs.id;
    });
    if (candidates.size() > 8) {
        candidates.resize(8);
    }

    if (!candidates.empty()) {
        const auto selection_request = make_orchestration_selection_request(context.config, context.scope);
        std::string selection_error;
        const auto selection_result = select_llama_cli_plan_result(
            context.inference, context.generation_config, selection_request, candidates, selection_error);
        if (selection_result.plan_id) {
            context.current_plan_id = *selection_result.plan_id;
            fprintf(stderr, "agent plan auto-selected: %s\n", context.current_plan_id.c_str());
        } else if (!selection_error.empty()) {
            fprintf(stderr, "agent plan auto-selection failed safely: %s; creating a new plan\n", selection_error.c_str());
        } else {
            fprintf(stderr, "agent plan auto-selection declined; creating a new plan\n");
        }
    }

    error.clear();
    return true;
}

bool maybe_auto_select_blueprint(
    const common_agent_orchestration_runtime_context & context,
    std::string & error) {
    if (context.config.agent_blueprint != "auto") {
        error.clear();
        return true;
    }

    auto selector = make_llama_cli_blueprint_selector(context.inference, context.generation_config);
    common_blueprint_selection_config selection_config;
    selection_config.task_plan_id = context.current_plan_id;
    selection_config.session_id = context.scope.session_id;
    selection_config.scope = context.scope.plan_scope;
    selection_config.now = std::time(nullptr);
    if (context.tooling != nullptr && context.tooling->profile_tools_active) {
        selection_config.capabilities_resolved = true;
        for (const auto & tool : context.tooling->capabilities) {
            selection_config.available_capabilities.push_back(tool);
        }
        selection_config.blocked_constraint_ids = context.tooling->blocked_constraint_ids;
    }
    common_blueprint_selection_result selection;
    auto selection_request = make_orchestration_selection_request(context.config, context.scope);
    if (context.policy_pack != nullptr) {
        selection_request.policy_pack = *context.policy_pack;
    }
    if (!common_agent_select_and_instantiate_blueprint(
            context.plan_store,
            selection_request,
            *selector,
            context.installed_blueprint_candidates,
            selection_config,
            selection,
            error)) {
        error = "agent blueprint selection failed: " + error;
        return false;
    }

    std::ostringstream diagnostic;
    diagnostic << "blueprint selection candidates=" << selection.candidate_count
               << " eligible=" << selection.eligible_count
               << " rejected=" << selection.rejections.size()
               << " outcome=" << static_cast<int>(selection.outcome);
    if (!selection.reason.empty()) diagnostic << " reason=" << selection.reason;
    context.pre_turn_events.push_back({
        common_agent_event_type::blueprint_selection_evaluated,
        diagnostic.str(),
        {},
        context.current_plan_id.empty()
            ? std::nullopt
            : std::optional<std::string>(context.current_plan_id),
    });
    context.pre_turn_trace.push_back({
        common_runtime_trace_stage::plan,
        common_runtime_trace_kind::decided,
        diagnostic.str(),
        context.current_plan_id,
        {}, {}, {}, selection.logical_id.value_or(std::string{}),
    });

    if (selection.outcome == common_blueprint_selection_outcome::instantiated) {
        fprintf(stderr, "agent blueprint auto-selected: %s -> %s\n", selection.logical_id->c_str(), context.current_plan_id.c_str());
        if (context.tooling != nullptr && context.tooling->profile_tools_active && context.tooling->tool_view != nullptr) {
            const auto binding_request = make_orchestration_selection_request(context.config, context.scope);
            std::string binding_error;
            const auto binding_result = bind_llama_cli_blueprint_tools_result(
                context.inference,
                context.generation_config,
                *context.tooling->tool_view,
                binding_request,
                context.plan_store,
                context.current_plan_id,
                binding_error);
            if (!binding_result.applied) {
                fprintf(stderr, "agent blueprint binding declined safely: %s\n", binding_error.c_str());
            }
        }
    } else if (selection.outcome == common_blueprint_selection_outcome::resumed) {
        fprintf(stderr, "agent blueprint selection skipped: existing plan resumed\n");
    } else {
        fprintf(stderr, "agent blueprint auto-selection declined or failed safely; using normal plan creation\n");
    }

    error.clear();
    return true;
}
