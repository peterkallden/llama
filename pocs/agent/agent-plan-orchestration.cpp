#include "agent-plan-orchestration.h"

#include "../memory/memory-cli-memory.h"

#include "agent-cli-selection.h"
#include "agent-tool-provider.h"
#include "common/cli-scope.h"
#include "agent/agent-bootstrap.h"
#include "agent/blueprint-selector.h"

#include <algorithm>
#include <cstdio>
#include <ctime>

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

common_agent_orchestration_config make_agent_orchestration_config(const args & options) {
    common_agent_orchestration_config config;
    config.prompt = options.prompt;
    config.agent_plan = options.agent_plan;
    config.agent_blueprint = options.agent_blueprint;
    config.agent_bootstrap = options.agent_bootstrap;
    config.agent_import = options.agent_import;
    config.agent_export = options.agent_export;
    return config;
}

common_agent_bootstrap_runtime_config make_agent_bootstrap_runtime_config(const args & options) {
    common_agent_bootstrap_runtime_config config;
    config.embed_procedure = [&options](const std::string & text, std::vector<float> & embedding, std::string & error) {
        if (!ensure_memory_cli_embedding(options, text, embedding, "bootstrap procedure", error)) {
            return false;
        }
        if (embedding.empty()) {
            error = "--agent-bootstrap default requires --embedding-model";
            return false;
        }
        return true;
    };
    return config;
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
            installed_blueprint_candidates.push_back({
                blueprint.id,
                prefix + blueprint.id,
                blueprint.selection_description.empty() ? blueprint.goal : blueprint.selection_description,
            });
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
    common_blueprint_selection_result selection;
    const auto selection_request = make_orchestration_selection_request(context.config, context.scope);
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
