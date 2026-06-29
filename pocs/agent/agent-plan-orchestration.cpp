#include "agent-plan-orchestration.h"

#include "../memory/memory-cli-memory.h"

#include "agent-cli-selection.h"
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

} // namespace

bool maybe_install_agent_bootstrap(
    common_memory_store & memory_store,
    common_plan_store & plan_store,
    const args & options,
    const common_agent_scope & scope,
    std::vector<common_blueprint_candidate> & installed_blueprint_candidates,
    std::string & error) {
    if (options.agent_bootstrap != "default" && options.agent_import.empty()) {
        error.clear();
        return true;
    }

    common_agent_bootstrap_config bootstrap_config;
    bootstrap_config.namespace_id = scope.namespace_id;
    bootstrap_config.session_id = scope.session_id;
    bootstrap_config.project_id = scope.project_id;
    bootstrap_config.now = std::time(nullptr);
    common_agent_bootstrap_result bootstrap_result;

    const auto embed = [&options](const std::string & text, std::vector<float> & embedding, std::string & embedding_error) {
        if (!ensure_memory_cli_embedding(options, text, embedding, "bootstrap procedure", embedding_error)) {
            return false;
        }
        if (embedding.empty()) {
            embedding_error = "--agent-bootstrap default requires --embedding-model";
            return false;
        }
        return true;
    };

    common_agent_bootstrap_package package;
    if (options.agent_import.empty()) {
        package = common_agent_default_bootstrap_package();
    } else if (!load_bootstrap_file(options.agent_import, package, error)) {
        error = "agent import failed: " + error;
        return false;
    }

    if (!common_agent_install_bootstrap_package(memory_store, plan_store, bootstrap_config, package, embed, bootstrap_result, error)) {
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

    if (!options.agent_blueprint.empty() && options.agent_blueprint != "auto") {
        common_explicit_blueprint_selector selector(options.agent_blueprint);
        common_blueprint_selection_config selection_config;
        selection_config.task_plan_id = options.plan_id;
        selection_config.session_id = scope.session_id;
        selection_config.scope = scope.plan_scope;
        selection_config.now = bootstrap_config.now;
        common_blueprint_selection_result selection;
        common_agent_request selection_request;
        selection_request.prompt = options.prompt;
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
            options.plan_id.c_str());
    }

    error.clear();
    return true;
}

bool maybe_export_agent_package(
    common_memory_store & memory_store,
    common_plan_store & plan_store,
    const args & options,
    bool & exported,
    std::string & error) {
    exported = false;
    if (options.agent_export.empty()) {
        error.clear();
        return true;
    }
    if (!export_agent_package(memory_store, plan_store, options, error)) {
        error = "agent export failed: " + error;
        return false;
    }
    fprintf(stderr, "agent export written: %s\n", options.agent_export.c_str());
    exported = true;
    error.clear();
    return true;
}

bool maybe_auto_select_plan(
    common_agent_inference & inference,
    const common_agent_generation_config & generation_config,
    const args & options,
    args & mutable_options,
    const common_agent_scope & scope,
    common_plan_store & plan_store,
    std::string & error) {
    if (options.agent_plan != "auto" || !mutable_options.plan_id.empty()) {
        error.clear();
        return true;
    }

    const auto plans = plan_store.list(error);
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
        if (!common_plan_scope_matches(plan, scope.plan_scope, scope.namespace_id, scope.session_id, scope.project_id, scope.turn_id)) {
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
        common_agent_request selection_request;
        selection_request.prompt = options.prompt;
        common_agent_scope_apply(scope, selection_request);
        std::string selection_error;
        const auto selection_result = select_llama_cli_plan_result(inference, generation_config, selection_request, candidates, selection_error);
        if (selection_result.plan_id) {
            mutable_options.plan_id = *selection_result.plan_id;
            fprintf(stderr, "agent plan auto-selected: %s\n", mutable_options.plan_id.c_str());
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
    common_agent_inference & inference,
    const common_agent_generation_config & generation_config,
    const args & options,
    args & mutable_options,
    const common_agent_scope & scope,
    common_plan_store & plan_store,
    const std::vector<common_blueprint_candidate> & installed_blueprint_candidates,
    bool profile_tools_active,
    const common_tool_registry * tool_registry,
    std::string & error) {
    if (options.agent_blueprint != "auto") {
        error.clear();
        return true;
    }

    auto selector = make_llama_cli_blueprint_selector(inference, generation_config);
    common_blueprint_selection_config config;
    config.task_plan_id = mutable_options.plan_id;
    config.session_id = scope.session_id;
    config.scope = scope.plan_scope;
    config.now = std::time(nullptr);
    common_blueprint_selection_result selection;
    common_agent_request selection_request;
    selection_request.prompt = options.prompt;
    common_agent_scope_apply(scope, selection_request);
    if (!common_agent_select_and_instantiate_blueprint(plan_store, selection_request, *selector, installed_blueprint_candidates, config, selection, error)) {
        error = "agent blueprint selection failed: " + error;
        return false;
    }

    if (selection.outcome == common_blueprint_selection_outcome::instantiated) {
        fprintf(stderr, "agent blueprint auto-selected: %s -> %s\n", selection.logical_id->c_str(), mutable_options.plan_id.c_str());
        if (profile_tools_active && tool_registry != nullptr) {
            common_agent_request binding_request;
            binding_request.prompt = options.prompt;
            common_agent_scope_apply(scope, binding_request);
            std::string binding_error;
            const auto binding_result = bind_llama_cli_blueprint_tools_result(
                inference, generation_config, *tool_registry, binding_request, plan_store, mutable_options.plan_id, binding_error);
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
