#include "agent-cli-run-adapter.h"

#include "../memory/memory-cli-memory.h"

#include "agent-cli-selection.h"

#include <ctime>

bool prepare_agent_cli_args(args & options, std::string & error) {
    if (!resolve_agent_profile(options, error)) {
        return false;
    }
    if (options.max_tool_rounds < 1 || options.max_tool_rounds > 4) {
        error = "--max-tool-rounds must be between 1 and 4";
        return false;
    }
#ifndef LLAMA_MEMORY_POC_USE_AGENT_TOOLS
    if (!options.tool_profile.empty()) {
        error = "--tool-profile requires a build with LLAMA_AGENT_REFLECTION=ON";
        return false;
    }
    if (!options.mcp_tool_command.empty()) {
        error = "--mcp-tool-command requires a build with LLAMA_AGENT_REFLECTION=ON";
        return false;
    }
#endif
    if (options.mcp_tool_command.empty() && !options.mcp_tool_args.empty()) {
        error = "--mcp-tool-arg requires --mcp-tool-command";
        return false;
    }
    if (options.mcp_tool_server_name.empty()) {
        error = "--mcp-tool-server-name must not be empty";
        return false;
    }
    if (options.planning_mode != "off" && options.planning_mode != "mini") {
        error = "--planning-mode must be off or mini";
        return false;
    }
    if (options.agent_bootstrap != "none" && options.agent_bootstrap != "default") {
        error = "--agent-bootstrap must be none or default";
        return false;
    }
    if (options.agent_plan != "off" && options.agent_plan != "auto") {
        error = "--agent-plan must be off or auto";
        return false;
    }
    if (options.agent_bootstrap == "default" && !options.agent_import.empty()) {
        error = "--agent-bootstrap default cannot be combined with --agent-import";
        return false;
    }

    const bool bootstrap_enabled = options.agent_bootstrap == "default" || !options.agent_import.empty();
    if (options.planning_mode == "mini" && options.plan_scope == "turn" && options.memory_turn.empty()) {
        options.memory_turn = "implicit-" + std::to_string(std::time(nullptr));
    }
    if (bootstrap_enabled && options.planning_mode == "mini" && options.agent_blueprint.empty()) {
        options.agent_blueprint = "auto";
    }
    if (bootstrap_enabled && options.agent_blueprint == "auto" && options.plan_id.empty()) {
        options.plan_id = "agent-blueprint:" + options.memory_session + ":" +
            (options.memory_turn.empty() ? std::string("turn") : options.memory_turn);
    }
    if (!options.agent_export.empty() && (bootstrap_enabled || !options.agent_blueprint.empty())) {
        error = "--agent-export cannot be combined with bootstrap or blueprint selection";
        return false;
    }
    if (bootstrap_enabled && options.planning_mode != "mini") {
        error = "--agent-bootstrap requires --planning-mode mini";
        return false;
    }
    if (!options.agent_blueprint.empty() && (!bootstrap_enabled || options.plan_id.empty())) {
        error = "--agent-blueprint requires bootstrap and an explicit --plan-id";
        return false;
    }
    if (options.reflection_mode != "off" && options.reflection_mode != "always") {
        error = "--reflection-mode must be off or always";
        return false;
    }
    if (options.reflection_mode != "off" && options.planning_mode == "off") {
        error = "--reflection-mode requires --planning-mode mini";
        return false;
    }
    if (options.memory_learn != "off" && options.memory_learn != "post-turn") {
        error = "--memory-learn must be off or post-turn";
        return false;
    }
    if (options.memory_learn == "post-turn" && options.planning_mode != "mini") {
        error = "--memory-learn post-turn requires --planning-mode mini";
        return false;
    }
    if (options.agent_inference_backend != "cli" && options.planning_mode != "mini") {
        error = "--agent-inference-backend currently requires --planning-mode mini";
        return false;
    }
    if (options.memory_learn_min_confidence < 0.0f || options.memory_learn_min_confidence > 1.0f ||
            options.memory_learn_min_reuse < 0.0f || options.memory_learn_min_reuse > 1.0f) {
        error = "memory learning thresholds must be between 0 and 1";
        return false;
    }
#ifndef LLAMA_MEMORY_POC_USE_AGENT_TOOLS
    if (options.planning_mode == "mini") {
        error = "--planning-mode mini requires a build with LLAMA_AGENT_REFLECTION=ON";
        return false;
    }
#endif

    error.clear();
    return true;
}

bool prepare_agent_cli_run_setup(
        common_memory_store & memory_store,
        args & options,
        common_agent_cli_run_setup & setup,
        bool & exported,
        std::string & error) {
    setup = {};
    setup.bootstrap_enabled = options.agent_bootstrap == "default" || !options.agent_import.empty();
    setup.requested_plan_scope = common_plan_scope::turn;
    setup.agent_scope = common_cli_make_agent_scope(options, setup.requested_plan_scope);
    setup.active_plan_id = options.plan_id;
    setup.orchestration_config = make_agent_orchestration_config({
        options.prompt,
        options.agent_plan,
        options.agent_blueprint,
        options.agent_bootstrap,
        options.agent_import,
        options.agent_export,
    });
    setup.bootstrap_runtime_config.embed_procedure =
        [&options](const std::string & text, std::vector<float> & embedding, std::string & embed_error) {
            if (!ensure_memory_cli_embedding(options, text, embedding, "bootstrap procedure", embed_error)) {
                return false;
            }
            if (embedding.empty()) {
                embed_error = "--agent-bootstrap default requires --embedding-model";
                return false;
            }
            return true;
        };
    exported = false;

#ifdef LLAMA_MEMORY_POC_USE_AGENT_TOOLS
    if (options.planning_mode == "mini") {
        if (!parse_plan_scope(options.plan_scope, setup.requested_plan_scope)) {
            error = "unsupported plan scope: " + options.plan_scope;
            return false;
        }
        setup.agent_scope = common_cli_make_agent_scope(options, setup.requested_plan_scope);
        setup.active_plan_id = options.plan_id;
        setup.plan_store = make_plan_store(options, error);
        if (!setup.plan_store || !setup.plan_store->open(options.plan_db, error)) {
            if (error.empty()) {
                error = "failed to open plan store";
            }
            return false;
        }
        if ((setup.bootstrap_enabled || !options.agent_export.empty()) &&
                !common_cli_supports_bootstrap_package_scope(setup.agent_scope)) {
            error = "bootstrap/import/export currently supports only session- or project-scoped package tenants";
            return false;
        }
        if (!maybe_install_agent_bootstrap(
                memory_store,
                *setup.plan_store,
                setup.orchestration_config,
                setup.bootstrap_runtime_config,
                setup.agent_scope,
                setup.active_plan_id,
                setup.installed_blueprint_candidates,
                error)) {
            return false;
        }
        if (!maybe_export_agent_package(
                memory_store,
                *setup.plan_store,
                setup.orchestration_config,
                setup.agent_scope,
                exported,
                error)) {
            return false;
        }
    }
#else
    (void) memory_store;
#endif

    error.clear();
    return true;
}
