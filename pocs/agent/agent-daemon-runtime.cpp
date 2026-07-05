#include "agent-daemon-adapter.h"

#include "agent-cli-host-adapter.h"
#include "agent-cli-selection.h"
#include "agent/agent-scope.h"

#include <cstdio>

namespace {

args make_store_args(const daemon_options & options) {
    args store_args;
    store_args.prompt = "";
    store_args.model = options.model;
    store_args.embedding_model = options.embedding_model;
    store_args.backend = options.backend;
    store_args.memory_db = options.memory_db;
    store_args.plan_backend = options.plan_backend;
    store_args.plan_db = options.plan_db;
    store_args.n_gpu_layers = options.n_gpu_layers;
    store_args.tool_profile = options.tool_profile;
    store_args.repository_root = options.repository_root;
    store_args.max_tool_rounds = options.max_tool_rounds;
    store_args.mcp_tool_command = options.mcp_tool_command;
    store_args.mcp_tool_args = options.mcp_tool_args;
    store_args.mcp_tool_server_name = options.mcp_tool_server_name;
    store_args.mcp_tool_prefix = options.mcp_tool_prefix;
    return store_args;
}

common_agent_runtime_policy make_daemon_runtime_policy(const daemon_options & options) {
    common_agent_runtime_policy policy;
    policy.agent_inference_backend = "server-context";
    policy.tool_profile = options.tool_profile;
    policy.memory_learn = options.memory_learn;
    policy.memory_learn_show_candidate = options.memory_learn_show_candidate;
    policy.plan_show_summary = options.plan_show_summary;
    policy.agent_trace = options.agent_trace;
    policy.enable_reflection = options.reflection_mode == "always";
    policy.max_iterations = policy.enable_reflection ? 2 : 1;
    policy.max_reflection_rounds = policy.enable_reflection ? 1 : 0;
    policy.max_tool_rounds = options.max_tool_rounds;
    policy.allow_policy_gated_tool_proposals = false;
    return policy;
}

common_agent_runtime_config make_daemon_runtime_config(const daemon_options & options) {
    common_agent_runtime_config config;
    config.generation_config.n_predict = options.n_predict;
    config.enable_memory_learning = options.memory_learn == "post-turn";
    config.memory_learning_config.min_confidence = options.memory_learn_min_confidence;
    config.memory_learning_config.min_expected_reuse = options.memory_learn_min_reuse;
    return config;
}

common_agent_orchestration_config make_daemon_orchestration_config(const daemon_options & options) {
    common_agent_orchestration_config config;
    config.prompt = "";
    config.agent_plan = options.agent_plan;
    config.agent_blueprint = "off";
    return config;
}

common_agent_runtime_resident_request_config make_resident_request_config(
        const daemon_options & options) {
    return {
        "",
        "",
        "",
        "",
        options.model,
        options.n_predict,
        options.n_gpu_layers,
        false,
        "server-context",
        common_memory_scope::session,
        common_plan_scope::turn,
    };
}

common_memory_query make_daemon_memory_query(
        const common_agent_runtime_session_host_turn_request & request) {
    common_memory_query query;
    query.text = request.prompt;
    query.limit = 8;
    query.token_budget = 768;

    common_agent_scope scope;
    scope.namespace_id = request.namespace_id;
    scope.session_id = request.session_id;
    scope.project_id = request.project_id;
    scope.turn_id = request.turn_id;
    scope.memory_scope = request.memory_scope;
    scope.plan_scope = request.plan_scope;
    common_agent_scope_apply(scope, query);
    return query;
}

args make_daemon_tool_args(
        const daemon_options & options,
        const common_agent_runtime_session_host_turn_request & request) {
    args tool_args = make_store_args(options);
    tool_args.prompt = request.prompt;
    tool_args.memory_scope = common_memory_scope_name(request.memory_scope);
    tool_args.memory_namespace = request.namespace_id;
    tool_args.memory_session = request.session_id;
    tool_args.memory_project = request.project_id;
    tool_args.memory_turn = request.turn_id;
    switch (request.plan_scope) {
        case common_plan_scope::turn: tool_args.plan_scope = "turn"; break;
        case common_plan_scope::session: tool_args.plan_scope = "session"; break;
        case common_plan_scope::project: tool_args.plan_scope = "project"; break;
        case common_plan_scope::global: tool_args.plan_scope = "global"; break;
    }
    return tool_args;
}

} // namespace

bool resolve_agent_daemon_tooling(
        const daemon_options & options,
        const common_agent_runtime_resident_runtime * runtime,
        const common_agent_runtime_session_host_turn_request & request,
        common_memory_store & memory_store,
        common_plan_store & plan_store,
        common_agent_runtime_tooling & tooling,
        std::string & error) {
    tooling = {};
    args tool_args = make_daemon_tool_args(options, request);
    common_memory_query query = make_daemon_memory_query(request);
    std::string * current_plan_id = runtime != nullptr
        ? const_cast<std::string *>(&runtime->current_plan_id())
        : nullptr;

    common_agent_cli_tool_selection selection;
    if (!resolve_agent_cli_tool_selection(
            memory_store,
            &plan_store,
            current_plan_id,
            tool_args,
            query,
            true,
            selection,
            error)) {
        error = "daemon tool provider resolution failed: " + error;
        return false;
    }

    tooling = std::move(selection.tooling);
    if (selection.mcp_client) {
        tooling.owned_resources.push_back(std::static_pointer_cast<void>(
            std::shared_ptr<agent_mcp_tool_client>(std::move(selection.mcp_client))));
    }
    if (selection.tool_view) {
        auto shared_view = std::shared_ptr<agent_tool_view>(std::move(selection.tool_view));
        tooling.tool_view = shared_view.get();
        tooling.owned_resources.push_back(std::static_pointer_cast<void>(shared_view));
    }
    error.clear();
    return true;
}

namespace {

common_agent_runtime_session_host_build_config make_session_host_build_config(
        common_memory_store & memory_store,
        common_plan_store & plan_store,
        const daemon_options & options) {
    return {
        memory_store,
        plan_store,
        make_resident_request_config(options),
        make_daemon_runtime_policy(options),
        make_daemon_runtime_config(options),
        make_daemon_orchestration_config(options),
        common_memory_scope::session,
        true,
        {},
        {},
        [&options, &memory_store, &plan_store](
                const common_agent_runtime_resident_runtime * runtime,
                const common_agent_runtime_session_host_turn_request & request,
                common_agent_runtime_tooling & tooling,
                std::string & error) {
            return resolve_agent_daemon_tooling(
                options,
                runtime,
                request,
                memory_store,
                plan_store,
                tooling,
                error);
        },
    };
}

bool open_daemon_memory_store(
        const daemon_options & options,
        std::unique_ptr<common_memory_store> & store,
        std::string & error) {
    const auto store_args = make_store_args(options);
    store = make_memory_store(store_args, error);
    if (!store) {
        return false;
    }
    return open_memory_store(*store, store_args, error);
}

bool open_daemon_plan_store(
        const daemon_options & options,
        std::unique_ptr<common_plan_store> & store,
        std::string & error) {
    const auto store_args = make_store_args(options);
    store = make_plan_store(store_args, error);
    if (!store) {
        return false;
    }
    return store->open(store_args.plan_db, error);
}

} // namespace

bool initialize_agent_daemon_environment(
        const daemon_options & options,
        common_agent_daemon_runtime & runtime,
        std::string & error) {
    if (!open_daemon_memory_store(options, runtime.memory_store, error)) {
        return false;
    }
    if (!open_daemon_plan_store(options, runtime.plan_store, error)) {
        return false;
    }
    if (!parse_mode(options.default_mode, runtime.default_mode)) {
        error = "unsupported default mode: " + options.default_mode;
        return false;
    }

    runtime.host = std::make_unique<common_agent_runtime_session_manager>(
        make_agent_runtime_session_manager_config(
            make_session_host_build_config(*runtime.memory_store, *runtime.plan_store, options)));
    error.clear();
    return true;
}
