#include "agent-daemon-adapter.h"

#include "agent-cli-selection.h"

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
    return store_args;
}

common_agent_runtime_policy make_daemon_runtime_policy(const daemon_options & options) {
    common_agent_runtime_policy policy;
    policy.agent_inference_backend = "server-context";
    policy.memory_learn = options.memory_learn;
    policy.memory_learn_show_candidate = options.memory_learn_show_candidate;
    policy.plan_show_summary = options.plan_show_summary;
    policy.agent_trace = options.agent_trace;
    policy.enable_reflection = options.reflection_mode == "always";
    policy.max_iterations = policy.enable_reflection ? 2 : 1;
    policy.max_reflection_rounds = policy.enable_reflection ? 1 : 0;
    policy.max_tool_rounds = 0;
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
