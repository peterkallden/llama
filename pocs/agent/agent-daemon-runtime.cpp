#include "agent-daemon-adapter.h"

#include "agent-cli-host-adapter.h"
#include "agent-cli-selection.h"
#include "agent/agent-scope.h"
#include "../memory/memory-cli-memory.h"
#include "memory/memory-in-memory.h"
#include "plan/plan-in-memory.h"
#ifdef LLAMA_MEMORY_USE_COZO
#include "memory/cozo/memory-cozo.h"
#endif
#ifdef LLAMA_PLAN_USE_COZO
#include "plan/cozo/plan-cozo.h"
#endif

#include <cstdio>
#include <filesystem>

namespace {

std::string resolve_memory_backend(
        const std::string & backend,
        const std::string & memory_db,
        std::string & error) {
    std::string resolved = backend;
    if (resolved == "auto") {
        resolved = memory_db.empty() ? "in-memory" : "cozo";
    }
    if (resolved == "in-memory" && !memory_db.empty()) {
        error = "--memory-db requires --backend cozo or the default auto backend";
        return {};
    }
    error.clear();
    return resolved;
}

std::string resolve_plan_backend(
        const std::string & backend,
        const std::string & plan_db,
        std::string & error) {
    std::string resolved = backend;
    if (resolved == "auto") {
        resolved = plan_db.empty() ? "in-memory" : "cozo";
    }
    if (resolved == "in-memory" && !plan_db.empty()) {
        error = "--plan-db requires --plan-backend cozo or the default auto backend";
        return {};
    }
    error.clear();
    return resolved;
}

class daemon_agent_embedding_provider final : public agent_embedding_provider {
public:
    daemon_agent_embedding_provider(std::string model_path, int n_gpu_layers)
        : model_path_(std::move(model_path)),
          n_gpu_layers_(n_gpu_layers) {}

    bool embed(
            const std::string & purpose,
            const std::string & text,
            std::vector<float> & embedding,
            std::string & error) override {
        return ensure_memory_cli_embedding_from_model(
            model_path_,
            n_gpu_layers_,
            text,
            embedding,
            purpose.c_str(),
            error);
    }

private:
    std::string model_path_;
    int n_gpu_layers_ = 0;
};

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

agent_host_tool_selection_request make_daemon_tool_request(
        const daemon_options & options,
        const common_agent_runtime_session_host_turn_request & request) {
    agent_host_tool_selection_request tool_request;
    tool_request.tool_context.request_id = "daemon";
    tool_request.tool_context.turn_id = request.turn_id;
    tool_request.tool_context.scope.namespace_id = request.namespace_id;
    tool_request.tool_context.scope.session_id = request.session_id;
    tool_request.tool_context.scope.project_id = request.project_id;
    tool_request.tool_context.scope.turn_id = request.turn_id;
    tool_request.tool_context.scope.memory_scope = request.memory_scope;
    tool_request.tool_context.scope.plan_scope = request.plan_scope;
    tool_request.tool_context.memory_scope = request.memory_scope;
    tool_request.tool_context.plan_scope = request.plan_scope;
    tool_request.tool_context.profile_id = options.tool_profile;
    tool_request.tool_context.repository_root = options.repository_root;
    tool_request.tool_context.allow_network =
        options.tool_profile == "research" || !options.mcp_tool_command.empty();
    tool_request.tool_context.allow_policy_gated_writes =
        options.tool_profile == "memory" || options.tool_profile == "research";
    tool_request.tool_context.allow_memory_proposals =
        tool_request.tool_context.allow_policy_gated_writes;
    tool_request.tool_context.allow_plan_proposals =
        tool_request.tool_context.allow_policy_gated_writes;
    tool_request.tool_context.max_calls = options.max_tool_rounds > 0 ? options.max_tool_rounds : 1;
    tool_request.repository_root = options.repository_root.empty()
        ? std::string()
        : std::filesystem::weakly_canonical(options.repository_root).string();
    tool_request.resource_store_config = {
        options.resource_blob_backend,
        options.resource_blob_root,
        options.resource_metadata_backend,
        options.resource_metadata_db,
    };
    tool_request.mcp_tool_command = options.mcp_tool_command;
    tool_request.mcp_tool_args = options.mcp_tool_args;
    tool_request.mcp_tool_server_name = options.mcp_tool_server_name;
    tool_request.mcp_tool_prefix = options.mcp_tool_prefix;
    return tool_request;
}

} // namespace

bool resolve_agent_daemon_tooling(
        const daemon_options & options,
        const common_agent_runtime_resident_runtime * runtime,
        const common_agent_runtime_session_host_turn_request & request,
        common_memory_store & memory_store,
        common_plan_store & plan_store,
        agent_resource_store * resource_store,
        common_agent_runtime_tooling & tooling,
        std::string & error) {
    tooling = {};
    common_memory_query query = make_daemon_memory_query(request);
    const auto tool_request = make_daemon_tool_request(options, request);
        std::string * current_plan_id = runtime != nullptr
        ? const_cast<std::string *>(&runtime->current_plan_id())
        : nullptr;
    std::unique_ptr<agent_embedding_provider> embedding_provider;
    const std::string embedding_model = options.embedding_model.empty()
        ? options.model
        : options.embedding_model;
    if (!embedding_model.empty()) {
        embedding_provider = std::make_unique<daemon_agent_embedding_provider>(
            embedding_model,
            options.n_gpu_layers);
    }

    common_agent_cli_tool_selection selection;
    if (!resolve_agent_host_tool_selection(
            memory_store,
            &plan_store,
            resource_store,
            current_plan_id,
            options.tool_profile,
            tool_request,
            query,
            embedding_provider.get(),
            selection,
            error)) {
        error = "daemon tool provider resolution failed: " + error;
        return false;
    }
    selection.embedding_provider = std::move(embedding_provider);

    tooling = std::move(selection.tooling);
    if (selection.embedding_provider) {
        auto shared_embedding = std::shared_ptr<agent_embedding_provider>(
            std::move(selection.embedding_provider));
        tooling.owned_resources.push_back(std::static_pointer_cast<void>(shared_embedding));
    }
    if (selection.owned_resource_store) {
        auto shared_resource_store = std::shared_ptr<agent_resource_store>(
            std::move(selection.owned_resource_store));
        tooling.owned_resources.push_back(std::static_pointer_cast<void>(shared_resource_store));
    }
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
        agent_resource_store & resource_store,
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
        [&options, &memory_store, &plan_store, &resource_store](
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
                &resource_store,
                tooling,
                error);
        },
    };
}

bool open_daemon_memory_store(
        const daemon_options & options,
        std::unique_ptr<common_memory_store> & store,
        std::string & error) {
    const std::string backend = resolve_memory_backend(options.backend, options.memory_db, error);
    if (!error.empty()) {
        return false;
    }
    if (backend == "cozo") {
#ifdef LLAMA_MEMORY_USE_COZO
        if (options.memory_db.empty()) {
            error = "--backend cozo requires --memory-db PATH";
            return false;
        }
        store = std::make_unique<common_memory_cozo_store>();
#else
        error = "this binary was built without LLAMA_MEMORY_COZO";
        return false;
#endif
    } else if (backend == "in-memory") {
        store = std::make_unique<common_memory_in_memory_store>();
    } else {
        error = "unknown memory backend: " + backend;
        return false;
    }
    return store->open(options.memory_db, error);
}

bool open_daemon_plan_store(
        const daemon_options & options,
        std::unique_ptr<common_plan_store> & store,
        std::string & error) {
    const std::string backend = resolve_plan_backend(options.plan_backend, options.plan_db, error);
    if (!error.empty()) {
        return false;
    }
    if (backend == "cozo") {
#ifdef LLAMA_PLAN_USE_COZO
        if (options.plan_db.empty()) {
            error = "--plan-backend cozo requires --plan-db PATH";
            return false;
        }
        store = std::make_unique<common_plan_cozo_store>();
#else
        error = "this binary was built without LLAMA_PLAN_COZO";
        return false;
#endif
    } else if (backend == "in-memory") {
        store = std::make_unique<common_plan_in_memory_store>();
    } else {
        error = "unknown plan backend: " + backend;
        return false;
    }
    return store->open(options.plan_db, error);
}

bool open_daemon_resource_store(
        const daemon_options & options,
        std::unique_ptr<agent_resource_store> & store,
        std::string & error) {
    store = make_agent_resource_store({
        options.resource_blob_backend,
        options.resource_blob_root,
        options.resource_metadata_backend,
        options.resource_metadata_db,
    }, error);
    return store != nullptr;
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
    if (!open_daemon_resource_store(options, runtime.resource_store, error)) {
        return false;
    }
    if (!parse_mode(options.default_mode, runtime.default_mode)) {
        error = "unsupported default mode: " + options.default_mode;
        return false;
    }

    runtime.host = std::make_unique<common_agent_runtime_session_manager>(
        make_agent_runtime_session_manager_config(
            make_session_host_build_config(*runtime.memory_store, *runtime.plan_store, *runtime.resource_store, options)));
    error.clear();
    return true;
}
