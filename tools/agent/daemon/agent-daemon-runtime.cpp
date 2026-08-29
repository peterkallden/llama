#include "agent-daemon-adapter.h"

#include "../cli/agent-cli-host-adapter.h"
#include "../cli/agent-cli-selection.h"
#include "../runtime/agent-inference-capacity-gate.h"
#include "agent/agent-scope.h"
#include "tools/agent/cli/agent-cli-memory-tools.h"
#include "memory/memory-in-memory.h"
#include "plan/plan-in-memory.h"
#include "../data/agent-data-store-factory.h"
#ifdef LLAMA_MEMORY_USE_COZO
#include "memory/cozo/memory-cozo.h"
#endif
#ifdef LLAMA_PLAN_USE_COZO
#include "plan/cozo/plan-cozo.h"
#endif
#ifdef LLAMA_MEMORY_USE_SQLITE
#include "memory/sqlite/memory-sqlite.h"
#endif
#ifdef LLAMA_PLAN_USE_SQLITE
#include "plan/sqlite/plan-sqlite.h"
#endif

#include <cstdio>
#include <filesystem>

namespace {

std::optional<common_memory_policy_pack> make_daemon_policy_pack(
        const daemon_options & options) {
    common_memory_policy_pack pack;
    pack.id = "daemon-session-policy";
    pack.purpose = "Run host-owned resident agent turns through the foreground daemon.";
    pack.goal = options.default_mode == "agent"
        ? "Plan, execute bounded steps, and answer without widening host authority."
        : "Answer through the resident runtime without widening host authority.";
    pack.constraints = {
        "Treat memory, tools, plans, and resources as host-owned authority rather than model choices.",
        "Keep tool use within the configured profile and bounded runtime policy.",
    };
    pack.decisions = {
        "Inference backend: server-context resident host.",
        std::string("Tool profile: ") + (options.tool_profile.empty() ? "none" : options.tool_profile),
    };
    if (options.default_mode == "agent") {
        pack.preferred_procedures.push_back("Prefer bounded planning plus evidence-backed synthesis before answering.");
    }
    return pack;
}

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
    policy.enable_reflection = true;
    policy.max_iterations = 2;
    policy.max_reflection_rounds = 1;
    // A zero value in host/bootstrap configuration means "use the safe
    // default". Keep daemon turns consistent with the CLI/runtime policy;
    // callers can still select a smaller positive budget explicitly.
    policy.max_tool_rounds = options.max_tool_rounds > 0
        ? options.max_tool_rounds
        : 16;
    common_tool_profile_snapshot profile_snapshot;
    std::string profile_error;
    if (resolve_common_tool_profile_snapshot(
            options.tool_profile,
            options.tool_capabilities,
            options.tool_profiles,
            profile_snapshot,
            profile_error)) {
        policy.allow_policy_gated_tool_proposals =
            profile_snapshot.allow_policy_gated_writes.value_or(false);
    }
    std::string deliberation_error;
    common_agent_deliberation_policy deliberation_policy;
    if (resolve_common_agent_deliberation_policy(
            options.thinking_mode,
            options.max_reflection_rounds,
            options.max_plan_revisions,
            options.max_research_iterations,
            deliberation_policy,
            deliberation_error)) {
        deliberation_policy.max_tool_rounds = options.max_tool_rounds > 0
            ? static_cast<int>(options.max_tool_rounds)
            : deliberation_policy.max_tool_rounds;
        policy.deliberation_policy = deliberation_policy;
    }
    return policy;
}

common_agent_runtime_config make_daemon_runtime_config(const daemon_options & options) {
    common_agent_runtime_config config;
    config.generation_config.n_predict = options.n_predict;
    config.generation_config.n_threads = options.n_threads;
    config.generation_config.context_size_tokens = static_cast<size_t>(std::max(0, options.context_size));
    // Keep daemon turns on the same host-owned family preflight path as the
    // CLI.  The daemon already exposes agent_plan=auto, but previously only
    // stored that value in orchestration config; the runtime generation flag
    // remained disabled and went straight to the full planner.
    config.generation_config.enable_tool_family_routing = options.agent_plan == "auto";
    config.generation_config.context_budgets = options.context_budgets;
    config.context_budgets = options.context_budgets;
    config.max_continuations = options.max_continuations;
    config.enable_memory_learning = options.memory_learn == "post-turn";
    config.memory_learning_config.min_confidence = options.memory_learn_min_confidence;
    config.memory_learning_config.min_expected_reuse = options.memory_learn_min_reuse;
    config.enable_adaptation_capture = options.adaptation_capture;
    config.adaptation_transaction_path = options.adaptation_transaction_path;
    config.adaptation_config.collection_allowed = options.adaptation_collection_allowed;
    config.adaptation_config.max_evidence = options.adaptation_max_evidence;
    config.adaptation_config.cause_classifier.stable_model_facing_tools =
        options.adaptation_stable_model_facing_tools;
    return config;
}

common_agent_orchestration_config make_daemon_orchestration_config(const daemon_options & options) {
    common_agent_orchestration_config config;
    config.prompt = "";
    config.agent_plan = options.agent_plan;
    config.agent_blueprint = options.agent_blueprint;
    return config;
}

common_agent_runtime_resident_request_config make_resident_request_config(
        const daemon_options & options) {
    common_agent_runtime_resident_request_config config{
        "",
        "",
        "",
        "",
        make_daemon_policy_pack(options),
        options.model,
        options.n_predict,
        options.n_gpu_layers,
        false,
        "server-context",
        common_memory_scope::session,
        common_plan_scope::turn,
        options.n_threads,
        static_cast<size_t>(std::max(0, options.context_size)),
    };
    config.mmproj = options.mmproj;
    return config;
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
        const common_agent_runtime_session_host_turn_request & request,
        std::optional<bool> allow_policy_gated_writes) {
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
    tool_request.tool_context.allowed_exposed_tool_names = request.allowed_exposed_tool_names;
    tool_request.tool_context.repository_root = options.repository_root;
    tool_request.tool_context.allow_network =
        has_enabled_mcp_provider(options.mcp_providers) ||
        !options.mcp_tool_command.empty();
    bool policy_gated_writes = false;
    if (allow_policy_gated_writes.has_value()) {
        policy_gated_writes = *allow_policy_gated_writes;
    } else if (request.allow_policy_gated_writes.has_value()) {
        policy_gated_writes = *request.allow_policy_gated_writes;
    }
    agent_tool_context_apply_policy_gated_writes(
        tool_request.tool_context, policy_gated_writes);
    tool_request.tool_context.max_calls = options.max_tool_rounds > 0 ? options.max_tool_rounds : 1;
    tool_request.tool_context.execution_control = request.execution_control;
    tool_request.tool_context.default_timeout_ms =
        options.tool_timeout_ms > 0
            ? options.tool_timeout_ms
            : tool_request.tool_context.default_timeout_ms;
    tool_request.repository_root = options.repository_root.empty()
        ? std::string()
        : std::filesystem::weakly_canonical(options.repository_root).string();
    tool_request.diagnostics = options.diagnostics;
    tool_request.resource_store_config = {
        options.resource_blob_backend,
        options.resource_blob_root,
        options.resource_metadata_backend,
        options.resource_metadata_db,
    };
    tool_request.data_store_config = {options.data_backend, options.data_db};
    tool_request.tool_capabilities = options.tool_capabilities;
    tool_request.tool_family_descriptions = options.tool_family_descriptions;
    tool_request.tool_profiles = options.tool_profiles;
    tool_request.sandbox = options.sandbox;
    tool_request.resource_processor_policies = options.resource_processor_policies;
    append_configured_stdio_mcp_providers(options.mcp_providers, tool_request.mcp_providers);
    tool_request.openapi_providers = options.openapi_providers;
    if (tool_request.mcp_providers.empty()) {
        append_legacy_stdio_mcp_provider(
            options.mcp_tool_command,
            options.mcp_tool_args,
            options.mcp_tool_server_name,
            options.mcp_tool_prefix,
            tool_request.mcp_providers);
    }
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
        std::string & error,
        common_agent_data_store * data_store,
        std::optional<bool> allow_policy_gated_writes) {
    tooling = {};
    common_memory_query query = make_daemon_memory_query(request);
    auto tool_request = make_daemon_tool_request(
        options, request, allow_policy_gated_writes);
    tool_request.data_store = data_store;
        std::string * current_plan_id = runtime != nullptr
        ? const_cast<std::string *>(&runtime->current_plan_id())
        : nullptr;
    std::unique_ptr<agent_embedding_provider> embedding_provider;
    const std::string embedding_model = options.embedding_model;
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
    for (auto & mcp_client : selection.mcp_clients) {
        tooling.owned_resources.push_back(std::static_pointer_cast<void>(
            std::shared_ptr<agent_mcp_tool_client>(std::move(mcp_client))));
    }
    for (auto & openapi_provider : selection.openapi_providers) {
        tooling.owned_resources.push_back(std::static_pointer_cast<void>(
            std::shared_ptr<agent_tool_provider>(std::move(openapi_provider))));
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
        common_agent_data_store * data_store,
        const daemon_options & options,
        const std::shared_ptr<common_agent_daemon_config_store> & config_store) {
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
        [config_store, &memory_store, &plan_store, &resource_store, data_store](
                const common_agent_runtime_resident_runtime * runtime,
                const common_agent_runtime_session_host_turn_request & request,
                common_agent_runtime_tooling & tooling,
                std::string & error) {
            const auto current_options = config_store->snapshot();
            return resolve_agent_daemon_tooling(
                *current_options,
                runtime,
                request,
                memory_store,
                plan_store,
                &resource_store,
                tooling,
                error,
                data_store);
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
    } else if (backend == "sqlite") {
#ifdef LLAMA_MEMORY_USE_SQLITE
        if (options.memory_db.empty()) { error = "--backend sqlite requires --memory-db PATH"; return false; }
        store = std::make_unique<common_memory_sqlite_store>();
#else
        error = "this binary was built without LLAMA_AGENT_STORAGE_SQLITE";
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
    } else if (backend == "sqlite") {
#ifdef LLAMA_PLAN_USE_SQLITE
        if (options.plan_db.empty()) { error = "--plan-backend sqlite requires --plan-db PATH"; return false; }
        store = std::make_unique<common_plan_sqlite_store>();
#else
        error = "this binary was built without LLAMA_AGENT_STORAGE_SQLITE";
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

bool open_daemon_data_store(
        const daemon_options & options,
        std::unique_ptr<common_agent_data_store> & store,
        std::string & error) {
    store = make_agent_data_store({options.data_backend, options.data_db}, error);
    return error.empty();
}

} // namespace

bool initialize_agent_daemon_environment(
        const daemon_options & options,
        common_agent_daemon_runtime & runtime,
        std::string & error) {
    if (!runtime.config_store) {
        runtime.config_store = std::make_shared<common_agent_daemon_config_store>(
            std::make_shared<const daemon_options>(options));
    }
    if (!open_daemon_memory_store(options, runtime.memory_store, error)) {
        return false;
    }
    if (!open_daemon_plan_store(options, runtime.plan_store, error)) {
        return false;
    }
    if (!open_daemon_data_store(options, runtime.data_store, error)) {
        return false;
    }
    if (!open_daemon_resource_store(options, runtime.resource_store, error)) {
        return false;
    }
    if (!parse_mode(options.default_mode, runtime.default_mode)) {
        error = "unsupported default mode: " + options.default_mode;
        return false;
    }

    auto session_manager_build_config = make_session_host_build_config(
        *runtime.memory_store,
        *runtime.plan_store,
        *runtime.resource_store,
        runtime.data_store.get(),
        options,
        runtime.config_store);
    runtime.inference_gate = std::make_shared<common_agent_inference_capacity_gate>(
        options.inference_max_active);
    runtime.host = std::make_unique<common_agent_runtime_session_manager>(
        make_agent_runtime_session_manager_config({
            session_manager_build_config.memory_store,
            session_manager_build_config.plan_store,
            std::move(session_manager_build_config.resident_request),
            std::move(session_manager_build_config.policy),
            std::move(session_manager_build_config.runtime_config),
            std::move(session_manager_build_config.orchestration_config),
            session_manager_build_config.memory_scope,
            session_manager_build_config.memory_enabled,
            std::move(session_manager_build_config.installed_blueprint_candidates),
            std::move(session_manager_build_config.tooling),
            std::move(session_manager_build_config.tooling_resolver),
            {},
            runtime.inference_gate,
        }));

    common_memory_store * memory_store = runtime.memory_store.get();
    common_plan_store * plan_store = runtime.plan_store.get();
    agent_resource_store * resource_store = runtime.resource_store.get();
    common_agent_data_store * data_store = runtime.data_store.get();
    runtime.tool_executor = [config_store = runtime.config_store, memory_store, plan_store, resource_store, data_store](
            const common_agent_daemon_tool_payload & payload,
            agent_tool_result & result,
            std::string & callback_error) mutable {
        if (memory_store == nullptr || plan_store == nullptr || resource_store == nullptr) {
            callback_error = "daemon tool executor stores are not initialized";
            return false;
        }
        daemon_options call_options = *config_store->snapshot();
        if (!payload.tool_profile.empty()) {
            call_options.tool_profile = payload.tool_profile;
        }
        common_agent_runtime_session_host_turn_request request;
        request.mode = common_agent_runtime_host_mode::chat;
        request.session_id = payload.session.session_id;
        request.namespace_id = payload.session.namespace_id;
        request.project_id = payload.project_id;
        request.turn_id = "mcp-tool";
        request.memory_scope = common_memory_scope::session;
        request.plan_scope = common_plan_scope::turn;

        common_agent_runtime_tooling tooling;
        if (!resolve_agent_daemon_tooling(
                call_options,
                nullptr,
                request,
                *memory_store,
                *plan_store,
                resource_store,
                tooling,
                callback_error,
                data_store,
                payload.allow_policy_gated_writes)) {
            return false;
        }
        if (tooling.tool_view == nullptr) {
            callback_error = "daemon tool executor resolved no tool view";
            return false;
        }
        agent_tool_call call{"daemon-mcp-tool", payload.tool_name, payload.arguments_json};
        if (!tooling.tool_view->validate(call, callback_error)) {
            return false;
        }
        result = tooling.tool_view->call(call, callback_error);
        return result.ok;
    };
    configure_agent_daemon_provider_probe(options, runtime);
    error.clear();
    return true;
}

void configure_agent_daemon_provider_probe(
        const daemon_options & options,
        common_agent_daemon_runtime & runtime) {
    common_memory_store * probe_memory_store = runtime.memory_store.get();
    common_plan_store * probe_plan_store = runtime.plan_store.get();
    agent_resource_store * probe_resource_store = runtime.resource_store.get();
    common_agent_data_store * probe_data_store = runtime.data_store.get();
    runtime.probe_mcp_providers = [
        probe_options = options,
        probe_memory_store,
        probe_plan_store,
        probe_resource_store,
        probe_data_store](
            common_agent_runtime_tooling & retained_tooling,
            std::vector<common_agent_daemon_provider_readiness> & provider_status,
            std::string & probe_error) {
        retained_tooling = {};
        provider_status.clear();
        bool all_required_ready = true;
        for (const auto & configured : probe_options.mcp_providers) {
            if (!configured.enabled) {
                continue;
            }

            common_agent_daemon_provider_readiness status;
            status.id = configured.id.empty() ? configured.server_name : configured.id;
            status.required = configured.required;

            daemon_options provider_options = probe_options;
            provider_options.mcp_providers = {configured};
            common_agent_runtime_session_host_turn_request probe_request;
            probe_request.mode = common_agent_runtime_host_mode::chat;
            probe_request.session_id = "daemon-provider-probe";
            probe_request.namespace_id = "local";
            probe_request.project_id = "llama-agent";
            probe_request.turn_id = "daemon-provider-probe";
            probe_request.memory_scope = common_memory_scope::session;
            probe_request.plan_scope = common_plan_scope::turn;

            common_agent_runtime_tooling provider_tooling;
            std::string provider_error;
            const bool provider_ready = resolve_agent_daemon_tooling(
                provider_options,
                nullptr,
                probe_request,
                *probe_memory_store,
                *probe_plan_store,
                probe_resource_store,
                provider_tooling,
                provider_error,
                probe_data_store);
            if (provider_ready) {
                status.status = "ready";
                for (auto & tool : provider_tooling.tools) {
                    retained_tooling.tools.push_back(std::move(tool));
                }
                for (auto & resource : provider_tooling.owned_resources) {
                    retained_tooling.owned_resources.push_back(std::move(resource));
                }
            } else {
                status.status = "degraded";
                status.warning = provider_error.empty()
                    ? "provider probe failed"
                    : provider_error;
                if (status.required) {
                    all_required_ready = false;
                }
            }
            provider_status.push_back(std::move(status));
        }
        probe_error.clear();
        return all_required_ready;
    };
}
