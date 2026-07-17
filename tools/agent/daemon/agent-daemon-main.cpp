#include "agent-daemon-adapter.h"
#include "agent-daemon-dispatcher.h"
#include "../mcp/agent-mcp-http-server.h"
#include "../host/agent-host-config.h"

#include "log.h"

#include <memory>
#include <thread>
#include <atomic>
#include <algorithm>
#include <filesystem>
#include <nlohmann/json.hpp>

int main(int argc, char ** argv) {
    daemon_options options;
    if (!parse_agent_daemon_args(argc, argv, options)) {
        print_agent_daemon_usage(argv[0]);
        return 2;
    }
    common_log_set_verbosity_thold(LOG_LEVEL_WARN);

    common_agent_daemon_runtime runtime;
    std::string error;
    if (!initialize_agent_daemon_environment(options, runtime, error)) {
        std::fprintf(stderr, "failed to initialize daemon environment: %s\n", error.c_str());
        return 2;
    }
    common_memory_store * catalog_memory_store = runtime.memory_store.get();
    common_plan_store * catalog_plan_store = runtime.plan_store.get();
    agent_resource_store * catalog_resource_store = runtime.resource_store.get();
    runtime.build_http_tool_catalog = [catalog_memory_store, catalog_plan_store, catalog_resource_store](
            const daemon_options & catalog_options,
            agent_mcp_server_tool_registry & registry,
            std::string & catalog_error) {
        common_agent_runtime_session_host_turn_request catalog_request;
        catalog_request.mode = common_agent_runtime_host_mode::chat;
        catalog_request.session_id = "daemon-http-catalog";
        catalog_request.namespace_id = "local";
        catalog_request.project_id = "llama-agent";
        catalog_request.turn_id = "daemon-http-catalog";
        catalog_request.memory_scope = common_memory_scope::session;
        catalog_request.plan_scope = common_plan_scope::turn;
        common_agent_runtime_tooling catalog_tooling;
        if (!resolve_agent_daemon_tooling(
                catalog_options,
                nullptr,
                catalog_request,
                *catalog_memory_store,
                *catalog_plan_store,
                catalog_resource_store,
                catalog_tooling,
                catalog_error)) {
            return false;
        }
        if (catalog_tooling.tool_view == nullptr) {
            catalog_error = "daemon HTTP catalog resolved no tool view";
            return false;
        }
        for (const auto & tool : catalog_tooling.tool_view->chat_tools()) {
            const bool read_only = catalog_tooling.tool_view->is_read_only(tool.name);
            const bool policy_gated = catalog_tooling.tool_view->is_policy_gated(tool.name);
            if (!registry.register_tool({
                    tool.name,
                    tool.description,
                    tool.parameters,
                    read_only,
                    policy_gated,
                    false,
                    false,
                    false,
                    [](const agent_mcp_json &, agent_mcp_server_tool_result &, std::string & handler_error) {
                        handler_error = "daemon HTTP catalog tools require the dispatcher executor";
                        return false;
                    },
                }, catalog_error)) {
                return false;
            }
        }
        catalog_error.clear();
        return true;
    };
    auto config_version = std::make_shared<std::atomic<uint64_t>>(1);
    auto refresh_http_catalog = std::make_shared<std::function<bool(
        const daemon_options &, std::string &)>>();
    const auto config_store = runtime.config_store;
    runtime.reload_config = [config_store, config_version, refresh_http_catalog](
            const std::string & path,
            common_agent_daemon_reload_result & result,
            std::string & reload_error) {
        agent_host_config config;
        if (!load_agent_host_config(path, config, reload_error)) {
            return false;
        }
        const auto active_snapshot = config_store->snapshot();
        const daemon_options & options = *active_snapshot;
        daemon_options candidate = options;
        apply_agent_host_config_to_daemon_options(config, candidate);
        if (!candidate.repository_root.empty()) {
            std::error_code root_error;
            const auto canonical_root = std::filesystem::weakly_canonical(
                candidate.repository_root,
                root_error);
            if (root_error || !std::filesystem::is_directory(canonical_root, root_error)) {
                reload_error = "tools.repository_root must resolve to a directory";
                return false;
            }
            candidate.repository_root = canonical_root.string();
        }

        auto require_restart = [&](bool changed, const char * field) {
            if (changed) {
                result.restart_required.emplace_back(field);
            }
        };
        require_restart(candidate.model != options.model, "model.path");
        require_restart(candidate.embedding_model != options.embedding_model, "model.embedding_model");
        require_restart(candidate.backend != options.backend, "stores.memory.backend");
        require_restart(candidate.memory_db != options.memory_db, "stores.memory.path");
        require_restart(candidate.plan_backend != options.plan_backend, "stores.plan.backend");
        require_restart(candidate.plan_db != options.plan_db, "stores.plan.path");
        require_restart(candidate.resource_blob_backend != options.resource_blob_backend, "resources.blob_backend");
        require_restart(candidate.resource_blob_root != options.resource_blob_root, "resources.blob_root");
        require_restart(candidate.resource_metadata_backend != options.resource_metadata_backend, "resources.metadata_backend");
        require_restart(candidate.resource_metadata_db != options.resource_metadata_db, "resources.metadata_db");
        require_restart(candidate.queue_capacity != options.queue_capacity, "limits.queue_capacity");
        require_restart(candidate.worker_count != options.worker_count, "limits.worker_count");
        require_restart(candidate.default_mode != options.default_mode, "runtime.default_mode");
        require_restart(candidate.n_predict != options.n_predict, "runtime.n_predict");
        require_restart(candidate.n_gpu_layers != options.n_gpu_layers, "runtime.n_gpu_layers");
        require_restart(candidate.planning_mode != options.planning_mode, "runtime.planning_mode");
        require_restart(candidate.reflection_mode != options.reflection_mode, "runtime.reflection_mode");
        require_restart(candidate.memory_learn != options.memory_learn, "runtime.memory_learn");
        require_restart(candidate.agent_plan != options.agent_plan, "runtime.agent_plan");
        auto provider_equal = [](const auto & a, const auto & b) {
            return a.id == b.id && a.enabled == b.enabled &&
                a.type == b.type && a.transport == b.transport &&
                a.command == b.command && a.url == b.url &&
                a.token_env == b.token_env && a.allowed_tools == b.allowed_tools &&
                a.connect_timeout_ms == b.connect_timeout_ms &&
                a.request_timeout_ms == b.request_timeout_ms &&
                a.shutdown_timeout_ms == b.shutdown_timeout_ms &&
                a.max_result_bytes == b.max_result_bytes &&
                a.prefix == b.prefix && a.server_name == b.server_name;
        };
        for (const auto & next : candidate.mcp_providers) {
            auto current = std::find_if(options.mcp_providers.begin(), options.mcp_providers.end(),
                [&](const auto & provider) { return provider.id == next.id; });
            if (current == options.mcp_providers.end()) {
                result.providers_added.push_back(next.id);
            } else if (!provider_equal(*current, next)) {
                result.providers_replaced.push_back(next.id);
            }
        }
        for (const auto & current : options.mcp_providers) {
            auto next = std::find_if(candidate.mcp_providers.begin(), candidate.mcp_providers.end(),
                [&](const auto & provider) { return provider.id == current.id; });
            if (next == candidate.mcp_providers.end()) {
                result.providers_removed.push_back(current.id);
            }
        }

        if (!result.restart_required.empty()) {
            result.warning = "configuration was not applied; restart the daemon to change the listed fields";
            reload_error.clear();
            return true;
        }

        const bool providers_changed = !result.providers_added.empty() ||
            !result.providers_removed.empty() || !result.providers_replaced.empty();
        const bool native_catalog_changed = providers_changed ||
            candidate.tool_profile != options.tool_profile ||
            candidate.repository_root != options.repository_root;
        if (native_catalog_changed) {
            if (*refresh_http_catalog) {
                if (!(*refresh_http_catalog)(candidate, reload_error)) {
                    result.warning = "native/MCP tool catalog change was not applied to the inbound HTTP catalog";
                    return false;
                }
            }
            if (providers_changed) {
                result.applied_fields.emplace_back("tools.providers");
            }
            if (candidate.tool_profile != options.tool_profile) {
                result.applied_fields.emplace_back("tools.profile");
            }
            if (candidate.repository_root != options.repository_root) {
                result.applied_fields.emplace_back("tools.repository_root");
            }
        }
        if (candidate.turn_timeout_ms != options.turn_timeout_ms ||
                candidate.max_turn_seconds != options.max_turn_seconds) {
            result.applied_fields.emplace_back("limits.turn_timeout_ms");
        }
        if (candidate.inference_step_timeout_ms != options.inference_step_timeout_ms) {
            result.applied_fields.emplace_back("limits.inference_step_timeout_ms");
        }
        if (candidate.tool_timeout_ms != options.tool_timeout_ms) {
            result.applied_fields.emplace_back("limits.tool_timeout_ms");
        }
        if (candidate.mcp_connect_timeout_ms != options.mcp_connect_timeout_ms) {
            result.applied_fields.emplace_back("limits.mcp_connect_timeout_ms");
        }
        if (candidate.mcp_request_timeout_ms != options.mcp_request_timeout_ms) {
            result.applied_fields.emplace_back("limits.mcp_request_timeout_ms");
        }
        if (candidate.mcp_shutdown_timeout_ms != options.mcp_shutdown_timeout_ms) {
            result.applied_fields.emplace_back("limits.mcp_shutdown_timeout_ms");
        }
        if (candidate.max_tool_rounds != options.max_tool_rounds) {
            result.applied_fields.emplace_back("limits.max_tool_rounds");
        }
        config_store->replace(std::make_shared<const daemon_options>(std::move(candidate)));
        result.config_version = config_version->fetch_add(1) + 1;
        reload_error.clear();
        return true;
    };
    agent_mcp_server_tool_registry http_registry;
    common_agent_daemon_dispatcher dispatcher(std::move(runtime), options.queue_capacity, options.worker_count);
    if (options.http_enabled && !dispatcher.build_http_tool_catalog(options, http_registry, error)) {
        std::fprintf(stderr, "failed to resolve daemon MCP HTTP tool catalog: %s\n", error.c_str());
        return 2;
    }
    std::unique_ptr<agent_mcp_http_server> http_server;
    std::thread http_thread;
    if (options.http_enabled) {
        agent_mcp_http_server_options http_options;
        http_options.listen_address = options.http_listen_address;
        http_options.port = options.http_port;
        http_options.path = options.http_path;
        http_options.allowed_origin = options.http_allowed_origin;
        http_options.bearer_token = options.http_bearer_token;
        http_options.max_body_bytes = options.http_max_body_bytes;
        http_options.max_result_bytes = options.http_max_result_bytes;
        http_options.server_name = "llama-agent-daemon-mcp";
        http_options.default_policy = {
            "daemon-http",
            "llama-agent",
            "local",
            "",
            options.tool_profile,
            {},
            options.tool_profile == "memory" || options.tool_profile == "research",
        };
        http_options.execute_tool = [&dispatcher](
                const agent_mcp_caller_policy & policy,
                const std::string & tool_name,
                const agent_mcp_json & arguments,
                agent_mcp_server_tool_result & result,
                std::string & callback_error) {
            common_agent_daemon_command command;
            command.request_id = "mcp-http-tool";
            command.type = common_agent_daemon_command_type::execute_tool;
            command.tool = common_agent_daemon_tool_payload{
                {policy.namespace_id, policy.caller_id + "-session"},
                policy.project_id,
                policy.tool_profile,
                tool_name,
                arguments.dump(),
                policy.allow_writes,
            };
            common_agent_daemon_command_result daemon_result;
            if (!dispatcher.execute(command, daemon_result, callback_error)) {
                result.ok = false;
                result.safe_summary = callback_error;
                result.content = {{{"type", "text"}, {"text", callback_error}}};
                return false;
            }
            result.ok = daemon_result.tool_result.ok;
            result.structured_content = nlohmann::ordered_json::parse(
                daemon_result.tool_result.content_json, nullptr, false);
            if (result.structured_content.is_discarded()) {
                result.structured_content = agent_mcp_json::object();
            }
            result.safe_summary = daemon_result.tool_result.content_summary;
            result.content = {{{"type", "text"}, {"text", result.safe_summary}}};
            callback_error = daemon_result.error;
            return result.ok;
        };
        http_server = std::make_unique<agent_mcp_http_server>(std::move(http_registry), std::move(http_options));
        *refresh_http_catalog = [&dispatcher, &http_server](
                const daemon_options & current_options,
                std::string & refresh_error) {
            agent_mcp_server_tool_registry next_registry;
            if (!dispatcher.build_http_tool_catalog(current_options, next_registry, refresh_error)) {
                return false;
            }
            if (!http_server->replace_registry(std::move(next_registry), refresh_error)) {
                return false;
            }
            http_server->replace_default_policy({
                "daemon-http",
                "llama-agent",
                "local",
                "",
                current_options.tool_profile,
                {},
                current_options.tool_profile == "memory" || current_options.tool_profile == "research",
            });
            refresh_error.clear();
            return true;
        };
        if (!http_server->bind(error)) {
            std::fprintf(stderr, "failed to bind daemon MCP HTTP server: %s\n", error.c_str());
            return 2;
        }
        http_thread = std::thread([&http_server, &error]() {
            http_server->listen(error);
        });
    }

    const bool jsonl_ok = run_agent_daemon_jsonl_adapter(
        stdin,
        stdout,
        options,
        config_store,
        dispatcher,
        error);
    if (http_server) {
        http_server->stop();
        if (http_thread.joinable()) http_thread.join();
    }
    if (!jsonl_ok) {
        if (!error.empty()) {
            std::fprintf(stderr, "daemon adapter failed: %s\n", error.c_str());
        }
        return 2;
    }

    return 0;
}
