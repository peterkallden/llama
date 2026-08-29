#include "agent-daemon-adapter.h"
#include "agent-daemon-dispatcher.h"
#include "agent-daemon-tcp.h"
#include "agent-daemon-unix.h"
#include "../mcp/agent-mcp-http-server.h"
#include "../host/agent-host-config.h"
#include "agent/agent-inbound-contract.h"

#include "log.h"

#include <memory>
#include <thread>
#include <atomic>
#include <algorithm>
#include <filesystem>
#include <nlohmann/json.hpp>

namespace {

bool read_optional_inbound_limit(
        const agent_mcp_json & arguments,
        const char * name,
        std::optional<int> & value,
        std::string & error) {
    if (!arguments.contains(name)) return true;
    if (!arguments.at(name).is_number_integer()) {
        error = std::string(name) + " must be an integer";
        return false;
    }
    const int parsed = arguments.at(name).get<int>();
    if (parsed < 0) {
        error = std::string(name) + " must be non-negative";
        return false;
    }
    value = parsed;
    return true;
}

bool resolve_inbound_deliberation_policy(
        const common_agent_inbound_request & inbound,
        const daemon_options & options,
        common_agent_deliberation_policy & policy,
        std::string & error) {
    if (!resolve_common_agent_deliberation_policy(
            inbound.thinking_request, policy, error)) {
        return false;
    }

    if (inbound.limits.max_reflection_rounds.has_value()) {
        policy.max_reflection_rounds = *inbound.limits.max_reflection_rounds;
    }
    if (inbound.limits.max_plan_revisions.has_value()) {
        policy.max_plan_revisions = *inbound.limits.max_plan_revisions;
    }
    if (inbound.limits.max_research_iterations.has_value()) {
        policy.max_research_iterations = *inbound.limits.max_research_iterations;
    }
    if (inbound.limits.max_tool_rounds.has_value()) {
        policy.max_tool_rounds = *inbound.limits.max_tool_rounds;
    }

    if (options.max_reflection_rounds > 0 &&
            policy.max_reflection_rounds > options.max_reflection_rounds) {
        error = "inbound max_reflection_rounds exceeds host limit";
        return false;
    }
    if (options.max_plan_revisions > 0 &&
            policy.max_plan_revisions > options.max_plan_revisions) {
        error = "inbound max_plan_revisions exceeds host limit";
        return false;
    }
    if (options.max_research_iterations > 0 &&
            policy.max_research_iterations > options.max_research_iterations) {
        error = "inbound max_research_iterations exceeds host limit";
        return false;
    }
    if (options.max_tool_rounds > 0) {
        if (policy.max_tool_rounds > options.max_tool_rounds) {
            error = "inbound max_tool_rounds exceeds host limit";
            return false;
        }
    }
    error.clear();
    return true;
}

bool build_inbound_mcp_authenticator(
        const daemon_options & options,
    std::shared_ptr<const agent_mcp_authenticator> & authenticator,
    std::string & error) {
    authenticator.reset();
    if (options.http_authorization_mode == "jwt") {
        agent_mcp_jwt_authenticator_options jwt_options;
        jwt_options.issuer = options.http_jwt_issuer;
        jwt_options.audience = options.http_jwt_audience;
        jwt_options.jwks_uri = options.http_jwt_jwks_uri;
        jwt_options.allowed_algorithms = options.http_jwt_allowed_algorithms;
        jwt_options.required_scopes = options.http_jwt_required_scopes;
        jwt_options.policy_template = {
            "jwt-caller",
            options.http_jwt_audience,
            "local",
            "",
            options.http_jwt_tool_profile,
            options.http_jwt_allowed_tools,
            options.http_jwt_allow_writes,
            options.http_jwt_allow_admin,
        };
        authenticator = std::make_shared<agent_mcp_jwt_authenticator>(std::move(jwt_options));
        error.clear();
        return true;
    }
    if (options.http_token_profiles.empty()) {
        error.clear();
        return true;
    }

    auto next = std::make_shared<agent_mcp_opaque_token_authenticator>();
    for (const auto & configured : options.http_token_profiles) {
        const char * token = std::getenv(configured.token_env.c_str());
        if (token == nullptr || *token == '\0') {
            error = "HTTP bearer token environment variable is empty: " + configured.token_env;
            return false;
        }
        agent_mcp_caller_policy policy{
            configured.id,
            configured.audience,
            configured.namespace_id,
            configured.project_id,
            configured.tool_profile,
            configured.allowed_tools,
            configured.allow_writes,
            configured.allow_admin,
        };
        if (!next->register_token(token, std::move(policy), error)) {
            return false;
        }
    }
    authenticator = std::move(next);
    error.clear();
    return true;
}

bool inbound_token_profiles_equal(
        const std::vector<agent_host_mcp_inbound_token_config> & left,
        const std::vector<agent_host_mcp_inbound_token_config> & right) {
    if (left.size() != right.size()) return false;
    for (size_t i = 0; i < left.size(); ++i) {
        const auto & a = left[i];
        const auto & b = right[i];
        if (a.id != b.id || a.token_env != b.token_env || a.audience != b.audience ||
                a.namespace_id != b.namespace_id || a.project_id != b.project_id ||
                a.tool_profile != b.tool_profile || a.allowed_tools != b.allowed_tools ||
                a.allow_writes != b.allow_writes) {
            return false;
        }
    }
    return true;
}

bool inbound_auth_configuration_equal(
        const daemon_options & left,
        const daemon_options & right) {
    return left.http_authorization_mode == right.http_authorization_mode &&
        inbound_token_profiles_equal(left.http_token_profiles, right.http_token_profiles) &&
        left.http_jwt_issuer == right.http_jwt_issuer &&
        left.http_jwt_audience == right.http_jwt_audience &&
        left.http_jwt_jwks_uri == right.http_jwt_jwks_uri &&
        left.http_jwt_allowed_algorithms == right.http_jwt_allowed_algorithms &&
        left.http_jwt_required_scopes == right.http_jwt_required_scopes &&
        left.http_jwt_tool_profile == right.http_jwt_tool_profile &&
        left.http_jwt_allowed_tools == right.http_jwt_allowed_tools &&
        left.http_jwt_allow_writes == right.http_jwt_allow_writes;
}

bool tool_profile_equal(const common_tool_profile & left, const common_tool_profile & right) {
    if (left.id != right.id || left.description != right.description ||
            left.enabled != right.enabled || left.include_capabilities != right.include_capabilities ||
            left.exclude_capabilities != right.exclude_capabilities ||
            left.allow_network != right.allow_network ||
            left.allow_policy_gated_writes != right.allow_policy_gated_writes ||
            left.members.size() != right.members.size()) {
        return false;
    }
    for (size_t i = 0; i < left.members.size(); ++i) {
        const auto & a = left.members[i];
        const auto & b = right.members[i];
        if (a.tool_name != b.tool_name || a.tool_version != b.tool_version ||
                a.enabled != b.enabled || a.config_override_json != b.config_override_json) {
            return false;
        }
    }
    return true;
}

bool tool_profiles_equal(
        const std::map<std::string, common_tool_profile> & left,
        const std::map<std::string, common_tool_profile> & right) {
    if (left.size() != right.size()) return false;
    for (const auto & [id, profile] : left) {
        const auto next = right.find(id);
        if (next == right.end() || !tool_profile_equal(profile, next->second)) return false;
    }
    return true;
}

bool tool_capabilities_equal(
        const std::map<std::string, std::vector<std::string>> & left,
        const std::map<std::string, std::vector<std::string>> & right) {
    return left == right;
}

bool configured_profile_allows_policy_gated_writes(const daemon_options & options) {
    common_tool_profile_snapshot snapshot;
    std::string error;
    if (!resolve_common_tool_profile_snapshot(
            options.tool_profile, options.tool_capabilities, options.tool_profiles,
            snapshot, error)) {
        return false;
    }
    return snapshot.allow_policy_gated_writes.value_or(false);
}

} // namespace

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
    auto build_http_tool_catalog = [catalog_memory_store, catalog_plan_store, catalog_resource_store](
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
        if (!agent_mcp_register_native_tool_view(
                catalog_tooling.tool_view->chat_tools(),
                [&tool_view = catalog_tooling.tool_view](const std::string & name) {
                    return tool_view->is_read_only(name);
                },
                [&tool_view = catalog_tooling.tool_view](const std::string & name) {
                    return tool_view->is_policy_gated(name);
                },
                registry,
                catalog_error)) {
            return false;
        }
        catalog_error.clear();
        return true;
    };
    auto config_version = std::make_shared<std::atomic<uint64_t>>(1);
    auto refresh_http_catalog = std::make_shared<std::function<bool(
        const daemon_options &, std::string &)>>();
    auto refresh_http_auth = std::make_shared<std::function<bool(
        const daemon_options &, std::string &)>>();
    const auto config_store = runtime.config_store;
    runtime.reload_config = [config_store, config_version, refresh_http_catalog, refresh_http_auth](
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
        require_restart(candidate.model_profile != options.model_profile, "model.profile");
        require_restart(
            common_agent_model_catalog_to_json(candidate.model_catalog) !=
                common_agent_model_catalog_to_json(options.model_catalog),
            "models");
        require_restart(candidate.embedding_model != options.embedding_model, "model.embedding_model");
        require_restart(candidate.backend != options.backend, "stores.memory.backend");
        require_restart(candidate.memory_db != options.memory_db, "stores.memory.path");
        require_restart(candidate.plan_backend != options.plan_backend, "stores.plan.backend");
        require_restart(candidate.plan_db != options.plan_db, "stores.plan.path");
        require_restart(candidate.data_backend != options.data_backend, "stores.data.backend");
        require_restart(candidate.data_db != options.data_db, "stores.data.path");
        require_restart(candidate.resource_blob_backend != options.resource_blob_backend, "resources.blob_backend");
        require_restart(candidate.resource_blob_root != options.resource_blob_root, "resources.blob_root");
        require_restart(candidate.resource_metadata_backend != options.resource_metadata_backend, "resources.metadata_backend");
        require_restart(candidate.resource_metadata_db != options.resource_metadata_db, "resources.metadata_db");
        require_restart(candidate.queue_capacity != options.queue_capacity, "limits.queue_capacity");
        require_restart(candidate.worker_count != options.worker_count, "limits.worker_count");
        require_restart(candidate.inference_max_active != options.inference_max_active, "limits.inference_max_active");
        require_restart(candidate.http_enabled != options.http_enabled, "mcp.inbound.enabled");
        require_restart(candidate.http_listen_address != options.http_listen_address, "mcp.inbound.listen");
        require_restart(candidate.http_port != options.http_port, "mcp.inbound.port");
        require_restart(candidate.http_path != options.http_path, "mcp.inbound.path");
        require_restart(candidate.http_allowed_origin != options.http_allowed_origin, "mcp.inbound.allowed_origin");
        require_restart(candidate.default_mode != options.default_mode, "runtime.default_mode");
        require_restart(candidate.n_predict != options.n_predict, "runtime.n_predict");
        require_restart(candidate.n_threads != options.n_threads, "runtime.n_threads");
        require_restart(candidate.n_gpu_layers != options.n_gpu_layers, "runtime.n_gpu_layers");
        require_restart(candidate.max_continuations != options.max_continuations, "limits.max_continuations");
        require_restart(candidate.thinking_mode != options.thinking_mode, "runtime.thinking_mode");
        require_restart(candidate.max_reflection_rounds != options.max_reflection_rounds, "runtime.max_reflection_rounds");
        require_restart(candidate.max_plan_revisions != options.max_plan_revisions, "runtime.max_plan_revisions");
        require_restart(candidate.max_research_iterations != options.max_research_iterations, "runtime.max_research_iterations");
        require_restart(candidate.memory_learn != options.memory_learn, "runtime.memory_learn");
        require_restart(candidate.agent_plan != options.agent_plan, "runtime.agent_plan");
        require_restart(candidate.tool_profile != options.tool_profile, "tools.profile");
        require_restart(!tool_capabilities_equal(candidate.tool_capabilities, options.tool_capabilities),
            "tools.capabilities");
        require_restart(candidate.tool_family_descriptions != options.tool_family_descriptions,
            "tools.families");
        require_restart(!tool_profiles_equal(candidate.tool_profiles, options.tool_profiles),
            "tools.profiles");
        auto provider_equal = [](const auto & a, const auto & b) {
            return a.id == b.id && a.enabled == b.enabled && a.required == b.required &&
                a.type == b.type && a.transport == b.transport &&
                a.command == b.command && a.url == b.url &&
                a.auth.type == b.auth.type && a.auth.scheme == b.auth.scheme &&
                a.auth.token_url == b.auth.token_url && a.auth.scopes == b.auth.scopes &&
                a.auth.token_env == b.auth.token_env &&
                a.auth.username_env == b.auth.username_env &&
                a.auth.password_env == b.auth.password_env &&
                a.auth.client_id_env == b.auth.client_id_env &&
                a.auth.client_secret_env == b.auth.client_secret_env &&
                a.auth.client_cert_path_env == b.auth.client_cert_path_env &&
                a.auth.client_key_path_env == b.auth.client_key_path_env &&
                a.auth.ca_cert_path_env == b.auth.ca_cert_path_env &&
                a.allowed_tools == b.allowed_tools &&
                a.connect_timeout_ms == b.connect_timeout_ms &&
                a.request_timeout_ms == b.request_timeout_ms &&
                a.shutdown_timeout_ms == b.shutdown_timeout_ms &&
                a.max_result_bytes == b.max_result_bytes &&
                a.prefix == b.prefix && a.server_name == b.server_name;
        };
        auto openapi_provider_equal = [](const auto & a, const auto & b) {
            if (a.operations.size() != b.operations.size()) return false;
            for (const auto & operation : a.operations) {
                const auto other = b.operations.find(operation.first);
                if (other == b.operations.end() ||
                        other->second.enabled != operation.second.enabled ||
                        other->second.access != operation.second.access) {
                    return false;
                }
            }
            return a.id == b.id && a.enabled == b.enabled && a.required == b.required &&
                a.type == b.type && a.spec_path == b.spec_path && a.base_url == b.base_url &&
                a.source_directory == b.source_directory &&
                a.prefix == b.prefix && a.access == b.access && a.exposure == b.exposure &&
                a.auth.type == b.auth.type && a.auth.scheme == b.auth.scheme &&
                a.auth.token_url == b.auth.token_url && a.auth.scopes == b.auth.scopes &&
                a.auth.token_env == b.auth.token_env &&
                a.auth.username_env == b.auth.username_env &&
                a.auth.password_env == b.auth.password_env &&
                a.auth.client_id_env == b.auth.client_id_env &&
                a.auth.client_secret_env == b.auth.client_secret_env &&
                a.auth.client_cert_path_env == b.auth.client_cert_path_env &&
                a.auth.client_key_path_env == b.auth.client_key_path_env &&
                a.auth.ca_cert_path_env == b.auth.ca_cert_path_env &&
                a.allow_private_network == b.allow_private_network &&
                a.connect_timeout_ms == b.connect_timeout_ms &&
                a.request_timeout_ms == b.request_timeout_ms &&
                a.max_result_bytes == b.max_result_bytes;
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
        for (const auto & next : candidate.openapi_providers) {
            auto current = std::find_if(options.openapi_providers.begin(), options.openapi_providers.end(),
                [&](const auto & provider) { return provider.id == next.id; });
            if (current == options.openapi_providers.end()) {
                result.providers_added.push_back(next.id);
            } else if (!openapi_provider_equal(*current, next)) {
                result.providers_replaced.push_back(next.id);
            }
        }
        for (const auto & current : options.openapi_providers) {
            auto next = std::find_if(candidate.openapi_providers.begin(), candidate.openapi_providers.end(),
                [&](const auto & provider) { return provider.id == current.id; });
            if (next == candidate.openapi_providers.end()) {
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
            if (candidate.repository_root != options.repository_root) {
                result.applied_fields.emplace_back("tools.repository_root");
            }
        }
        const bool auth_configuration_changed = !inbound_auth_configuration_equal(candidate, options);
        if (auth_configuration_changed && *refresh_http_auth) {
            if (!(*refresh_http_auth)(candidate, reload_error)) {
                result.warning = "inbound MCP authentication policy was not applied";
                return false;
            }
            result.applied_fields.emplace_back("mcp.inbound.authorization");
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
    if (options.http_enabled && !build_http_tool_catalog(options, http_registry, error)) {
        std::fprintf(stderr, "failed to resolve daemon MCP HTTP tool catalog: %s\n", error.c_str());
        return 2;
    }
    std::unique_ptr<agent_mcp_http_server> http_server;
    std::thread http_thread;
    std::shared_ptr<const agent_mcp_authenticator> inbound_authenticator;
    if (options.http_enabled || options.tcp_enabled) {
        if (!build_inbound_mcp_authenticator(options, inbound_authenticator, error)) {
            std::fprintf(stderr, "failed to configure inbound authentication: %s\n", error.c_str());
            return 2;
        }
    }
    if (options.tcp_enabled && !inbound_authenticator) {
        std::fprintf(stderr, "TCP mode requires configured inbound authentication\n");
        return 2;
    }
    if (options.http_enabled) {
        agent_mcp_http_server_options http_options;
        http_options.listen_address = options.http_listen_address;
        http_options.port = options.http_port;
        http_options.path = options.http_path;
        http_options.allowed_origin = options.http_allowed_origin;
        http_options.bearer_token = options.http_bearer_token;
        http_options.authenticator = inbound_authenticator;
        http_options.max_body_bytes = options.http_max_body_bytes;
        http_options.max_result_bytes = options.http_max_result_bytes;
        http_options.server_name = "llama-agent-daemon-mcp";
        http_options.agent_tools_enabled = options.http_agent_tools_enabled;
        http_options.tasks_enabled = options.http_agent_tools_enabled;
        if (http_options.tasks_enabled) {
            http_options.protocol_version = "2025-11-25";
        }
        http_options.max_delegation_depth = options.http_max_delegation_depth;
        http_options.default_policy = {
            "daemon-http",
            "llama-agent",
            "local",
            "",
            options.tool_profile,
            {},
            configured_profile_allows_policy_gated_writes(options),
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
        http_options.execute_agent_tool = [&dispatcher, &options](
                const agent_mcp_caller_policy & policy,
                const std::string & operation,
                const agent_mcp_json & arguments,
                agent_mcp_server_tool_result & result,
                std::string & callback_error) {
            const std::string input = arguments.value("task", arguments.value("text", std::string()));
            if (input.empty()) {
                callback_error = "agent delegation requires a non-empty task or text";
                return false;
            }

            static std::atomic<uint64_t> next_inbound_id = 1;
            const uint64_t inbound_id = next_inbound_id.fetch_add(1);
            common_agent_inbound_request inbound;
            inbound.request_id = policy.caller_id + "-delegation-" + std::to_string(inbound_id);
            inbound.caller_id = policy.caller_id;
            inbound.task = input;
            inbound.namespace_id = policy.namespace_id;
            inbound.project_id = policy.project_id;
            inbound.session_id = policy.caller_id + "-delegated";
            inbound.turn_id = "delegated-" + std::to_string(inbound_id);
            inbound.delegation_depth = arguments.value("delegation_depth", 0);

            const std::string thinking_mode = arguments.value("thinking_mode", "reflective");
            if (!parse_common_agent_thinking_request(thinking_mode, inbound.thinking_request)) {
                callback_error = "unsupported inbound thinking_mode: " + thinking_mode;
                return false;
            }
            if (!read_optional_inbound_limit(
                    arguments, "max_reflection_rounds", inbound.limits.max_reflection_rounds, callback_error) ||
                    !read_optional_inbound_limit(
                    arguments, "max_plan_revisions", inbound.limits.max_plan_revisions, callback_error)) {
                return false;
            }
            if (arguments.contains("max_research_iterations")) {
                if (!arguments["max_research_iterations"].is_number_integer() ||
                        arguments["max_research_iterations"].get<int>() < 0) {
                    callback_error = "max_research_iterations must be a non-negative integer";
                    return false;
                }
                inbound.limits.max_research_iterations =
                    static_cast<size_t>(arguments["max_research_iterations"].get<int>());
            }
            if (arguments.contains("max_tool_rounds")) {
                if (!arguments["max_tool_rounds"].is_number_integer() ||
                        arguments["max_tool_rounds"].get<int>() < 0) {
                    callback_error = "max_tool_rounds must be a non-negative integer";
                    return false;
                }
                inbound.limits.max_tool_rounds =
                    static_cast<size_t>(arguments["max_tool_rounds"].get<int>());
            }
            if (arguments.contains("resource_refs")) {
                if (!arguments["resource_refs"].is_array()) {
                    callback_error = "resource_refs must be an array";
                    return false;
                }
                for (const auto & value : arguments["resource_refs"]) {
                    if (!value.is_string() || value.get<std::string>().empty()) {
                        callback_error = "resource_refs must contain non-empty strings";
                        return false;
                    }
                    common_agent_input_resource input_resource;
                    input_resource.resource.uri = value.get<std::string>();
                    input_resource.role = "reference";
                    inbound.input_resources.push_back(std::move(input_resource));
                }
            }

            common_agent_deliberation_policy inbound_policy;
            if (!resolve_inbound_deliberation_policy(inbound, options, inbound_policy, callback_error)) {
                return false;
            }

            common_agent_daemon_command command;
            command.request_id = inbound.request_id;
            command.type = common_agent_daemon_command_type::run_turn;
            command.turn = common_agent_daemon_turn_payload{};
            // Agent-to-agent callers need a compact handoff summary by default.
            // This remains scoped to the delegated turn; ordinary client turns
            // still opt in through their own include_summary field.
            command.turn->include_summary = true;
            command.turn->request.request_id = command.request_id;
            command.turn->request.turn.mode = common_agent_runtime_host_mode::agent;
            command.turn->request.turn.prompt = operation == "summarize"
                ? "Summarize the following text concisely:\n\n" + input
                : operation == "review_plan"
                    ? "Review the following plan for risks, omissions, and next steps:\n\n" + input
                    : input;
            command.turn->request.turn.session_id = inbound.session_id;
            command.turn->request.turn.namespace_id = inbound.namespace_id;
            command.turn->request.turn.project_id = inbound.project_id;
            command.turn->request.turn.turn_id = inbound.turn_id;
            command.turn->request.turn.deliberation_policy_override = inbound_policy;
            command.turn->request.turn.input_resources = std::move(inbound.input_resources);
            command.turn->request.turn.allow_policy_gated_writes = policy.allow_writes;
            if (!policy.allowed_tools.empty()) {
                command.turn->request.turn.allowed_exposed_tool_names = policy.allowed_tools;
            }
            command.turn->request.turn.n_predict = static_cast<int>(options.n_predict);
            command.turn->request.turn.execution_control = make_common_agent_runtime_execution_control({
                options.turn_timeout_ms,
                options.inference_step_timeout_ms,
                options.tool_timeout_ms,
                options.mcp_connect_timeout_ms,
                options.mcp_request_timeout_ms,
                options.mcp_shutdown_timeout_ms,
            });
            common_agent_daemon_command_result command_result;
            if (!dispatcher.execute(command, command_result, callback_error)) {
                return false;
            }
            result.ok = command_result.turn_result.ok;
            result.structured_content = {
                {"operation", operation},
                {"request_id", command.request_id},
                {"operation_id", command.request_id},
                {"thinking_mode", common_agent_thinking_mode_name(inbound_policy.mode)},
                {"response", command_result.turn_result.response},
                {"plan_id", command_result.turn_result.plan_id},
                {"event_count", command_result.turn_result.event_count},
                {"trace_count", command_result.turn_result.trace_count},
            };
            if (command_result.turn_summary.has_value()) {
                const auto & summary = *command_result.turn_summary;
                result.structured_content["turn_summary"] = {
                    {"mode", summary.mode},
                    {"status", summary.status},
                    {"objective", summary.objective},
                    {"phases", summary.phases},
                    {"tools_used", summary.tools_used},
                    {"plan_revisions", summary.plan_revisions},
                    {"sources", summary.sources},
                    {"evidence_items", summary.evidence_items},
                    {"unresolved_items", summary.unresolved_items},
                    {"verified", summary.verified},
                    {"stop_reason", summary.stop_reason},
                    {"unresolved", summary.unresolved},
                };
            }
            result.content = {{{"type", "text"}, {"text", command_result.turn_result.response}}};
            result.safe_summary = command_result.turn_result.response;
            if (!result.ok) {
                result.failure_code = "agent.delegation_failed";
                result.failure_class = command_result.turn_result.failure_class == common_agent_failure_class::timeout
                    ? "timeout" : "execution";
                callback_error = command_result.turn_result.error;
            }
            return result.ok;
        };
        http_options.subscribe_events = [&dispatcher](common_agent_event_stream_subscription subscription) {
            return dispatcher.subscribe_events(std::move(subscription));
        };
        http_options.unsubscribe_events = [&dispatcher](const std::string & subscription_id) {
            dispatcher.unsubscribe_events(subscription_id);
        };
        http_options.wait_for_event = [&dispatcher](
                const std::string & subscription_id,
                common_agent_event_stream_delivery & delivery,
                std::chrono::milliseconds timeout) {
            return dispatcher.wait_for_event(subscription_id, delivery, timeout);
        };
        http_server = std::make_unique<agent_mcp_http_server>(std::move(http_registry), std::move(http_options));
        *refresh_http_catalog = [&dispatcher, &http_server, &build_http_tool_catalog](
            const daemon_options & current_options,
            std::string & refresh_error) {
            agent_mcp_server_tool_registry next_registry;
            if (!build_http_tool_catalog(current_options, next_registry, refresh_error)) {
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
                configured_profile_allows_policy_gated_writes(current_options),
            });
            refresh_error.clear();
            return true;
        };
        *refresh_http_auth = [&http_server](
                const daemon_options & current_options,
                std::string & refresh_error) {
            std::shared_ptr<const agent_mcp_authenticator> next_authenticator;
            if (!build_inbound_mcp_authenticator(current_options, next_authenticator, refresh_error)) {
                return false;
            }
            http_server->replace_authenticator(std::move(next_authenticator));
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

    const bool jsonl_ok = options.unix_socket_enabled
        ? run_agent_daemon_unix_socket_adapter(options, config_store, dispatcher, inbound_authenticator, error)
        : options.tcp_enabled
            ? run_agent_daemon_tcp_adapter(options, config_store, dispatcher, inbound_authenticator, error)
            : run_agent_daemon_jsonl_adapter(stdin, stdout, options, config_store, dispatcher, error);
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
