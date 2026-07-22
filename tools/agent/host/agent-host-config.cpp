#include "agent-host-config.h"
#include "../daemon/agent-daemon-adapter.h"
#include "agent/deliberation-policy.h"

#include <fstream>
#include <nlohmann/json.hpp>

#include <unordered_set>

using json = nlohmann::ordered_json;

namespace {

template<typename T>
void read_optional(
        const json & value,
        const char * key,
        T & target) {
    if (value.contains(key) && !value[key].is_null()) {
        target = value[key].get<T>();
    }
}

bool read_command_array(
        const json & value,
        std::vector<std::string> & command,
        std::string & error) {
    if (!value.is_array()) {
        error = "MCP provider command must be an array of strings";
        return false;
    }
    command.clear();
    for (const auto & entry : value) {
        if (!entry.is_string()) {
            error = "MCP provider command entries must be strings";
            return false;
        }
        command.push_back(entry.get<std::string>());
    }
    if (command.empty()) {
        error = "MCP provider command must not be empty";
        return false;
    }
    error.clear();
    return true;
}

bool read_string_array(
        const json & value,
        std::vector<std::string> & target,
        const char * field,
        std::string & error) {
    if (!value.is_array()) {
        error = std::string(field) + " must be an array of strings";
        return false;
    }
    target.clear();
    for (const auto & entry : value) {
        if (!entry.is_string()) {
            error = std::string(field) + " entries must be strings";
            return false;
        }
        target.push_back(entry.get<std::string>());
    }
    error.clear();
    return true;
}

bool read_sandbox_network(
        const json & value,
        common_agent_sandbox_network_scope & target,
        std::string & error) {
    if (!value.is_string()) { error = "sandbox network must be a string"; return false; }
    const auto name = value.get<std::string>();
    if (name == "none") target = common_agent_sandbox_network_scope::none;
    else if (name == "dns_only") target = common_agent_sandbox_network_scope::dns_only;
    else if (name == "allowlisted") target = common_agent_sandbox_network_scope::allowlisted;
    else if (name == "package_registry") target = common_agent_sandbox_network_scope::package_registry;
    else if (name == "research_web") target = common_agent_sandbox_network_scope::research_web;
    else { error = "unsupported sandbox network scope: " + name; return false; }
    return true;
}

bool read_sandbox_filesystem(
        const json & value,
        common_agent_sandbox_filesystem_scope & target,
        std::string & error) {
    if (!value.is_string()) { error = "sandbox filesystem must be a string"; return false; }
    const auto name = value.get<std::string>();
    if (name == "readonly") target = common_agent_sandbox_filesystem_scope::readonly;
    else if (name == "workspace_write") target = common_agent_sandbox_filesystem_scope::workspace_write;
    else if (name == "artifact_write") target = common_agent_sandbox_filesystem_scope::artifact_write;
    else { error = "unsupported sandbox filesystem scope: " + name; return false; }
    return true;
}

json sandbox_class_to_json(const common_agent_sandbox_policy & policy) {
    return {
        {"image", policy.image},
        {"timeout_ms", policy.limits.timeout_ms},
        {"memory_bytes", policy.limits.memory_bytes},
        {"cpu_count", policy.limits.cpu_count},
        {"process_count", policy.limits.process_count},
        {"max_output_bytes", policy.limits.max_output_bytes},
        {"network", [&policy]() {
            switch (policy.network) {
                case common_agent_sandbox_network_scope::none: return std::string("none");
                case common_agent_sandbox_network_scope::dns_only: return std::string("dns_only");
                case common_agent_sandbox_network_scope::allowlisted: return std::string("allowlisted");
                case common_agent_sandbox_network_scope::package_registry: return std::string("package_registry");
                case common_agent_sandbox_network_scope::research_web: return std::string("research_web");
            }
            return std::string("none");
        }()},
        {"filesystem", [&policy]() {
            switch (policy.filesystem) {
                case common_agent_sandbox_filesystem_scope::readonly: return std::string("readonly");
                case common_agent_sandbox_filesystem_scope::workspace_write: return std::string("workspace_write");
                case common_agent_sandbox_filesystem_scope::artifact_write: return std::string("artifact_write");
            }
            return std::string("readonly");
        }()},
        {"allow_artifacts", policy.allow_artifacts},
    };
}

json sandbox_defaults_to_json(const common_agent_sandbox_policy & policy) {
    return sandbox_class_to_json(policy);
}

bool read_mcp_provider(
        const json & value,
        agent_host_mcp_provider_config & provider,
        std::string & error) {
    if (!value.is_object()) {
        error = "MCP provider entry must be an object";
        return false;
    }

    read_optional(value, "type", provider.type);
    read_optional(value, "id", provider.id);
    read_optional(value, "enabled", provider.enabled);
    read_optional(value, "transport", provider.transport);
    read_optional(value, "url", provider.url);
    read_optional(value, "token_env", provider.token_env);
    read_optional(value, "connect_timeout_ms", provider.connect_timeout_ms);
    read_optional(value, "request_timeout_ms", provider.request_timeout_ms);
    read_optional(value, "shutdown_timeout_ms", provider.shutdown_timeout_ms);
    read_optional(value, "max_result_bytes", provider.max_result_bytes);
    read_optional(value, "prefix", provider.prefix);
    read_optional(value, "server_name", provider.server_name);
    if (provider.server_name.empty()) {
        provider.server_name = provider.id;
    }

    if (value.contains("command") && !read_command_array(value["command"], provider.command, error)) {
        return false;
    }
    if (value.contains("allowed_tools")) {
        if (!value["allowed_tools"].is_array()) {
            error = "MCP provider allowed_tools must be an array of strings";
            return false;
        }
        provider.allowed_tools.clear();
        for (const auto & tool : value["allowed_tools"]) {
            if (!tool.is_string()) {
                error = "MCP provider allowed_tools entries must be strings";
                return false;
            }
            provider.allowed_tools.push_back(tool.get<std::string>());
        }
    }

    if (provider.type.empty()) {
        provider.type = "mcp";
    }
    if (provider.transport.empty()) {
        provider.transport = "stdio";
    }

    error.clear();
    return true;
}

bool read_inbound_token(
        const json & value,
        agent_host_mcp_inbound_token_config & token,
        std::string & error) {
    if (!value.is_object()) {
        error = "MCP inbound token entry must be an object";
        return false;
    }
    read_optional(value, "id", token.id);
    read_optional(value, "token_env", token.token_env);
    read_optional(value, "audience", token.audience);
    read_optional(value, "namespace", token.namespace_id);
    read_optional(value, "project", token.project_id);
    read_optional(value, "tool_profile", token.tool_profile);
    read_optional(value, "allow_writes", token.allow_writes);
    read_optional(value, "allow_admin", token.allow_admin);
    if (value.contains("allowed_tools")) {
        if (!value["allowed_tools"].is_array()) {
            error = "MCP inbound token allowed_tools must be an array";
            return false;
        }
        token.allowed_tools.clear();
        for (const auto & tool : value["allowed_tools"]) {
            if (!tool.is_string()) {
                error = "MCP inbound token allowed_tools entries must be strings";
                return false;
            }
            token.allowed_tools.push_back(tool.get<std::string>());
        }
    }
    error.clear();
    return true;
}

} // namespace

bool parse_agent_host_config_json(
        const json & parsed,
        agent_host_config & config,
        std::string & error) {
    if (!parsed.is_object()) {
        error = "host config must be a JSON object";
        return false;
    }

    config = {};
    read_optional(parsed, "schema_version", config.schema_version);
    if (config.schema_version <= 0) {
        error = "host config schema_version must be a positive integer";
        return false;
    }
    if (config.schema_version != 1) {
        error = "unsupported host config schema_version: " + std::to_string(config.schema_version);
        return false;
    }

    if (parsed.contains("model") && parsed["model"].is_object()) {
        const auto & model = parsed["model"];
        read_optional(model, "backend", config.model_backend);
        read_optional(model, "path", config.model_path);
        read_optional(model, "embedding_model", config.embedding_model);
    }

    if (parsed.contains("runtime") && parsed["runtime"].is_object()) {
        const auto & runtime = parsed["runtime"];
        read_optional(runtime, "context_size", config.runtime_context_size);
        read_optional(runtime, "n_predict", config.n_predict);
        if (runtime.contains("context_budgets") && runtime["context_budgets"].is_object()) {
            const auto & budgets = runtime["context_budgets"];
            read_optional(budgets, "plan_chars", config.context_budgets.plan_chars);
            read_optional(budgets, "step_chars", config.context_budgets.step_chars);
            read_optional(budgets, "tool_observation_chars", config.context_budgets.tool_observation_chars);
            read_optional(budgets, "input_resources_chars", config.context_budgets.input_resources_chars);
            read_optional(budgets, "deliberate_input_resources_chars", config.context_budgets.deliberate_input_resources_chars);
            read_optional(budgets, "memory_chars", config.context_budgets.memory_chars);
            read_optional(budgets, "memory_per_item_chars", config.context_budgets.memory_per_item_chars);
            read_optional(budgets, "overlay_chars", config.context_budgets.overlay_chars);
            read_optional(budgets, "overlay_per_item_chars", config.context_budgets.overlay_per_item_chars);
            read_optional(budgets, "deliberate_memory_chars", config.context_budgets.deliberate_memory_chars);
            read_optional(budgets, "deliberate_memory_per_item_chars", config.context_budgets.deliberate_memory_per_item_chars);
            read_optional(budgets, "deliberate_overlay_chars", config.context_budgets.deliberate_overlay_chars);
            read_optional(budgets, "deliberate_overlay_per_item_chars", config.context_budgets.deliberate_overlay_per_item_chars);
        }
        read_optional(runtime, "n_gpu_layers", config.n_gpu_layers);
        read_optional(runtime, "default_mode", config.default_mode);
        read_optional(runtime, "thinking_mode", config.thinking_mode);
        read_optional(runtime, "max_reflection_rounds", config.max_reflection_rounds);
        read_optional(runtime, "max_plan_revisions", config.max_plan_revisions);
        read_optional(runtime, "max_research_iterations", config.max_research_iterations);
        read_optional(runtime, "memory_learn", config.memory_learn);
        read_optional(runtime, "agent_plan", config.agent_plan);
        read_optional(runtime, "memory_learn_show_candidate", config.memory_learn_show_candidate);
        read_optional(runtime, "memory_learn_min_confidence", config.memory_learn_min_confidence);
        read_optional(runtime, "memory_learn_min_reuse", config.memory_learn_min_reuse);
        read_optional(runtime, "plan_show_summary", config.plan_show_summary);
        read_optional(runtime, "agent_trace", config.agent_trace);
    }

    if (parsed.contains("stores") && parsed["stores"].is_object()) {
        const auto & stores = parsed["stores"];
        if (stores.contains("memory") && stores["memory"].is_object()) {
            const auto & memory = stores["memory"];
            read_optional(memory, "backend", config.memory_backend);
            read_optional(memory, "path", config.memory_db);
        }
        if (stores.contains("plan") && stores["plan"].is_object()) {
            const auto & plan = stores["plan"];
            read_optional(plan, "backend", config.plan_backend);
            read_optional(plan, "path", config.plan_db);
        }
    }

    if (parsed.contains("resources") && parsed["resources"].is_object()) {
        const auto & resources = parsed["resources"];
        read_optional(resources, "blob_backend", config.resource_blob_backend);
        read_optional(resources, "blob_root", config.resource_blob_root);
        read_optional(resources, "metadata_backend", config.resource_metadata_backend);
        read_optional(resources, "metadata_db", config.resource_metadata_db);
    }

    if (parsed.contains("sandbox")) {
        if (!parsed["sandbox"].is_object()) { error = "sandbox must be an object"; return false; }
        const auto & sandbox = parsed["sandbox"];
        read_optional(sandbox, "backend", config.sandbox.backend);
        if (sandbox.contains("docker")) {
            if (!sandbox["docker"].is_object()) { error = "sandbox.docker must be an object"; return false; }
            const auto & docker = sandbox["docker"];
            read_optional(docker, "executable", config.sandbox.docker_executable);
            read_optional(docker, "default_image", config.sandbox.docker_default_image);
        }
        if (sandbox.contains("kubernetes")) {
            if (!sandbox["kubernetes"].is_object()) { error = "sandbox.kubernetes must be an object"; return false; }
            const auto & kubernetes = sandbox["kubernetes"];
            read_optional(kubernetes, "executable", config.sandbox.kubernetes_executable);
            read_optional(kubernetes, "namespace", config.sandbox.kubernetes_namespace);
            read_optional(kubernetes, "service_account", config.sandbox.kubernetes_service_account);
            read_optional(kubernetes, "runtime_class", config.sandbox.kubernetes_runtime_class);
            read_optional(kubernetes, "storage_class", config.sandbox.kubernetes_storage_class);
            read_optional(kubernetes, "workspace_storage_size", config.sandbox.kubernetes_workspace_storage_size);
            read_optional(kubernetes, "artifact_storage_size", config.sandbox.kubernetes_artifact_storage_size);
            read_optional(kubernetes, "cleanup", config.sandbox.kubernetes_cleanup);
        }
        if (sandbox.contains("workspace")) {
            if (!sandbox["workspace"].is_object()) { error = "sandbox.workspace must be an object"; return false; }
            const auto & workspace = sandbox["workspace"];
            read_optional(workspace, "root", config.sandbox.workspace.workspace_root);
            read_optional(workspace, "artifact_root", config.sandbox.workspace.artifact_root);
            read_optional(workspace, "operation_mode", config.sandbox.workspace.operation_mode);
            read_optional(workspace, "project_mode", config.sandbox.workspace.project_mode);
        }
        if (sandbox.contains("defaults")) {
            if (!sandbox["defaults"].is_object()) { error = "sandbox.defaults must be an object"; return false; }
            const auto & defaults = sandbox["defaults"];
            read_optional(defaults, "image", config.sandbox.defaults.image);
            if (defaults.contains("timeout_ms")) read_optional(defaults, "timeout_ms", config.sandbox.defaults.limits.timeout_ms);
            if (defaults.contains("memory_bytes")) read_optional(defaults, "memory_bytes", config.sandbox.defaults.limits.memory_bytes);
            if (defaults.contains("cpu_count")) read_optional(defaults, "cpu_count", config.sandbox.defaults.limits.cpu_count);
            if (defaults.contains("process_count")) read_optional(defaults, "process_count", config.sandbox.defaults.limits.process_count);
            if (defaults.contains("max_output_bytes")) read_optional(defaults, "max_output_bytes", config.sandbox.defaults.limits.max_output_bytes);
            if (defaults.contains("network") && !read_sandbox_network(defaults["network"], config.sandbox.defaults.network, error)) return false;
            if (defaults.contains("filesystem") && !read_sandbox_filesystem(defaults["filesystem"], config.sandbox.defaults.filesystem, error)) return false;
            read_optional(defaults, "allow_artifacts", config.sandbox.defaults.allow_artifacts);
        }
        if (sandbox.contains("classes")) {
            if (!sandbox["classes"].is_object()) { error = "sandbox.classes must be an object"; return false; }
            config.sandbox.classes.clear();
            for (auto it = sandbox["classes"].begin(); it != sandbox["classes"].end(); ++it) {
                if (!it.value().is_object()) { error = "sandbox.classes entries must be objects"; return false; }
                common_agent_sandbox_policy policy = config.sandbox.defaults;
                policy.execution_class = it.key();
                const auto & value = it.value();
                read_optional(value, "image", policy.image);
                if (value.contains("timeout_ms")) read_optional(value, "timeout_ms", policy.limits.timeout_ms);
                if (value.contains("memory_bytes")) read_optional(value, "memory_bytes", policy.limits.memory_bytes);
                if (value.contains("cpu_count")) read_optional(value, "cpu_count", policy.limits.cpu_count);
                if (value.contains("process_count")) read_optional(value, "process_count", policy.limits.process_count);
                if (value.contains("max_output_bytes")) read_optional(value, "max_output_bytes", policy.limits.max_output_bytes);
                if (value.contains("network") && !read_sandbox_network(value["network"], policy.network, error)) return false;
                if (value.contains("filesystem") && !read_sandbox_filesystem(value["filesystem"], policy.filesystem, error)) return false;
                if (value.contains("allow_artifacts")) read_optional(value, "allow_artifacts", policy.allow_artifacts);
                config.sandbox.classes.emplace(it.key(), std::move(policy));
            }
        }
    }

    if (parsed.contains("tools") && parsed["tools"].is_object()) {
        const auto & tools = parsed["tools"];
        read_optional(tools, "profile", config.tool_profile);
        read_optional(tools, "repository_root", config.repository_root);
        if (tools.contains("capabilities")) {
            if (!tools["capabilities"].is_object()) {
                error = "tools.capabilities must be an object mapping capability ids to tool arrays";
                return false;
            }
            config.tool_capabilities.clear();
            for (auto it = tools["capabilities"].begin(); it != tools["capabilities"].end(); ++it) {
                if (!read_string_array(it.value(), config.tool_capabilities[it.key()],
                        (std::string("tools.capabilities.") + it.key()).c_str(), error)) {
                    return false;
                }
            }
        }
        if (tools.contains("profiles")) {
            if (!tools["profiles"].is_object()) {
                error = "tools.profiles must be an object mapping profile ids to profile definitions";
                return false;
            }
            config.tool_profiles.clear();
            for (auto it = tools["profiles"].begin(); it != tools["profiles"].end(); ++it) {
                if (!it.value().is_object()) {
                    error = "tools.profiles entries must be objects";
                    return false;
                }
                common_tool_profile profile;
                profile.id = it.key();
                if (it.value().contains("include_capabilities") &&
                        !read_string_array(it.value()["include_capabilities"], profile.include_capabilities,
                            (std::string("tools.profiles.") + it.key() + ".include_capabilities").c_str(), error)) {
                    return false;
                }
                if (it.value().contains("exclude_capabilities") &&
                        !read_string_array(it.value()["exclude_capabilities"], profile.exclude_capabilities,
                            (std::string("tools.profiles.") + it.key() + ".exclude_capabilities").c_str(), error)) {
                    return false;
                }
                if (it.value().contains("allow_network")) {
                    if (!it.value()["allow_network"].is_boolean()) {
                        error = "tools.profiles.allow_network must be a boolean";
                        return false;
                    }
                    profile.allow_network = it.value()["allow_network"].get<bool>();
                }
                if (it.value().contains("allow_policy_gated_writes")) {
                    if (!it.value()["allow_policy_gated_writes"].is_boolean()) {
                        error = "tools.profiles.allow_policy_gated_writes must be a boolean";
                        return false;
                    }
                    profile.allow_policy_gated_writes = it.value()["allow_policy_gated_writes"].get<bool>();
                }
                config.tool_profiles.emplace(it.key(), std::move(profile));
            }
        }
        if (tools.contains("providers")) {
            if (!tools["providers"].is_array()) {
                error = "tools.providers must be an array";
                return false;
            }
            config.mcp_providers.clear();
            for (const auto & entry : tools["providers"]) {
                agent_host_mcp_provider_config provider;
                if (!read_mcp_provider(entry, provider, error)) {
                    return false;
                }
                config.mcp_providers.push_back(std::move(provider));
            }
        }
    }

    if (parsed.contains("mcp") && parsed["mcp"].is_object()) {
        const auto & mcp = parsed["mcp"];
        if (mcp.contains("inbound") && mcp["inbound"].is_object()) {
            const auto & inbound = mcp["inbound"];
            read_optional(inbound, "enabled", config.inbound_mcp_enabled);
            read_optional(inbound, "listen", config.inbound_mcp_listen_address);
            read_optional(inbound, "port", config.inbound_mcp_port);
            read_optional(inbound, "path", config.inbound_mcp_path);
            read_optional(inbound, "allowed_origin", config.inbound_mcp_allowed_origin);
            read_optional(inbound, "max_body_bytes", config.inbound_mcp_max_body_bytes);
            read_optional(inbound, "max_result_bytes", config.inbound_mcp_max_result_bytes);
            read_optional(inbound, "agent_tools", config.inbound_mcp_agent_tools_enabled);
            read_optional(inbound, "max_delegation_depth", config.inbound_mcp_max_delegation_depth);
            if (inbound.contains("tokens")) {
                if (!inbound["tokens"].is_array()) {
                    error = "mcp.inbound.tokens must be an array";
                    return false;
                }
                config.inbound_mcp_tokens.clear();
                for (const auto & entry : inbound["tokens"]) {
                    agent_host_mcp_inbound_token_config token;
                    if (!read_inbound_token(entry, token, error)) return false;
                    config.inbound_mcp_tokens.push_back(std::move(token));
                }
            }
            if (inbound.contains("authorization") && inbound["authorization"].is_object()) {
                const auto & authorization = inbound["authorization"];
                read_optional(authorization, "mode", config.inbound_mcp_authorization_mode);
                read_optional(authorization, "issuer", config.inbound_mcp_jwt_issuer);
                read_optional(authorization, "audience", config.inbound_mcp_jwt_audience);
                read_optional(authorization, "jwks_uri", config.inbound_mcp_jwt_jwks_uri);
                read_optional(authorization, "tool_profile", config.inbound_mcp_jwt_tool_profile);
                read_optional(authorization, "allow_writes", config.inbound_mcp_jwt_allow_writes);
                read_optional(authorization, "allow_admin", config.inbound_mcp_jwt_allow_admin);
                if (authorization.contains("allowed_algorithms") &&
                        !read_string_array(authorization["allowed_algorithms"], config.inbound_mcp_jwt_allowed_algorithms, "mcp.inbound.authorization.allowed_algorithms", error)) {
                    return false;
                }
                if (authorization.contains("required_scopes") &&
                        !read_string_array(authorization["required_scopes"], config.inbound_mcp_jwt_required_scopes, "mcp.inbound.authorization.required_scopes", error)) {
                    return false;
                }
                if (authorization.contains("allowed_tools") &&
                        !read_string_array(authorization["allowed_tools"], config.inbound_mcp_jwt_allowed_tools, "mcp.inbound.authorization.allowed_tools", error)) {
                    return false;
                }
            }
        }
    }

    if (parsed.contains("jsonl") && parsed["jsonl"].is_object() &&
            parsed["jsonl"].contains("tcp") && parsed["jsonl"]["tcp"].is_object()) {
        const auto & tcp = parsed["jsonl"]["tcp"];
        read_optional(tcp, "enabled", config.jsonl_tcp_enabled);
        read_optional(tcp, "listen", config.jsonl_tcp_listen_address);
        read_optional(tcp, "port", config.jsonl_tcp_port);
        read_optional(tcp, "max_line_bytes", config.jsonl_tcp_max_line_bytes);
        read_optional(tcp, "idle_timeout_seconds", config.jsonl_tcp_idle_timeout_seconds);
    }
    if (parsed.contains("jsonl") && parsed["jsonl"].is_object() &&
            parsed["jsonl"].contains("unix_socket") && parsed["jsonl"]["unix_socket"].is_object()) {
        const auto & socket = parsed["jsonl"]["unix_socket"];
        read_optional(socket, "enabled", config.jsonl_unix_socket_enabled);
        read_optional(socket, "path", config.jsonl_unix_socket_path);
        read_optional(socket, "mode", config.jsonl_unix_socket_mode);
    }

    if (parsed.contains("limits") && parsed["limits"].is_object()) {
        const auto & limits = parsed["limits"];
        read_optional(limits, "queue_capacity", config.queue_capacity);
        read_optional(limits, "worker_count", config.worker_count);
        read_optional(limits, "inference_max_active", config.inference_max_active);
        read_optional(limits, "max_turn_seconds", config.max_turn_seconds);
        read_optional(limits, "turn_timeout_ms", config.turn_timeout_ms);
        read_optional(limits, "inference_step_timeout_ms", config.inference_step_timeout_ms);
        read_optional(limits, "tool_timeout_ms", config.tool_timeout_ms);
        read_optional(limits, "mcp_connect_timeout_ms", config.mcp_connect_timeout_ms);
        read_optional(limits, "mcp_request_timeout_ms", config.mcp_request_timeout_ms);
        read_optional(limits, "mcp_shutdown_timeout_ms", config.mcp_shutdown_timeout_ms);
        read_optional(limits, "max_tool_rounds", config.max_tool_rounds);
    }

    if (config.turn_timeout_ms == 0 && config.max_turn_seconds > 0) {
        config.turn_timeout_ms = config.max_turn_seconds * 1000;
    }

    if (!validate_agent_host_config(config, error)) {
        return false;
    }

    error.clear();
    return true;
}

bool load_agent_host_config(
        const std::string & path,
        agent_host_config & config,
        std::string & error) {
    std::ifstream input(path);
    if (!input.is_open()) {
        error = "failed to open host config: " + path;
        return false;
    }

    json parsed = json::parse(input, nullptr, false);
    if (parsed.is_discarded()) {
        error = "host config must be a JSON object";
        return false;
    }
    return parse_agent_host_config_json(parsed, config, error);
}

nlohmann::ordered_json agent_host_config_to_json(
        const agent_host_config & config) {
    json providers = json::array();
    for (const auto & provider : config.mcp_providers) {
        providers.push_back({
            {"type", provider.type},
            {"id", provider.id},
            {"enabled", provider.enabled},
            {"transport", provider.transport},
            {"url", provider.url},
            {"token_env", provider.token_env},
            {"allowed_tools", provider.allowed_tools},
            {"connect_timeout_ms", provider.connect_timeout_ms},
            {"request_timeout_ms", provider.request_timeout_ms},
            {"shutdown_timeout_ms", provider.shutdown_timeout_ms},
            {"max_result_bytes", provider.max_result_bytes},
            {"prefix", provider.prefix},
            {"server_name", provider.server_name},
            {"command", provider.command},
        });
    }
    json capabilities = json::object();
    for (const auto & entry : config.tool_capabilities) {
        capabilities[entry.first] = entry.second;
    }

    json profiles = json::object();
    for (const auto & entry : config.tool_profiles) {
        json profile = {
            {"include_capabilities", entry.second.include_capabilities},
            {"exclude_capabilities", entry.second.exclude_capabilities},
        };
        if (entry.second.allow_network.has_value()) {
            profile["allow_network"] = *entry.second.allow_network;
        }
        if (entry.second.allow_policy_gated_writes.has_value()) {
            profile["allow_policy_gated_writes"] = *entry.second.allow_policy_gated_writes;
        }
        profiles[entry.first] = std::move(profile);
    }
    json sandbox_classes = json::object();
    for (const auto & entry : config.sandbox.classes) sandbox_classes[entry.first] = sandbox_class_to_json(entry.second);
    return {
        {"schema_version", config.schema_version},
        {"model", {
            {"backend", config.model_backend},
            {"path", config.model_path},
            {"embedding_model", config.embedding_model},
        }},
        {"runtime", {
            {"context_size", config.runtime_context_size},
            {"n_predict", config.n_predict},
            {"context_budgets", {
                {"plan_chars", config.context_budgets.plan_chars},
                {"step_chars", config.context_budgets.step_chars},
                {"tool_observation_chars", config.context_budgets.tool_observation_chars},
                {"input_resources_chars", config.context_budgets.input_resources_chars},
                {"deliberate_input_resources_chars", config.context_budgets.deliberate_input_resources_chars},
                {"memory_chars", config.context_budgets.memory_chars},
                {"memory_per_item_chars", config.context_budgets.memory_per_item_chars},
                {"overlay_chars", config.context_budgets.overlay_chars},
                {"overlay_per_item_chars", config.context_budgets.overlay_per_item_chars},
                {"deliberate_memory_chars", config.context_budgets.deliberate_memory_chars},
                {"deliberate_memory_per_item_chars", config.context_budgets.deliberate_memory_per_item_chars},
                {"deliberate_overlay_chars", config.context_budgets.deliberate_overlay_chars},
                {"deliberate_overlay_per_item_chars", config.context_budgets.deliberate_overlay_per_item_chars},
            }},
            {"n_gpu_layers", config.n_gpu_layers},
            {"default_mode", config.default_mode},
            {"thinking_mode", config.thinking_mode},
            {"max_reflection_rounds", config.max_reflection_rounds},
            {"max_plan_revisions", config.max_plan_revisions},
            {"max_research_iterations", config.max_research_iterations},
            {"memory_learn", config.memory_learn},
            {"agent_plan", config.agent_plan},
            {"memory_learn_show_candidate", config.memory_learn_show_candidate},
            {"memory_learn_min_confidence", config.memory_learn_min_confidence},
            {"memory_learn_min_reuse", config.memory_learn_min_reuse},
            {"plan_show_summary", config.plan_show_summary},
            {"agent_trace", config.agent_trace},
        }},
        {"stores", {
            {"memory", {
                {"backend", config.memory_backend},
                {"path", config.memory_db},
            }},
            {"plan", {
                {"backend", config.plan_backend},
                {"path", config.plan_db},
            }},
        }},
        {"resources", {
            {"blob_backend", config.resource_blob_backend},
            {"blob_root", config.resource_blob_root},
            {"metadata_backend", config.resource_metadata_backend},
            {"metadata_db", config.resource_metadata_db},
        }},
        {"tools", {
            {"profile", config.tool_profile},
            {"repository_root", config.repository_root},
            {"capabilities", std::move(capabilities)},
            {"profiles", std::move(profiles)},
            {"providers", std::move(providers)},
        }},
        {"sandbox", {
            {"backend", config.sandbox.backend},
            {"docker", {
                {"executable", config.sandbox.docker_executable},
                {"default_image", config.sandbox.docker_default_image},
            }},
            {"kubernetes", {
                {"executable", config.sandbox.kubernetes_executable},
                {"namespace", config.sandbox.kubernetes_namespace},
                {"service_account", config.sandbox.kubernetes_service_account},
                {"runtime_class", config.sandbox.kubernetes_runtime_class},
                {"storage_class", config.sandbox.kubernetes_storage_class},
                {"workspace_storage_size", config.sandbox.kubernetes_workspace_storage_size},
                {"artifact_storage_size", config.sandbox.kubernetes_artifact_storage_size},
                {"cleanup", config.sandbox.kubernetes_cleanup},
            }},
            {"workspace", {
                {"root", config.sandbox.workspace.workspace_root},
                {"artifact_root", config.sandbox.workspace.artifact_root},
                {"operation_mode", config.sandbox.workspace.operation_mode},
                {"project_mode", config.sandbox.workspace.project_mode},
            }},
            {"defaults", sandbox_defaults_to_json(config.sandbox.defaults)},
            {"classes", std::move(sandbox_classes)},
        }},
        {"mcp", {
            {"inbound", {
                {"enabled", config.inbound_mcp_enabled},
                {"listen", config.inbound_mcp_listen_address},
                {"port", config.inbound_mcp_port},
                {"path", config.inbound_mcp_path},
                {"allowed_origin", config.inbound_mcp_allowed_origin},
                {"max_body_bytes", config.inbound_mcp_max_body_bytes},
                {"max_result_bytes", config.inbound_mcp_max_result_bytes},
                {"agent_tools", config.inbound_mcp_agent_tools_enabled},
                {"max_delegation_depth", config.inbound_mcp_max_delegation_depth},
                {"tokens", [&config]() {
                    json tokens = json::array();
                    for (const auto & token : config.inbound_mcp_tokens) {
                        tokens.push_back({
                            {"id", token.id},
                            {"token_env", token.token_env},
                            {"audience", token.audience},
                            {"namespace", token.namespace_id},
                            {"project", token.project_id},
                            {"tool_profile", token.tool_profile},
                            {"allowed_tools", token.allowed_tools},
                            {"allow_writes", token.allow_writes},
                            {"allow_admin", token.allow_admin},
                        });
                    }
                    return tokens;
                }()},
                {"authorization", {
                    {"mode", config.inbound_mcp_authorization_mode},
                    {"issuer", config.inbound_mcp_jwt_issuer},
                    {"audience", config.inbound_mcp_jwt_audience},
                    {"jwks_uri", config.inbound_mcp_jwt_jwks_uri},
                    {"allowed_algorithms", config.inbound_mcp_jwt_allowed_algorithms},
                    {"required_scopes", config.inbound_mcp_jwt_required_scopes},
                    {"tool_profile", config.inbound_mcp_jwt_tool_profile},
                    {"allowed_tools", config.inbound_mcp_jwt_allowed_tools},
                    {"allow_writes", config.inbound_mcp_jwt_allow_writes},
                    {"allow_admin", config.inbound_mcp_jwt_allow_admin},
                }},
            }},
        }},
        {"jsonl", {
            {"tcp", {
                {"enabled", config.jsonl_tcp_enabled},
                {"listen", config.jsonl_tcp_listen_address},
                {"port", config.jsonl_tcp_port},
                {"max_line_bytes", config.jsonl_tcp_max_line_bytes},
                {"idle_timeout_seconds", config.jsonl_tcp_idle_timeout_seconds},
            }},
            {"unix_socket", {
                {"enabled", config.jsonl_unix_socket_enabled},
                {"path", config.jsonl_unix_socket_path},
                {"mode", config.jsonl_unix_socket_mode},
            }},
        }},
        {"limits", {
            {"queue_capacity", config.queue_capacity},
            {"worker_count", config.worker_count},
            {"inference_max_active", config.inference_max_active},
            {"max_turn_seconds", config.max_turn_seconds},
            {"turn_timeout_ms", config.turn_timeout_ms},
            {"inference_step_timeout_ms", config.inference_step_timeout_ms},
            {"tool_timeout_ms", config.tool_timeout_ms},
            {"mcp_connect_timeout_ms", config.mcp_connect_timeout_ms},
            {"mcp_request_timeout_ms", config.mcp_request_timeout_ms},
            {"mcp_shutdown_timeout_ms", config.mcp_shutdown_timeout_ms},
            {"max_tool_rounds", config.max_tool_rounds},
        }},
    };
}

bool validate_agent_host_config(
        const agent_host_config & config,
        std::string & error) {
    if (config.jsonl_tcp_enabled && config.jsonl_unix_socket_enabled) {
        error = "jsonl.tcp and jsonl.unix_socket cannot both be enabled for one JSONL host";
        return false;
    }
    if (config.schema_version != 1) {
        error = "unsupported host config schema_version: " + std::to_string(config.schema_version);
        return false;
    }
    if (config.queue_capacity == 0) {
        error = "limits.queue_capacity must be greater than zero";
        return false;
    }
    if (config.worker_count == 0) {
        error = "limits.worker_count must be greater than zero";
        return false;
    }
    if (config.inference_max_active == 0) {
        error = "limits.inference_max_active must be greater than zero";
        return false;
    }
    const auto & budgets = config.context_budgets;
    if (budgets.plan_chars == 0 || budgets.step_chars == 0 ||
            budgets.tool_observation_chars == 0 || budgets.input_resources_chars == 0 ||
            budgets.deliberate_input_resources_chars == 0 || budgets.memory_chars == 0 ||
            budgets.memory_per_item_chars == 0 || budgets.overlay_chars == 0 ||
            budgets.overlay_per_item_chars == 0 || budgets.deliberate_memory_chars == 0 ||
            budgets.deliberate_memory_per_item_chars == 0 || budgets.deliberate_overlay_chars == 0 ||
            budgets.deliberate_overlay_per_item_chars == 0) {
        error = "runtime.context_budgets values must be greater than zero";
        return false;
    }
    common_agent_thinking_request thinking_request;
    if (!parse_common_agent_thinking_request(config.thinking_mode, thinking_request)) {
        error = "runtime.thinking_mode must be auto, reflective, deliberate, or research";
        return false;
    }
    if (config.max_reflection_rounds < 0 || config.max_plan_revisions < 0) {
        error = "runtime deliberation limits must not be negative";
        return false;
    }
    if (config.inbound_mcp_enabled) {
        if (config.inbound_mcp_listen_address.empty() || config.inbound_mcp_path.empty()) {
            error = "mcp.inbound listen and path must not be empty";
            return false;
        }
        if (config.inbound_mcp_tokens.empty()) {
            if (config.inbound_mcp_authorization_mode != "jwt") {
                error = "mcp.inbound.tokens must not be empty when opaque inbound MCP is enabled";
                return false;
            }
        }
        if (config.inbound_mcp_authorization_mode != "opaque" &&
                config.inbound_mcp_authorization_mode != "jwt") {
            error = "mcp.inbound.authorization.mode must be opaque or jwt";
            return false;
        }
        if (config.inbound_mcp_authorization_mode == "jwt" &&
                (config.inbound_mcp_jwt_issuer.empty() ||
                 config.inbound_mcp_jwt_audience.empty() ||
                 config.inbound_mcp_jwt_jwks_uri.empty() ||
                 config.inbound_mcp_jwt_tool_profile.empty())) {
            error = "JWT inbound MCP authorization requires issuer, audience, jwks_uri and tool_profile";
            return false;
        }
    }

    if (config.jsonl_tcp_enabled) {
        if (config.jsonl_tcp_listen_address.empty() ||
                config.jsonl_tcp_port <= 0 || config.jsonl_tcp_port > 65535) {
            error = "jsonl.tcp requires a valid listen address and port";
            return false;
        }
        if (config.jsonl_tcp_max_line_bytes == 0) {
            error = "jsonl.tcp.max_line_bytes must be at least 1";
            return false;
        }
        if (config.inbound_mcp_tokens.empty() && config.inbound_mcp_authorization_mode != "jwt") {
            error = "jsonl.tcp reuses mcp.inbound authentication and requires opaque token profiles or jwt mode";
            return false;
        }
    }
    if (config.jsonl_unix_socket_enabled) {
        if (config.jsonl_unix_socket_path.empty()) {
            error = "jsonl.unix_socket requires a path";
            return false;
        }
        if (config.jsonl_unix_socket_mode < 0 || config.jsonl_unix_socket_mode > 0777) {
            error = "jsonl.unix_socket.mode must be an octal permission value between 0000 and 0777";
            return false;
        }
        if (config.inbound_mcp_tokens.empty() && config.inbound_mcp_authorization_mode != "jwt") {
            error = "jsonl.unix_socket reuses mcp.inbound authentication and requires opaque token profiles or jwt mode";
            return false;
        }
    }
    std::unordered_set<std::string> inbound_token_ids;
    for (const auto & token : config.inbound_mcp_tokens) {
        if (token.id.empty() || token.token_env.empty() || token.audience.empty() ||
                token.namespace_id.empty() || token.project_id.empty() || token.tool_profile.empty()) {
            error = "MCP inbound token requires id, token_env, audience, namespace, project and tool_profile";
            return false;
        }
        if (!inbound_token_ids.insert(token.id).second) {
            error = "MCP inbound token ids must be unique: " + token.id;
            return false;
        }
    }
    std::unordered_set<std::string> provider_ids;
    for (const auto & capability : config.tool_capabilities) {
        if (capability.first.empty()) {
            error = "tools.capabilities ids must not be empty";
            return false;
        }
        std::unordered_set<std::string> tool_names;
        for (const auto & tool : capability.second) {
            if (tool.empty()) {
                error = "tools.capabilities entries must not contain empty tool names";
                return false;
            }
            if (!tool_names.insert(tool).second) {
                error = "tools.capabilities entries must not contain duplicate tools: " + tool;
                return false;
            }
        }
    }
    if (config.sandbox.backend != "none" && config.sandbox.backend != "docker" && config.sandbox.backend != "kubernetes") {
        error = "sandbox.backend must be none, docker or kubernetes";
        return false;
    }
    if (config.sandbox.backend == "kubernetes" && config.sandbox.kubernetes_namespace.empty()) {
        error = "sandbox.kubernetes.namespace must not be empty";
        return false;
    }
    if (config.sandbox.backend == "kubernetes" &&
            (config.sandbox.kubernetes_workspace_storage_size.empty() ||
             config.sandbox.kubernetes_artifact_storage_size.empty())) {
        error = "sandbox.kubernetes storage sizes must not be empty";
        return false;
    }
    if (!config.sandbox.classes.empty() && config.sandbox.workspace.workspace_root.empty()) {
        error = "sandbox.workspace.root is required when sandbox classes are configured";
        return false;
    }
    if (config.sandbox.workspace.operation_mode != "ephemeral" &&
            config.sandbox.workspace.operation_mode != "persistent") {
        error = "sandbox.workspace.operation_mode must be ephemeral or persistent";
        return false;
    }
    if (config.sandbox.workspace.project_mode != "persistent" &&
            config.sandbox.workspace.project_mode != "ephemeral") {
        error = "sandbox.workspace.project_mode must be persistent or ephemeral";
        return false;
    }
    for (const auto & entry : config.sandbox.classes) {
        if (entry.first.empty() || entry.second.execution_class != entry.first ||
                entry.second.limits.timeout_ms == 0 || entry.second.limits.cpu_count == 0 ||
                entry.second.limits.max_output_bytes == 0) {
            error = "sandbox class has invalid identity or limits: " + entry.first;
            return false;
        }
    }
    for (const auto & profile : config.tool_profiles) {
        if (profile.first.empty()) {
            error = "tools.profiles ids must not be empty";
            return false;
        }
        std::unordered_set<std::string> included;
        for (const auto & capability : profile.second.include_capabilities) {
            if (!config.tool_capabilities.count(capability)) {
                error = "tool profile references an unknown included capability: " + capability;
                return false;
            }
            if (!included.insert(capability).second) {
                error = "tool profile includes a duplicate capability: " + capability;
                return false;
            }
        }
        std::unordered_set<std::string> excluded;
        for (const auto & capability : profile.second.exclude_capabilities) {
            if (!config.tool_capabilities.count(capability)) {
                error = "tool profile references an unknown excluded capability: " + capability;
                return false;
            }
            if (!excluded.insert(capability).second) {
                error = "tool profile excludes a duplicate capability: " + capability;
                return false;
            }
            if (included.count(capability)) {
                error = "tool profile cannot both include and exclude capability: " + capability;
                return false;
            }
        }
    }
    for (const auto & provider : config.mcp_providers) {
        if (provider.id.empty()) {
            error = "MCP provider is missing a stable id";
            return false;
        }
        if (!provider_ids.insert(provider.id).second) {
            error = "MCP provider ids must be unique: " + provider.id;
            return false;
        }
    }
    for (const auto & provider : config.mcp_providers) {
        if (!provider.enabled) {
            continue;
        }
        if (provider.transport.empty()) {
            error = "enabled MCP provider is missing a transport";
            return false;
        }
        if (provider.transport == "stdio" && provider.command.empty()) {
            error = "enabled stdio MCP provider is missing a command";
            return false;
        }
        if (provider.transport == "http" || provider.transport == "https" || provider.transport == "streamable_http") {
            if (provider.url.empty()) {
                error = "enabled HTTP MCP provider is missing a url";
                return false;
            }
            if (provider.command.size() > 0) {
                error = "HTTP MCP provider must not define a command";
                return false;
            }
        } else if (provider.transport != "stdio") {
            error = "unsupported MCP provider transport: " + provider.transport;
            return false;
        }
    }
    error.clear();
    return true;
}

void apply_agent_host_config_to_daemon_options(
        const agent_host_config & config,
        daemon_options & options) {
    options.model = config.model_path;
    options.embedding_model = config.embedding_model;
    options.default_mode = config.default_mode;
    options.thinking_mode = config.thinking_mode;
    options.max_reflection_rounds = config.max_reflection_rounds;
    options.max_plan_revisions = config.max_plan_revisions;
    options.max_research_iterations = config.max_research_iterations;
    options.n_predict = config.n_predict;
    options.context_budgets = config.context_budgets;
    options.n_gpu_layers = config.n_gpu_layers;
    options.memory_learn = config.memory_learn;
    options.agent_plan = config.agent_plan;
    options.backend = config.memory_backend;
    options.memory_db = config.memory_db;
    options.plan_backend = config.plan_backend;
    options.plan_db = config.plan_db;
    options.tool_profile = config.tool_profile;
    options.tool_capabilities = config.tool_capabilities;
    options.tool_profiles = config.tool_profiles;
    options.sandbox = config.sandbox;
    options.repository_root = config.repository_root;
    options.resource_blob_backend = config.resource_blob_backend;
    options.resource_blob_root = config.resource_blob_root;
    options.resource_metadata_backend = config.resource_metadata_backend;
    options.resource_metadata_db = config.resource_metadata_db;
    options.memory_learn_show_candidate = config.memory_learn_show_candidate;
    options.memory_learn_min_confidence = config.memory_learn_min_confidence;
    options.memory_learn_min_reuse = config.memory_learn_min_reuse;
    options.plan_show_summary = config.plan_show_summary;
    options.agent_trace = config.agent_trace;
    options.max_tool_rounds = config.max_tool_rounds;
    options.queue_capacity = config.queue_capacity;
    options.worker_count = config.worker_count;
    options.inference_max_active = config.inference_max_active;
    options.max_turn_seconds = config.max_turn_seconds;
    options.turn_timeout_ms = config.turn_timeout_ms;
    options.inference_step_timeout_ms = config.inference_step_timeout_ms;
    options.tool_timeout_ms = config.tool_timeout_ms;
    options.mcp_connect_timeout_ms = config.mcp_connect_timeout_ms;
    options.mcp_request_timeout_ms = config.mcp_request_timeout_ms;
    options.mcp_shutdown_timeout_ms = config.mcp_shutdown_timeout_ms;

    agent_host_mcp_provider_config provider;
    std::string ignored_error;
    if (select_agent_host_stdio_mcp_provider(config, provider, ignored_error)) {
        options.mcp_tool_command = provider.command.front();
        options.mcp_tool_args.assign(provider.command.begin() + 1, provider.command.end());
        options.mcp_tool_server_name = provider.server_name.empty() ? provider.id : provider.server_name;
        options.mcp_tool_prefix = provider.prefix;
    }
    options.mcp_providers = config.mcp_providers;
    options.http_enabled = config.inbound_mcp_enabled;
    options.http_listen_address = config.inbound_mcp_listen_address;
    options.http_port = config.inbound_mcp_port;
    options.http_path = config.inbound_mcp_path;
    options.http_allowed_origin = config.inbound_mcp_allowed_origin;
    options.http_max_body_bytes = config.inbound_mcp_max_body_bytes;
    options.http_max_result_bytes = config.inbound_mcp_max_result_bytes;
    options.http_agent_tools_enabled = config.inbound_mcp_agent_tools_enabled;
    options.http_max_delegation_depth = config.inbound_mcp_max_delegation_depth;
    options.http_token_profiles = config.inbound_mcp_tokens;
    options.http_authorization_mode = config.inbound_mcp_authorization_mode;
    options.http_jwt_issuer = config.inbound_mcp_jwt_issuer;
    options.http_jwt_audience = config.inbound_mcp_jwt_audience;
    options.http_jwt_jwks_uri = config.inbound_mcp_jwt_jwks_uri;
    options.http_jwt_allowed_algorithms = config.inbound_mcp_jwt_allowed_algorithms;
    options.http_jwt_required_scopes = config.inbound_mcp_jwt_required_scopes;
    options.http_jwt_tool_profile = config.inbound_mcp_jwt_tool_profile;
    options.http_jwt_allowed_tools = config.inbound_mcp_jwt_allowed_tools;
    options.http_jwt_allow_writes = config.inbound_mcp_jwt_allow_writes;
    options.http_jwt_allow_admin = config.inbound_mcp_jwt_allow_admin;
    options.tcp_enabled = config.jsonl_tcp_enabled;
    options.tcp_listen_address = config.jsonl_tcp_listen_address;
    options.tcp_port = config.jsonl_tcp_port;
    options.tcp_max_line_bytes = config.jsonl_tcp_max_line_bytes;
    options.tcp_idle_timeout_seconds = config.jsonl_tcp_idle_timeout_seconds;
    options.unix_socket_enabled = config.jsonl_unix_socket_enabled;
    options.unix_socket_path = config.jsonl_unix_socket_path;
    options.unix_socket_mode = config.jsonl_unix_socket_mode;
}

void apply_agent_host_config_to_args(
        const agent_host_config & config,
        args & options) {
    options.model = config.model_path;
    options.embedding_model = config.embedding_model;
    options.backend = config.memory_backend;
    options.memory_db = config.memory_db;
    options.n_predict = config.n_predict;
    options.context_budgets = config.context_budgets;
    options.n_gpu_layers = config.n_gpu_layers;
    options.tool_profile = config.tool_profile;
    options.tool_capabilities = config.tool_capabilities;
    options.tool_profiles = config.tool_profiles;
    options.sandbox = config.sandbox;
    options.repository_root = config.repository_root;
    options.resource_blob_backend = config.resource_blob_backend;
    options.resource_blob_root = config.resource_blob_root;
    options.resource_metadata_backend = config.resource_metadata_backend;
    options.resource_metadata_db = config.resource_metadata_db;
    options.plan_backend = config.plan_backend;
    options.plan_db = config.plan_db;
    options.thinking_mode = config.thinking_mode;
    options.max_reflection_rounds = config.max_reflection_rounds;
    options.max_plan_revisions = config.max_plan_revisions;
    options.max_research_iterations = config.max_research_iterations;
    options.memory_learn = config.memory_learn;
    options.agent_plan = config.agent_plan;
    options.max_tool_rounds = config.max_tool_rounds;
    options.turn_timeout_ms = config.turn_timeout_ms;
    options.inference_step_timeout_ms = config.inference_step_timeout_ms;
    options.tool_timeout_ms = config.tool_timeout_ms;
    options.mcp_connect_timeout_ms = config.mcp_connect_timeout_ms;
    options.mcp_request_timeout_ms = config.mcp_request_timeout_ms;
    options.mcp_shutdown_timeout_ms = config.mcp_shutdown_timeout_ms;
    options.memory_learn_show_candidate = config.memory_learn_show_candidate;
    options.memory_learn_min_confidence = config.memory_learn_min_confidence;
    options.memory_learn_min_reuse = config.memory_learn_min_reuse;
    options.plan_show_summary = config.plan_show_summary;
    options.agent_trace = config.agent_trace;
}

bool select_agent_host_stdio_mcp_provider(
        const agent_host_config & config,
        agent_host_mcp_provider_config & provider,
        std::string & error) {
    bool found = false;
    for (const auto & candidate : config.mcp_providers) {
        if (!candidate.enabled || candidate.type != "mcp") {
            continue;
        }
        if (candidate.transport != "stdio") {
            error = "only stdio MCP providers are currently supported by this host config path";
            return false;
        }
        if (found) {
            error = "multiple enabled stdio MCP providers are not yet supported by this host config path";
            return false;
        }
        provider = candidate;
        found = true;
    }

    if (!found) {
        error.clear();
        return false;
    }
    if (provider.command.empty()) {
        error = "enabled stdio MCP provider is missing a command";
        return false;
    }
    error.clear();
    return true;
}
