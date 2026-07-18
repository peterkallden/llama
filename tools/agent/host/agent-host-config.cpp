#include "agent-host-config.h"
#include "../daemon/agent-daemon-adapter.h"

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
        read_optional(runtime, "n_gpu_layers", config.n_gpu_layers);
        read_optional(runtime, "default_mode", config.default_mode);
        read_optional(runtime, "planning_mode", config.planning_mode);
        read_optional(runtime, "reflection_mode", config.reflection_mode);
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

    if (parsed.contains("tools") && parsed["tools"].is_object()) {
        const auto & tools = parsed["tools"];
        read_optional(tools, "profile", config.tool_profile);
        read_optional(tools, "repository_root", config.repository_root);
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

    if (parsed.contains("limits") && parsed["limits"].is_object()) {
        const auto & limits = parsed["limits"];
        read_optional(limits, "queue_capacity", config.queue_capacity);
        read_optional(limits, "worker_count", config.worker_count);
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
            {"n_gpu_layers", config.n_gpu_layers},
            {"default_mode", config.default_mode},
            {"planning_mode", config.planning_mode},
            {"reflection_mode", config.reflection_mode},
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
            {"providers", std::move(providers)},
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
        }},
        {"limits", {
            {"queue_capacity", config.queue_capacity},
            {"worker_count", config.worker_count},
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
    options.n_predict = config.n_predict;
    options.n_gpu_layers = config.n_gpu_layers;
    options.planning_mode = config.planning_mode;
    options.reflection_mode = config.reflection_mode;
    options.memory_learn = config.memory_learn;
    options.agent_plan = config.agent_plan;
    options.backend = config.memory_backend;
    options.memory_db = config.memory_db;
    options.plan_backend = config.plan_backend;
    options.plan_db = config.plan_db;
    options.tool_profile = config.tool_profile;
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
    options.tcp_enabled = config.jsonl_tcp_enabled;
    options.tcp_listen_address = config.jsonl_tcp_listen_address;
    options.tcp_port = config.jsonl_tcp_port;
    options.tcp_max_line_bytes = config.jsonl_tcp_max_line_bytes;
    options.tcp_idle_timeout_seconds = config.jsonl_tcp_idle_timeout_seconds;
}

void apply_agent_host_config_to_args(
        const agent_host_config & config,
        args & options) {
    options.model = config.model_path;
    options.embedding_model = config.embedding_model;
    options.backend = config.memory_backend;
    options.memory_db = config.memory_db;
    options.n_predict = config.n_predict;
    options.n_gpu_layers = config.n_gpu_layers;
    options.tool_profile = config.tool_profile;
    options.repository_root = config.repository_root;
    options.resource_blob_backend = config.resource_blob_backend;
    options.resource_blob_root = config.resource_blob_root;
    options.resource_metadata_backend = config.resource_metadata_backend;
    options.resource_metadata_db = config.resource_metadata_db;
    options.plan_backend = config.plan_backend;
    options.plan_db = config.plan_db;
    options.planning_mode = config.planning_mode;
    options.reflection_mode = config.reflection_mode;
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
