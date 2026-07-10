#include "agent-host-config.h"
#include "agent-daemon-adapter.h"

#include <fstream>
#include <nlohmann/json.hpp>

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
    read_optional(value, "prefix", provider.prefix);
    read_optional(value, "server_name", provider.server_name);
    if (provider.server_name.empty()) {
        provider.server_name = provider.id;
    }

    if (value.contains("command") && !read_command_array(value["command"], provider.command, error)) {
        return false;
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

    if (parsed.contains("limits") && parsed["limits"].is_object()) {
        const auto & limits = parsed["limits"];
        read_optional(limits, "queue_capacity", config.queue_capacity);
        read_optional(limits, "max_turn_seconds", config.max_turn_seconds);
        read_optional(limits, "max_tool_rounds", config.max_tool_rounds);
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
        {"limits", {
            {"queue_capacity", config.queue_capacity},
            {"max_turn_seconds", config.max_turn_seconds},
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
    options.max_turn_seconds = config.max_turn_seconds;

    agent_host_mcp_provider_config provider;
    std::string ignored_error;
    if (select_agent_host_stdio_mcp_provider(config, provider, ignored_error)) {
        options.mcp_tool_command = provider.command.front();
        options.mcp_tool_args.assign(provider.command.begin() + 1, provider.command.end());
        options.mcp_tool_server_name = provider.server_name.empty() ? provider.id : provider.server_name;
        options.mcp_tool_prefix = provider.prefix;
    }
    options.mcp_providers = config.mcp_providers;
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
