#include "agent-openapi-catalog.h"

#include <algorithm>
#include <cctype>
#include <set>

namespace {

bool is_http_method(const std::string & value) {
    static const std::set<std::string> methods = {
        "get", "head", "post", "put", "patch", "delete", "options", "trace"
    };
    return methods.find(value) != methods.end();
}

agent_openapi_access default_access(const std::string & method) {
    if (method == "get" || method == "head") return agent_openapi_access::read;
    if (method == "delete") return agent_openapi_access::destructive;
    return agent_openapi_access::write;
}

std::string make_operation_id(const std::string & method, const std::string & path) {
    std::string result = method;
    for (const char c : path) {
        if (std::isalnum(static_cast<unsigned char>(c))) result += c;
        else result += '_';
    }
    return result;
}

bool policy_allows(
        const agent_openapi_operation & operation,
        const agent_host_openapi_provider_config & config,
        const agent_host_openapi_operation_policy * override_policy) {
    if (override_policy != nullptr && !override_policy->enabled) return false;
    if (config.exposure == "exclude" && override_policy != nullptr) return false;
    if (config.exposure == "include" && override_policy == nullptr) return false;

    const std::string access = override_policy != nullptr && !override_policy->access.empty()
        ? override_policy->access
        : agent_openapi_access_name(operation.access);
    if (config.access == "read_only") return access == "read";
    if (config.access == "read_write") return access == "read" || access == "write";
    return access == "read" || access == "write" || access == "destructive";
}

std::string operation_input_schema(const nlohmann::json & operation) {
    nlohmann::json schema = {
        {"type", "object"},
        {"properties", nlohmann::json::object()},
        {"required", nlohmann::json::array()},
    };
    if (operation.contains("parameters") && operation["parameters"].is_array()) {
        for (const auto & parameter : operation["parameters"]) {
            if (!parameter.is_object() || !parameter.value("name", "").size()) continue;
            const std::string name = parameter.value("name", "");
            schema["properties"][name] = parameter.value("schema", nlohmann::json({{"type", "string"}}));
            if (parameter.value("required", false)) schema["required"].push_back(name);
        }
    }
    if (operation.contains("requestBody") && operation["requestBody"].is_object()) {
        const auto & content = operation["requestBody"].value("content", nlohmann::json::object());
        if (content.is_object() && content.contains("application/json")) {
            const auto & media = content["application/json"];
            if (media.is_object() && media.contains("schema")) schema["properties"]["body"] = media["schema"];
            if (operation["requestBody"].value("required", false)) schema["required"].push_back("body");
        }
    }
    if (schema["required"].empty()) schema.erase("required");
    return schema.dump();
}

}

std::string agent_openapi_access_name(agent_openapi_access access) {
    switch (access) {
        case agent_openapi_access::read: return "read";
        case agent_openapi_access::write: return "write";
        case agent_openapi_access::destructive: return "destructive";
    }
    return "write";
}

bool build_agent_openapi_catalog(
        const nlohmann::json & document,
        const agent_host_openapi_provider_config & config,
        agent_openapi_catalog & catalog,
        std::string & error) {
    if (!document.is_object() || document.value("openapi", "").empty()) {
        error = "OpenAPI provider requires an OpenAPI document";
        return false;
    }
    if (!document.contains("paths") || !document["paths"].is_object()) {
        error = "OpenAPI document is missing paths";
        return false;
    }
    catalog = {};
    catalog.provider_id = config.id;
    catalog.base_url = config.base_url;
    catalog.prefix = config.prefix.empty() ? config.id : config.prefix;

    for (auto path_it = document["paths"].begin(); path_it != document["paths"].end(); ++path_it) {
        if (!path_it.value().is_object()) continue;
        for (auto operation_it = path_it.value().begin(); operation_it != path_it.value().end(); ++operation_it) {
            const std::string method = operation_it.key();
            if (!is_http_method(method) || !operation_it.value().is_object()) continue;
            const auto & value = operation_it.value();
            agent_openapi_operation operation;
            operation.method = method;
            operation.path = path_it.key();
            operation.operation_id = value.value("operationId", make_operation_id(method, operation.path));
            operation.summary = value.value("summary", "");
            operation.description = value.value("description", operation.summary);
            operation.access = default_access(method);
            operation.read_only = operation.access == agent_openapi_access::read;
            operation.requires_confirmation = !operation.read_only;
            operation.input_schema_json = operation_input_schema(value);

            const auto policy_it = config.operations.find(operation.operation_id);
            const auto * override_policy = policy_it == config.operations.end() ? nullptr : &policy_it->second;
            if (!policy_allows(operation, config, override_policy)) continue;
            if (override_policy != nullptr && !override_policy->access.empty()) {
                if (override_policy->access == "read") {
                    operation.access = agent_openapi_access::read;
                    operation.read_only = true;
                    operation.requires_confirmation = false;
                } else if (override_policy->access == "destructive") {
                    operation.access = agent_openapi_access::destructive;
                } else if (override_policy->access == "write") {
                    operation.access = agent_openapi_access::write;
                }
            }
            catalog.operations.push_back(std::move(operation));
        }
    }
    if (catalog.operations.empty() && config.exposure == "include") {
        error = "OpenAPI include policy did not expose any operations";
        return false;
    }
    return true;
}

std::string agent_openapi_exposed_tool_name(
        const agent_openapi_catalog & catalog,
        const agent_openapi_operation & operation) {
    return catalog.prefix.empty() ? operation.operation_id : catalog.prefix + "." + operation.operation_id;
}
