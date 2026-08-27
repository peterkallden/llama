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

void resolve_local_refs(const nlohmann::json & document, nlohmann::json & value, std::set<std::string> & seen) {
    if (value.is_object() && value.contains("$ref") && value["$ref"].is_string()) {
        const std::string ref = value["$ref"].get<std::string>();
        const std::string prefix = "#/components/schemas/";
        if (ref.rfind(prefix, 0) == 0 && seen.insert(ref).second) {
            const auto it = document.find("components");
            if (it != document.end() && it->is_object() && (*it).contains("schemas") && (*it)["schemas"].is_object()) {
                const auto schema_it = (*it)["schemas"].find(ref.substr(prefix.size()));
                if (schema_it != (*it)["schemas"].end()) {
                    value = *schema_it;
                    resolve_local_refs(document, value, seen);
                }
            }
        }
        return;
    }
    if (value.is_object()) for (auto & item : value.items()) resolve_local_refs(document, item.value(), seen);
    if (value.is_array()) for (auto & item : value) resolve_local_refs(document, item, seen);
}

std::string operation_input_schema(
        const nlohmann::json & document,
        const nlohmann::json & operation,
        std::vector<std::string> & path_parameters,
        std::vector<std::string> & query_parameters) {
    nlohmann::json schema = {
        {"type", "object"},
        {"properties", nlohmann::json::object()},
        {"required", nlohmann::json::array()},
    };
    nlohmann::json inferable = nlohmann::json::array();
    if (operation.contains("parameters") && operation["parameters"].is_array()) {
        for (const auto & parameter : operation["parameters"]) {
            if (!parameter.is_object() || !parameter.value("name", "").size()) continue;
            const std::string name = parameter.value("name", "");
            if (parameter.value("in", "") == "path") path_parameters.push_back(name);
            if (parameter.value("in", "") == "query") query_parameters.push_back(name);
            schema["properties"][name] = parameter.value("schema", nlohmann::json({{"type", "string"}}));
            std::set<std::string> seen;
            resolve_local_refs(document, schema["properties"][name], seen);
            if (parameter.value("x-agent-inferable", false)) {
                schema["properties"][name]["x-agent-inferable"] = true;
                inferable.push_back(name);
            }
            if (parameter.value("required", false)) schema["required"].push_back(name);
        }
    }
    if (operation.contains("requestBody") && operation["requestBody"].is_object()) {
        const auto & content = operation["requestBody"].value("content", nlohmann::json::object());
        if (content.is_object() && content.contains("application/json")) {
            const auto & media = content["application/json"];
            if (media.is_object() && media.contains("schema")) {
                schema["properties"]["body"] = media["schema"];
                std::set<std::string> seen;
                resolve_local_refs(document, schema["properties"]["body"], seen);
            }
            if (operation["requestBody"].value("required", false)) schema["required"].push_back("body");
        }
    }
    if (!inferable.empty()) schema["x-agent-autowire-fields"] = std::move(inferable);
    if (schema["required"].empty()) schema.erase("required");
    return schema.dump();
}

std::string operation_result_schema(
        const nlohmann::json & document,
        const nlohmann::json & operation) {
    if (!operation.contains("responses") || !operation["responses"].is_object()) {
        return R"({"type":"object"})";
    }
    const nlohmann::json * response = nullptr;
    for (const auto code : {"200", "201", "202", "default"}) {
        const auto it = operation["responses"].find(code);
        if (it != operation["responses"].end() && it->is_object()) {
            response = &*it;
            break;
        }
    }
    if (response == nullptr) return R"({"type":"object"})";
    const auto content = response->value("content", nlohmann::json::object());
    if (!content.is_object() || !content.contains("application/json") ||
            !content["application/json"].is_object()) return R"({"type":"object"})";
    auto schema = content["application/json"].value("schema", nlohmann::json({{"type", "object"}}));
    std::set<std::string> seen;
    resolve_local_refs(document, schema, seen);
    return schema.dump();
}

std::string collection_path_for(const std::string & path, std::string & item_parameter) {
    item_parameter.clear();
    std::string collection_path;
    size_t cursor = 0;
    while (cursor < path.size()) {
        const size_t open = path.find('{', cursor);
        if (open == std::string::npos) {
            collection_path += path.substr(cursor);
            break;
        }
        collection_path += path.substr(cursor, open - cursor);
        const size_t close = path.find('}', open + 1);
        if (close == std::string::npos || !item_parameter.empty()) return {};
        item_parameter = path.substr(open + 1, close - open - 1);
        cursor = close + 1;
    }
    while (collection_path.size() > 1 && collection_path.back() == '/') collection_path.pop_back();
    return collection_path.empty() ? "/" : collection_path;
}

}

bool classify_agent_openapi_result_json(
        const std::string & structured_content_json,
        const agent_openapi_result_projection_limits & limits,
        agent_openapi_result_projection & projection,
        std::string & error) {
    projection = {};
    if (structured_content_json.size() > limits.max_bytes) {
        projection.reason = "structured API result exceeds the materialization byte limit";
        error.clear();
        return true;
    }
    const auto value = nlohmann::json::parse(structured_content_json, nullptr, false);
    if (value.is_discarded()) {
        error = "OpenAPI structured result is not valid JSON";
        return false;
    }
    if (!value.is_array() || value.empty() || value.size() > limits.max_rows) {
        projection.reason = value.is_array() && value.size() > limits.max_rows
            ? "collection exceeds the materialization row limit"
            : "result is not a non-empty bounded JSON collection";
        error.clear();
        return true;
    }
    std::set<std::string> columns;
    size_t cells = 0;
    for (const auto & row : value) {
        if (!row.is_object()) {
            projection.reason = "collection contains a non-object item";
            error.clear();
            return true;
        }
        cells += row.size();
        if (cells > limits.max_cells) {
            projection.reason = "collection exceeds the materialization cell limit";
            error.clear();
            return true;
        }
        for (const auto & item : row.items()) {
            if (!(item.value().is_null() || item.value().is_boolean() ||
                    item.value().is_number() || item.value().is_string())) {
                projection.reason = "collection contains nested or heterogeneous values";
                error.clear();
                return true;
            }
            columns.insert(item.key());
        }
        if (columns.size() > limits.max_columns) {
            projection.reason = "collection exceeds the materialization column limit";
            error.clear();
            return true;
        }
    }
    projection.kind = agent_openapi_result_projection_kind::dataset;
    projection.row_count = value.size();
    projection.columns.assign(columns.begin(), columns.end());
    projection.reason = "bounded shallow JSON collection is dataset-compatible";
    error.clear();
    return true;
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
            operation.input_schema_json = operation_input_schema(document, value, operation.path_parameters, operation.query_parameters);
            operation.result_schema_json = operation_result_schema(document, value);

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
    for (const auto & item : catalog.operations) {
        std::string item_parameter;
        const std::string collection_path = collection_path_for(item.path, item_parameter);
        if (item_parameter.empty() || collection_path.empty()) continue;
        for (const auto & collection : catalog.operations) {
            if (collection.operation_id == item.operation_id || collection.path != collection_path) continue;
            if (!collection.read_only || item.method != "get") continue;
            catalog.relations.push_back({
                collection.operation_id,
                item.operation_id,
                collection_path,
                item_parameter});
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

bool make_agent_openapi_item_references(
        const agent_openapi_catalog & catalog,
        const std::string & collection_operation_id,
        const std::string & structured_content_json,
        const agent_openapi_result_projection_limits & limits,
        std::vector<agent_openapi_item_reference> & references,
        std::string & error) {
    references.clear();
    if (structured_content_json.size() > limits.max_bytes) {
        error = "OpenAPI collection exceeds the materialization byte limit";
        return false;
    }
    const auto value = nlohmann::json::parse(structured_content_json, nullptr, false);
    if (value.is_discarded() || !value.is_array()) {
        error = "OpenAPI item references require a JSON collection";
        return false;
    }
    if (value.size() > limits.max_rows) {
        error = "OpenAPI collection exceeds the materialization row limit";
        return false;
    }
    const auto relation_it = std::find_if(
        catalog.relations.begin(), catalog.relations.end(),
        [&](const agent_openapi_relation & relation) {
            return relation.collection_operation_id == collection_operation_id;
        });
    if (relation_it == catalog.relations.end()) {
        error = "OpenAPI collection operation has no item relation";
        return false;
    }
    for (size_t index = 0; index < value.size(); ++index) {
        const auto & row = value[index];
        if (!row.is_object()) continue;
        const auto item_it = row.find(relation_it->item_parameter);
        if (item_it == row.end() || item_it->is_null() ||
                !(item_it->is_string() || item_it->is_boolean() || item_it->is_number())) continue;
        std::string item_value;
        if (item_it->is_string()) item_value = item_it->get<std::string>();
        else if (item_it->is_boolean()) item_value = item_it->get<bool>() ? "true" : "false";
        else item_value = item_it->dump();
        if (item_value.empty()) continue;
        references.push_back({
            relation_it->item_operation_id + "#" + std::to_string(index + 1),
            catalog.provider_id,
            relation_it->collection_operation_id,
            relation_it->item_operation_id,
            relation_it->item_parameter,
            std::move(item_value),
            index});
    }
    error.clear();
    return true;
}
