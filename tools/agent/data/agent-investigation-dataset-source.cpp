#include "agent-investigation-dataset-source.h"

#include "tools/agent/openapi/agent-openapi-http.h"

#include <algorithm>
#include <ctime>
#include <limits>
#include <set>

namespace {
using json = nlohmann::ordered_json;

const agent_host_investigation_source_config * find_source(
        const agent_host_investigation_config & config, const std::string & id) {
    const auto found = std::find_if(config.sources.begin(), config.sources.end(),
        [&](const auto & source) { return source.id == id; });
    return found == config.sources.end() ? nullptr : &*found;
}

bool mapped_field(const agent_host_investigation_source_config & source,
        const json & field, std::string & out, std::string & error) {
    if (!field.is_string()) { error = "query field must be a string"; return false; }
    const auto found = source.fields.find(field.get<std::string>());
    if (found == source.fields.end()) {
        error = "query references a field outside the configured investigation schema";
        return false;
    }
    out = found->second;
    return true;
}

bool build_filter(const agent_host_investigation_source_config & source,
        const json & where, json & filters, std::string & error) {
    if (!where.is_array() || where.size() > 16) {
        error = "virtual investigation query where must contain at most 16 predicates";
        return false;
    }
    filters = json::array();
    for (const auto & predicate : where) {
        if (!predicate.is_object() || !predicate.contains("field") ||
                !predicate.contains("operator")) {
            error = "query predicate requires field and operator";
            return false;
        }
        std::string field;
        if (!mapped_field(source, predicate["field"], field, error)) return false;
        const auto op = predicate["operator"].is_string()
            ? predicate["operator"].get<std::string>() : std::string();
        const bool needs_value = op != "is_null" && op != "not_null";
        if (needs_value && !predicate.contains("value")) {
            error = "query predicate requires a value";
            return false;
        }
        if (op == "=" || op == "==") filters.push_back({{"term", {{field, predicate["value"]}}}});
        else if (op == "!=") filters.push_back({{"bool", {{"must_not", json::array({{{"term", {{field, predicate["value"]}}}}})}}}});
        else if (op == ">" || op == ">=" || op == "<" || op == "<=") {
            const std::string range_op = op == ">" ? "gt" : op == ">=" ? "gte" : op == "<" ? "lt" : "lte";
            filters.push_back({{"range", {{field, {{range_op, predicate["value"]}}}}}});
        } else if (op == "in" && predicate["value"].is_array() &&
                !predicate["value"].empty() && predicate["value"].size() <= 128) {
            filters.push_back({{"terms", {{field, predicate["value"]}}}});
        } else if (op == "contains" && predicate["value"].is_string() &&
                predicate["value"].get<std::string>().size() <= 256) {
            filters.push_back({{"match_phrase", {{field, predicate["value"]}}}});
        } else if (op == "is_null") filters.push_back({{"bool", {{"must_not", json::array({{{"exists", {{"field", field}}}}})}}}});
        else if (op == "not_null") filters.push_back({{"exists", {{"field", field}}}});
        else {
            error = "query predicate uses an unsupported operator or value";
            return false;
        }
    }
    return true;
}

json * lookup_path(json & value, const std::string & path) {
    size_t begin = 0;
    while (begin <= path.size()) {
        const size_t end = path.find('.', begin);
        const auto part = path.substr(begin, end == std::string::npos ? end : end - begin);
        if (!value.is_object() || !value.contains(part)) return nullptr;
        value = value[part];
        if (end == std::string::npos) return &value;
        begin = end + 1;
    }
    return nullptr;
}

bool parse_request(const std::string & raw, json & value, std::string & error) {
    value = json::parse(raw, nullptr, false);
    if (!value.is_object()) { error = "virtual dataset query must be a JSON object"; return false; }
    return true;
}

bool read_nonnegative_size(const json & object, const char * key, size_t fallback,
        size_t & value, std::string & error) {
    value = fallback;
    if (!object.contains(key)) return true;
    const auto & field = object[key];
    if (!field.is_number_integer() && !field.is_number_unsigned()) {
        error = std::string("virtual query field '") + key + "' must be a nonnegative integer";
        return false;
    }
    if (!field.is_number_unsigned()) {
        const int64_t signed_value = field.get<int64_t>();
        if (signed_value < 0) {
            error = std::string("virtual query field '") + key + "' must be a nonnegative integer";
            return false;
        }
        value = static_cast<size_t>(signed_value);
        return true;
    }
    const uint64_t unsigned_value = field.get<uint64_t>();
    if (unsigned_value > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        error = std::string("virtual query field '") + key + "' exceeds the host size bound";
        return false;
    }
    value = static_cast<size_t>(unsigned_value);
    return true;
}
}

bool agent_investigation_search_page_complete(bool total_exact, bool timed_out,
        bool shard_status_known, size_t failed_shards, size_t offset,
        size_t returned_rows, size_t total) {
    return total_exact && !timed_out && shard_status_known && failed_shards == 0 &&
        offset <= total && returned_rows <= total - offset && offset + returned_rows >= total;
}

agent_investigation_virtual_dataset_source::agent_investigation_virtual_dataset_source(
        agent_host_investigation_config config) : config_(std::move(config)) {}

bool agent_investigation_virtual_dataset_source::authorize_source(
        const std::string & source_id, const common_agent_scope & scope, std::string & error) const {
    const auto * source = find_source(config_, source_id);
    if (!config_.enabled || !source) { error = "investigation source is not currently configured"; return false; }
    if (std::find(source->allowed_namespaces.begin(), source->allowed_namespaces.end(), scope.namespace_id) == source->allowed_namespaces.end()) {
        error = "investigation source is not authorized for this namespace"; return false;
    }
    if (!source->allowed_projects.empty() &&
            std::find(source->allowed_projects.begin(), source->allowed_projects.end(), scope.project_id) == source->allowed_projects.end()) {
        error = "investigation source is not authorized for this project"; return false;
    }
    error.clear();
    return true;
}

bool agent_investigation_virtual_dataset_source::register_datasets(
        common_agent_data_store & store, std::string & error) const {
    if (!config_.enabled) { error = "investigation feature is disabled"; return false; }
    for (const auto & source : config_.sources) {
        for (const auto & index : source.allowed_indices) {
            common_agent_dataset_descriptor descriptor;
            descriptor.ref.uri = "dataset://investigation/" + source.id + "/" + index;
            descriptor.ref.name = source.id + ":" + index;
            descriptor.ref.source_representation = source.dialect + ":search";
            descriptor.ref.source_provider = source.id;
            descriptor.ref.source_operation = "search";
            descriptor.ref.source_request_json = json{{"index", index}, {"dialect", source.dialect}}.dump();
            descriptor.ref.retrieved_at = static_cast<int64_t>(std::time(nullptr));
            descriptor.ref.storage_mode = common_agent_dataset_storage_mode::virtual_;
            descriptor.ref.count_kind = common_agent_dataset_count_kind::unknown;
            descriptor.columns.reserve(source.fields.size());
            for (const auto & field : source.fields) {
                descriptor.columns.push_back({field.first,
                    common_agent_dataset_column_type::unknown, true});
            }
            descriptor.columns.push_back({"@source.index", common_agent_dataset_column_type::string, false});
            descriptor.columns.push_back({"@source.id", common_agent_dataset_column_type::string, false});
            descriptor.ref.column_count = descriptor.columns.size();
            descriptor.capabilities = {"filter", "project", "sort", "page"};
            descriptor.read_context_id = "investigation-read-context:" + source.id;
            if (!store.put_dataset_descriptor(descriptor, error)) return false;
        }
    }
    error.clear();
    return true;
}

bool agent_investigation_virtual_dataset_source::execute(
        const common_agent_dataset_descriptor & descriptor, const std::string & operation,
        const std::string & request_json, const common_agent_scope & scope, bool allow_network,
        std::string & result_json, std::string & error) {
    if (operation != "data.query") {
        error = "virtual investigation datasets currently support data.query only";
        return false;
    }
    const auto * source = find_source(config_, descriptor.ref.source_provider);
    if (!source || std::find(source->allowed_namespaces.begin(), source->allowed_namespaces.end(),
            scope.namespace_id) == source->allowed_namespaces.end()) {
        error = "investigation source is not authorized for this namespace";
        return false;
    }
    if (!source->allowed_projects.empty() &&
            std::find(source->allowed_projects.begin(), source->allowed_projects.end(),
                scope.project_id) == source->allowed_projects.end()) {
        error = "investigation source is not authorized for this project";
        return false;
    }
    if (!allow_network) { error = "network permission is required for this investigation source"; return false; }
    if (descriptor.ref.uri.rfind("dataset://investigation/" + source->id + "/", 0) != 0) {
        error = "virtual dataset provider is not registered for this source";
        return false;
    }
    const auto provenance = json::parse(descriptor.ref.source_request_json, nullptr, false);
    if (!provenance.is_object() || !provenance.contains("index") || !provenance["index"].is_string()) {
        error = "virtual dataset source provenance is malformed";
        return false;
    }
    const std::string index = provenance["index"].get<std::string>();
    if (std::find(source->allowed_indices.begin(), source->allowed_indices.end(), index) == source->allowed_indices.end()) {
        error = "virtual dataset index is outside its provider allowlist";
        return false;
    }
    json request;
    if (!parse_request(request_json, request, error)) return false;
    if (!request.contains("dataset") || !request["dataset"].is_string() ||
            request["dataset"].get<std::string>() != descriptor.ref.uri) {
        error = "virtual query dataset identity is invalid";
        return false;
    }
    if (request.contains("distinct") && !request["distinct"].is_boolean()) {
        error = "virtual query distinct flag must be boolean";
        return false;
    }
    if (request.value("distinct", false)) {
        error = "virtual query dataset identity is invalid or requests unsupported distinct rows";
        return false;
    }
    json selected = json::array();
    if (request.contains("select")) {
        if (!request["select"].is_array() || request["select"].size() > 32) {
            error = "virtual query projection is invalid"; return false;
        }
        for (const auto & field : request["select"]) {
            if (!field.is_string()) { error = "query projection fields must be strings"; return false; }
            const auto logical = field.get<std::string>();
            if (logical == "@source.index" || logical == "@source.id") continue;
            std::string actual;
            if (!mapped_field(*source, field, actual, error)) return false;
            selected.push_back(actual);
        }
    } else {
        for (const auto & field : source->fields) selected.push_back(field.second);
    }
    json filters;
    if (!build_filter(*source, request.value("where", json::array()), filters, error)) return false;
    json sort = json::array();
    if (request.contains("order_by")) {
        if (!request["order_by"].is_array() || request["order_by"].size() > 8) {
            error = "virtual query sort is invalid"; return false;
        }
        for (const auto & item : request["order_by"]) {
            if (!item.is_object() || !item.contains("field")) { error = "query sort needs a field"; return false; }
            std::string actual;
            if (!mapped_field(*source, item["field"], actual, error)) return false;
            if (item.contains("direction") && !item["direction"].is_string()) {
                error = "query sort direction must be a string"; return false;
            }
            const auto direction = item.value("direction", std::string("asc"));
            if (direction != "asc" && direction != "desc") { error = "query sort direction must be asc or desc"; return false; }
            sort.push_back({{actual, direction}});
        }
    }
    size_t requested_limit = source->max_page_size;
    size_t requested_result_rows = source->max_page_size;
    size_t offset = 0;
    if (!read_nonnegative_size(request, "limit", source->max_page_size, requested_limit, error) ||
            !read_nonnegative_size(request, "max_result_rows", source->max_page_size, requested_result_rows, error) ||
            !read_nonnegative_size(request, "offset", 0, offset, error)) return false;
    const size_t limit = std::min({requested_limit, requested_result_rows, source->max_page_size});
    if (limit == 0 || offset > config_.max_materialized_rows || limit > config_.max_materialized_rows - offset) {
        error = "virtual query page exceeds configured investigation bounds"; return false;
    }
    const json query = {
        {"from", offset}, {"size", limit}, {"_source", selected},
        {"query", {{"bool", {{"filter", filters}}}}}, {"sort", sort},
    };
    agent_tool_context context;
    context.scope = scope;
    context.allow_network = allow_network;
    std::string response;
    if (!agent_investigation_http_search(context, *source, index, query, response, error)) return false;
    const auto parsed = json::parse(response, nullptr, false);
    if (!parsed.is_object() || !parsed.contains("hits") || !parsed["hits"].is_object() ||
            !parsed["hits"].contains("hits") || !parsed["hits"]["hits"].is_array()) {
        error = "search provider returned an invalid hits structure"; return false;
    }
    std::vector<std::string> column_names;
    std::vector<std::string> paths;
    for (const auto & requested : (request.contains("select") ? request["select"] : json::array())) {
        if (!requested.is_string()) { error = "query projection fields must be strings"; return false; }
        const std::string logical = requested.get<std::string>();
        if (logical == "@source.index" || logical == "@source.id") {
            column_names.push_back(logical);
            paths.push_back(logical);
            continue;
        }
        const auto found = source->fields.find(logical);
        if (found == source->fields.end()) { error = "query projection uses an unmapped field"; return false; }
        column_names.push_back(found->first);
        paths.push_back(found->second);
    }
    if (!request.contains("select")) {
        for (const auto & field : source->fields) { column_names.push_back(field.first); paths.push_back(field.second); }
    }
    json rows = json::array();
    for (const auto & hit : parsed["hits"]["hits"]) {
        if (!hit.is_object() || !hit.contains("_source") || !hit["_source"].is_object()) continue;
        json values = json::array();
        for (const auto & path : paths) {
            if (path == "@source.index") { values.push_back(hit.value("_index", std::string{})); continue; }
            if (path == "@source.id") { values.push_back(hit.value("_id", std::string{})); continue; }
            auto document = hit["_source"];
            const auto * value = lookup_path(document, path);
            values.push_back(value ? *value : json());
        }
        rows.push_back(std::move(values));
    }
    size_t total = rows.size();
    bool total_exact = false;
    const auto & hits = parsed["hits"];
    if (hits.contains("total")) {
        if (hits["total"].is_number_unsigned() || hits["total"].is_number_integer()) {
            total = hits["total"].get<size_t>(); total_exact = true;
        } else if (hits["total"].is_object() && hits["total"].contains("value") && hits["total"]["value"].is_number()) {
            total = hits["total"]["value"].get<size_t>();
            total_exact = hits["total"].value("relation", std::string("eq")) == "eq";
        }
    }
    const bool timed_out = parsed.value("timed_out", false);
    size_t failed_shards = 0;
    const bool shard_status_known = parsed.contains("_shards") && parsed["_shards"].is_object() &&
        parsed["_shards"].contains("failed") && parsed["_shards"]["failed"].is_number_integer();
    if (shard_status_known) {
        failed_shards = parsed["_shards"]["failed"].get<size_t>();
    }
    const bool complete = agent_investigation_search_page_complete(total_exact, timed_out,
        shard_status_known, failed_shards, offset, rows.size(), total);
    result_json = json{{"columns", column_names}, {"rows", rows}, {"row_count", rows.size()},
        {"scanned_rows", rows.size()}, {"complete", complete},
        {"scan_truncated", !complete}, {"result_truncated", offset + rows.size() < total},
        {"total", total}, {"total_exact", total_exact}, {"timed_out", timed_out},
        {"shard_status_known", shard_status_known}, {"failed_shards", failed_shards},
        {"source_provider", source->id}}.dump();
    error.clear();
    return true;
}
