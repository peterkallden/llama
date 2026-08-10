#include "agent-resource-catalog.h"

#ifdef LLAMA_MEMORY_USE_COZO

#include <nlohmann/json.hpp>

#include <algorithm>

extern "C" {
#include <cozo_c.h>
}

using json = nlohmann::ordered_json;

namespace {

std::string agent_resource_cozo_schema_script() {
    return R"COZO(
        {
            ?[
                uri,
                resource_id,
                name,
                description,
                mime_type,
                size_bytes,
                scope,
                sha256,
                namespace_id,
                session_id,
                project_id,
                turn_id,
                tool_call_id,
                source_provider,
                source_tool,
                created_at,
                expires_at,
                purpose,
                content_summary,
                usage_hint,
                limitations,
                keywords_json,
                entities_json,
                processing_cache_key,
                declared_language,
                resolved_language,
                language_confidence,
                language_source
            ] <- [[
                'agent-resource://schema/probe',
                'resource-probe',
                '',
                '',
                'text/plain',
                0,
                'turn',
                '',
                'local',
                'default',
                '',
                '',
                '',
                '',
                '',
                0,
                0,
                '',
                '',
                '',
                '',
                '[]',
                '[]',
                '',
                '',
                '',
                0.0,
                ''
            ]]
            :create agent_resource {
                uri: String =>
                resource_id: String,
                name: String,
                description: String,
                mime_type: String,
                size_bytes: Int,
                scope: String,
                sha256: String,
                namespace_id: String,
                session_id: String,
                project_id: String,
                turn_id: String,
                tool_call_id: String,
                source_provider: String,
                source_tool: String,
                created_at: Int,
                expires_at: Int,
                purpose: String,
                content_summary: String,
                usage_hint: String,
                limitations: String,
                keywords_json: String,
                entities_json: String,
                processing_cache_key: String,
                declared_language: String,
                resolved_language: String,
                language_confidence: Float,
                language_source: String
            }
        }
        {
            ?[uri] <- [['agent-resource://schema/probe']]
            :delete agent_resource { uri }
        }
    )COZO";
}

agent_resource_descriptor descriptor_from_row(const json & row) {
    agent_resource_descriptor descriptor;
    descriptor.uri = row.at(0).get<std::string>();
    descriptor.resource_id = row.at(1).get<std::string>();
    descriptor.name = row.at(2).get<std::string>();
    descriptor.description = row.at(3).get<std::string>();
    descriptor.mime_type = row.at(4).get<std::string>();
    descriptor.size_bytes = static_cast<size_t>(row.at(5).get<int64_t>());
    common_runtime_resource_scope scope = common_runtime_resource_scope::turn;
    if (const std::string scope_name = row.at(6).get<std::string>(); scope_name == "session") {
        scope = common_runtime_resource_scope::session;
    } else if (scope_name == "project") {
        scope = common_runtime_resource_scope::project;
    }
    descriptor.scope = scope;
    descriptor.sha256 = row.at(7).get<std::string>();
    descriptor.namespace_id = row.at(8).get<std::string>();
    descriptor.session_id = row.at(9).get<std::string>();
    descriptor.project_id = row.at(10).get<std::string>();
    descriptor.turn_id = row.at(11).get<std::string>();
    descriptor.tool_call_id = row.at(12).get<std::string>();
    descriptor.source_provider = row.at(13).get<std::string>();
    descriptor.source_tool = row.at(14).get<std::string>();
    descriptor.created_at = row.at(15).get<int64_t>();
    descriptor.expires_at = row.at(16).get<int64_t>();
    descriptor.metadata.purpose = row.at(17).get<std::string>();
    descriptor.metadata.content_summary = row.at(18).get<std::string>();
    descriptor.metadata.usage_hint = row.at(19).get<std::string>();
    descriptor.metadata.limitations = row.at(20).get<std::string>();
    const auto keywords = json::parse(row.at(21).get<std::string>(), nullptr, false);
    if (keywords.is_array()) {
        for (const auto & item : keywords) {
            if (item.is_string()) descriptor.metadata.keywords.push_back(item.get<std::string>());
        }
    }
    const auto entities = json::parse(row.at(22).get<std::string>(), nullptr, false);
    if (entities.is_array()) {
        for (const auto & item : entities) {
            if (item.is_string()) descriptor.metadata.entities.push_back(item.get<std::string>());
        }
    }
    descriptor.metadata.processing_cache_key = row.at(23).get<std::string>();
    descriptor.metadata.declared_language = row.at(24).get<std::string>();
    descriptor.metadata.resolved_language = row.at(25).get<std::string>();
    descriptor.metadata.language_confidence = row.at(26).get<double>();
    descriptor.metadata.language_source = row.at(27).get<std::string>();
    return descriptor;
}

} // namespace

agent_cozo_resource_catalog::~agent_cozo_resource_catalog() {
    close();
}

bool agent_cozo_resource_catalog::run(
    const std::string & script,
    const std::string & params_json,
    std::string & result_json,
    std::string & error) const {
    if (db_id_ < 0) {
        error = "Cozo resource catalog is not open";
        return false;
    }
    char * result = cozo_run_query(db_id_, script.c_str(), params_json.empty() ? "{}" : params_json.c_str(), false);
    if (!result) {
        error = "Cozo query failed without diagnostic output";
        return false;
    }
    result_json = result;
    cozo_free_str(result);

    const auto parsed = json::parse(result_json, nullptr, false);
    if (parsed.is_object() && parsed.value("ok", true) == false) {
        error = "Cozo query failed: " + parsed.value("message", std::string("unknown error"));
        return false;
    }
    error.clear();
    return true;
}

bool agent_cozo_resource_catalog::open(const std::string & path, std::string & error) {
    close();
    const std::string db_path = path.empty() ? "resource-metadata.cozo" : path;
    int32_t opened_db_id = -1;
    char * open_error = cozo_open_db("sqlite", db_path.c_str(), "{}", &opened_db_id);
    if (open_error != nullptr) {
        error = open_error;
        cozo_free_str(open_error);
        return false;
    }
    db_id_ = opened_db_id;

    std::string result;
    if (!run("::relations", "{}", result, error)) {
        close();
        return false;
    }

    const auto relations = json::parse(result, nullptr, false);
    if (!relations.is_object() || !relations.contains("rows") || !relations["rows"].is_array()) {
        close();
        error = "Cozo returned an unexpected relation list";
        return false;
    }

    bool has_resource = false;
    for (const auto & row : relations["rows"]) {
        if (row.is_array() && !row.empty() && row[0].is_string() && row[0].get<std::string>() == "agent_resource") {
            has_resource = true;
            break;
        }
    }

    if (!has_resource && !run(agent_resource_cozo_schema_script(), "{}", result, error)) {
        close();
        return false;
    }

    const json next_id_params = {{"prefix", "resource-"}};
    if (!run(
            "?[resource_id] := *agent_resource[uri, resource_id, name, description, mime_type, size_bytes, scope, sha256, namespace_id, session_id, project_id, turn_id, tool_call_id, source_provider, source_tool, created_at, expires_at, purpose, content_summary, usage_hint, limitations, keywords_json, entities_json, processing_cache_key, declared_language, resolved_language, language_confidence, language_source], starts_with(resource_id, $prefix)",
            next_id_params.dump(),
            result,
            error)) {
        close();
        return false;
    }
    const auto ids = json::parse(result, nullptr, false);
    uint64_t max_id = 0;
    if (ids.is_object() && ids.contains("rows")) {
        for (const auto & row : ids["rows"]) {
            if (!row.is_array() || row.empty() || !row[0].is_string()) {
                continue;
            }
            const std::string value = row[0].get<std::string>();
            if (value.rfind("resource-", 0) != 0) {
                continue;
            }
            try {
                max_id = std::max<uint64_t>(max_id, std::stoull(value.substr(9)));
            } catch (...) {
            }
        }
    }
    next_id_ = max_id + 1;
    error.clear();
    return true;
}

void agent_cozo_resource_catalog::close() {
    if (db_id_ >= 0) {
        cozo_close_db(db_id_);
        db_id_ = -1;
    }
}

bool agent_cozo_resource_catalog::next_resource_id(
    std::string & out,
    std::string & error) {
    out = "resource-" + std::to_string(next_id_++);
    error.clear();
    return true;
}

bool agent_cozo_resource_catalog::put_descriptor(
    const agent_resource_descriptor & descriptor,
    std::string & error) {
    const json params = {
        {"rows", json::array({json::array({
            descriptor.uri,
            descriptor.resource_id,
            descriptor.name,
            descriptor.description,
            descriptor.mime_type,
            static_cast<int64_t>(descriptor.size_bytes),
            common_runtime_resource_scope_name(descriptor.scope),
            descriptor.sha256,
            descriptor.namespace_id,
            descriptor.session_id,
            descriptor.project_id,
            descriptor.turn_id,
            descriptor.tool_call_id,
            descriptor.source_provider,
            descriptor.source_tool,
            descriptor.created_at,
            descriptor.expires_at,
            descriptor.metadata.purpose,
            descriptor.metadata.content_summary,
            descriptor.metadata.usage_hint,
            descriptor.metadata.limitations,
            json(descriptor.metadata.keywords).dump(),
            json(descriptor.metadata.entities).dump(),
            descriptor.metadata.processing_cache_key,
            descriptor.metadata.declared_language,
            descriptor.metadata.resolved_language,
            descriptor.metadata.language_confidence,
            descriptor.metadata.language_source,
        })})},
    };
    std::string result;
    return run(
        "?[uri, resource_id, name, description, mime_type, size_bytes, scope, sha256, namespace_id, session_id, project_id, turn_id, tool_call_id, source_provider, source_tool, created_at, expires_at, purpose, content_summary, usage_hint, limitations, keywords_json, entities_json, processing_cache_key, declared_language, resolved_language, language_confidence, language_source] <- $rows :put agent_resource { uri => resource_id, name, description, mime_type, size_bytes, scope, sha256, namespace_id, session_id, project_id, turn_id, tool_call_id, source_provider, source_tool, created_at, expires_at, purpose, content_summary, usage_hint, limitations, keywords_json, entities_json, processing_cache_key, declared_language, resolved_language, language_confidence, language_source }",
        params.dump(),
        result,
        error);
}

bool agent_cozo_resource_catalog::find_descriptor(
    const std::string & uri,
    agent_resource_descriptor & out,
    std::string & error) const {
    const json params = {{"uri", uri}};
    std::string result;
    if (!run(
            "?[uri, resource_id, name, description, mime_type, size_bytes, scope, sha256, namespace_id, session_id, project_id, turn_id, tool_call_id, source_provider, source_tool, created_at, expires_at, purpose, content_summary, usage_hint, limitations, keywords_json, entities_json, processing_cache_key, declared_language, resolved_language, language_confidence, language_source] := *agent_resource[uri, resource_id, name, description, mime_type, size_bytes, scope, sha256, namespace_id, session_id, project_id, turn_id, tool_call_id, source_provider, source_tool, created_at, expires_at, purpose, content_summary, usage_hint, limitations, keywords_json, entities_json, processing_cache_key, declared_language, resolved_language, language_confidence, language_source], uri == $uri",
            params.dump(),
            result,
            error)) {
        return false;
    }

    const auto parsed = json::parse(result, nullptr, false);
    if (!parsed.is_object() || !parsed.contains("rows") || parsed["rows"].empty()) {
        error = "resource was not found";
        return false;
    }
    out = descriptor_from_row(parsed["rows"][0]);
    error.clear();
    return true;
}

bool agent_cozo_resource_catalog::list_descriptors(
    std::vector<agent_resource_descriptor> & out,
    std::string & error) const {
    std::string result;
    if (!run(
            "?[uri, resource_id, name, description, mime_type, size_bytes, scope, sha256, namespace_id, session_id, project_id, turn_id, tool_call_id, source_provider, source_tool, created_at, expires_at, purpose, content_summary, usage_hint, limitations, keywords_json, entities_json, processing_cache_key, declared_language, resolved_language, language_confidence, language_source] := *agent_resource[uri, resource_id, name, description, mime_type, size_bytes, scope, sha256, namespace_id, session_id, project_id, turn_id, tool_call_id, source_provider, source_tool, created_at, expires_at, purpose, content_summary, usage_hint, limitations, keywords_json, entities_json, processing_cache_key, declared_language, resolved_language, language_confidence, language_source]",
            "{}",
            result,
            error)) {
        return false;
    }

    const auto parsed = json::parse(result, nullptr, false);
    if (!parsed.is_object() || !parsed.contains("rows") || !parsed["rows"].is_array()) {
        error = "unexpected Cozo resource list result";
        return false;
    }

    out.clear();
    for (const auto & row : parsed["rows"]) {
        if (row.is_array()) {
            out.push_back(descriptor_from_row(row));
        }
    }
    error.clear();
    return true;
}

#endif
