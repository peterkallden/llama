#include "agent-resource-store.h"

#ifdef LLAMA_MEMORY_USE_COZO

#include <nlohmann/json.hpp>

#include <algorithm>
#include <ctime>

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
                expires_at
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
                0
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
                expires_at: Int
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
    return descriptor;
}

bool resource_expired(const agent_resource_descriptor & descriptor, int64_t now) {
    return descriptor.expires_at > 0 && now > 0 && now >= descriptor.expires_at;
}

bool authority_allows(
    const agent_resource_descriptor & descriptor,
    const agent_resource_read_authority & authority,
    std::string & error) {
    if (descriptor.namespace_id != authority.namespace_id) {
        error = "resource authority namespace mismatch";
        return false;
    }
    if (resource_expired(descriptor, authority.now)) {
        error = "resource has expired";
        return false;
    }
    switch (descriptor.scope) {
        case common_runtime_resource_scope::turn:
            if (descriptor.session_id != authority.session_id || descriptor.turn_id != authority.turn_id) {
                error = "resource authority turn mismatch";
                return false;
            }
            return true;
        case common_runtime_resource_scope::session:
            if (descriptor.session_id != authority.session_id) {
                error = "resource authority session mismatch";
                return false;
            }
            return true;
        case common_runtime_resource_scope::project:
            if (descriptor.project_id != authority.project_id) {
                error = "resource authority project mismatch";
                return false;
            }
            return true;
    }
    error = "resource scope is invalid";
    return false;
}

} // namespace

agent_cozo_resource_store::agent_cozo_resource_store(std::shared_ptr<agent_blob_store> blob_store) :
    blob_store_(std::move(blob_store)) {
    if (!blob_store_) {
        blob_store_ = std::make_shared<agent_in_memory_blob_store>();
    }
}

agent_cozo_resource_store::~agent_cozo_resource_store() {
    close();
}

bool agent_cozo_resource_store::run(
    const std::string & script,
    const std::string & params_json,
    std::string & result_json,
    std::string & error) const {
    if (db_id_ < 0) {
        error = "Cozo resource store is not open";
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

bool agent_cozo_resource_store::open(const std::string & path, std::string & error) {
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
            "?[resource_id] := *agent_resource[uri, resource_id, name, description, mime_type, size_bytes, scope, sha256, namespace_id, session_id, project_id, turn_id, tool_call_id, source_provider, source_tool, created_at, expires_at], starts_with(resource_id, $prefix)",
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

void agent_cozo_resource_store::close() {
    if (db_id_ >= 0) {
        cozo_close_db(db_id_);
        db_id_ = -1;
    }
}

bool agent_cozo_resource_store::get_descriptor(
    const std::string & uri,
    agent_resource_descriptor & out,
    std::string & error) const {
    const json params = {{"uri", uri}};
    std::string result;
    if (!run(
            "?[uri, resource_id, name, description, mime_type, size_bytes, scope, sha256, namespace_id, session_id, project_id, turn_id, tool_call_id, source_provider, source_tool, created_at, expires_at] := *agent_resource[uri, resource_id, name, description, mime_type, size_bytes, scope, sha256, namespace_id, session_id, project_id, turn_id, tool_call_id, source_provider, source_tool, created_at, expires_at], uri == $uri",
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

bool agent_cozo_resource_store::put_text(
    const agent_resource_put_request & request,
    agent_resource_descriptor & out,
    std::string & error) {
    agent_blob_descriptor blob;
    if (!blob_store_->put_bytes(request.text, blob, error)) {
        return false;
    }

    agent_resource_descriptor descriptor;
    descriptor.resource_id = "resource-" + std::to_string(next_id_++);
    descriptor.uri = "agent-resource://resource/" + descriptor.resource_id;
    descriptor.name = request.name;
    descriptor.description = request.description;
    descriptor.mime_type = request.mime_type;
    descriptor.size_bytes = request.text.size();
    descriptor.scope = request.scope;
    descriptor.sha256 = blob.sha256;
    descriptor.namespace_id = request.namespace_id;
    descriptor.session_id = request.session_id;
    descriptor.project_id = request.project_id;
    descriptor.turn_id = request.turn_id;
    descriptor.tool_call_id = request.tool_call_id;
    descriptor.source_provider = request.source_provider;
    descriptor.source_tool = request.source_tool;
    descriptor.created_at = request.created_at > 0 ? request.created_at : static_cast<int64_t>(std::time(nullptr));
    descriptor.expires_at = request.expires_at;

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
        })})},
    };
    std::string result;
    if (!run(
            "?[uri, resource_id, name, description, mime_type, size_bytes, scope, sha256, namespace_id, session_id, project_id, turn_id, tool_call_id, source_provider, source_tool, created_at, expires_at] <- $rows :put agent_resource { uri => resource_id, name, description, mime_type, size_bytes, scope, sha256, namespace_id, session_id, project_id, turn_id, tool_call_id, source_provider, source_tool, created_at, expires_at }",
            params.dump(),
            result,
            error)) {
        return false;
    }

    out = descriptor;
    error.clear();
    return true;
}

bool agent_cozo_resource_store::read_text(
    const std::string & uri,
    const agent_resource_read_authority & authority,
    size_t max_bytes,
    std::string & out,
    std::string & error) const {
    agent_resource_descriptor descriptor;
    if (!get_descriptor(uri, descriptor, error)) {
        return false;
    }
    if (!authority_allows(descriptor, authority, error)) {
        return false;
    }
    return blob_store_->get_bytes(descriptor.sha256, max_bytes, out, error);
}

bool agent_cozo_resource_store::stat(
    const std::string & uri,
    const agent_resource_read_authority & authority,
    agent_resource_descriptor & out,
    std::string & error) const {
    if (!get_descriptor(uri, out, error)) {
        return false;
    }
    if (!authority_allows(out, authority, error)) {
        return false;
    }
    error.clear();
    return true;
}

#endif
