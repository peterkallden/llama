#include "agent_android_resource_catalog.h"
#include "agent/tooling/contracts/tool-result-contracts.h"
#include "plan/plan-resource-provenance-json.h"
#include <nlohmann/json.hpp>
#include <stdexcept>

using json = nlohmann::ordered_json;

namespace {
json encode(const agent_resource_descriptor & d) {
    json v = common_tool_resource_descriptor_to_json(d);
    common_plan_resource_provenance_to_json(d, v);
    v.update(json{{"schema_version", 1}, {"resource_id", d.resource_id}, {"namespace_id", d.namespace_id},
        {"session_id", d.session_id}, {"project_id", d.project_id}, {"turn_id", d.turn_id},
        {"tool_call_id", d.tool_call_id}, {"source_provider", d.source_provider},
        {"source_tool", d.source_tool}, {"created_at", d.created_at}, {"expires_at", d.expires_at}});
    return v;
}
bool decode(const std::string & text, agent_resource_descriptor & d, std::string & error) {
    try {
        const auto v = json::parse(text);
        if (v.at("schema_version").get<int>() != 1) throw std::runtime_error("unsupported version");
        d = {};
        d.uri = v.at("uri").get<std::string>(); d.resource_id = v.at("resource_id").get<std::string>();
        d.name = v.at("name").get<std::string>(); d.description = v.at("description").get<std::string>();
        d.mime_type = v.at("mime_type").get<std::string>(); d.sha256 = v.at("sha256").get<std::string>();
        d.size_bytes = v.at("size_bytes").get<size_t>();
        const auto scope = v.at("scope").get<std::string>();
        if (scope == "session") d.scope = common_runtime_resource_scope::session;
        else if (scope == "project") d.scope = common_runtime_resource_scope::project;
        else if (scope == "turn") d.scope = common_runtime_resource_scope::turn;
        else throw std::runtime_error("invalid scope");
        d.namespace_id = v.at("namespace_id").get<std::string>(); d.session_id = v.at("session_id").get<std::string>();
        d.project_id = v.at("project_id").get<std::string>(); d.turn_id = v.at("turn_id").get<std::string>();
        d.tool_call_id = v.at("tool_call_id").get<std::string>(); d.source_provider = v.at("source_provider").get<std::string>();
        d.source_tool = v.at("source_tool").get<std::string>(); d.created_at = v.at("created_at").get<int64_t>();
        d.expires_at = v.at("expires_at").get<int64_t>();
        const auto & m = v.at("metadata");
        d.metadata.purpose = m.value("purpose", ""); d.metadata.content_summary = m.value("content_summary", "");
        d.metadata.usage_hint = m.value("usage_hint", ""); d.metadata.limitations = m.value("limitations", "");
        d.metadata.keywords = m.value("keywords", std::vector<std::string>{}); d.metadata.entities = m.value("entities", std::vector<std::string>{});
        d.metadata.processing_cache_key = m.value("processing_cache_key", ""); d.metadata.declared_language = m.value("declared_language", "");
        d.metadata.resolved_language = m.value("resolved_language", ""); d.metadata.language_confidence = m.value("language_confidence", 0.0);
        d.metadata.language_source = m.value("language_source", "");
        common_plan_resource_provenance_from_json(v, d);
        error.clear(); return true;
    } catch (const std::exception &) { error = "invalid Android resource catalog record"; return false; }
}
}

bool android_resource_catalog::open(const std::string & path, std::string & error) {
    return db_.open(path, error) && db_.execute("CREATE TABLE IF NOT EXISTS resource_sequence (id INTEGER PRIMARY KEY AUTOINCREMENT);"
        "CREATE TABLE IF NOT EXISTS resource_catalog (uri TEXT PRIMARY KEY, descriptor TEXT NOT NULL);", error);
}
bool android_resource_catalog::next_resource_id(std::string & out, std::string & error) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!db_.execute("INSERT INTO resource_sequence DEFAULT VALUES;", error)) return false;
    common_sqlite_statement stmt; bool row = false;
    if (!db_.prepare("SELECT last_insert_rowid();", stmt, error) || !stmt.step(row, error) || !row) return false;
    out = "resource-" + std::to_string(stmt.column_int64(0)); return true;
}
bool android_resource_catalog::put_descriptor(const agent_resource_descriptor & d, std::string & error) {
    std::lock_guard<std::mutex> lock(mutex_); common_sqlite_statement stmt; bool row = false;
    return db_.prepare("INSERT INTO resource_catalog(uri,descriptor) VALUES(?1,?2);", stmt, error) &&
        stmt.bind_text(1, d.uri, error) && stmt.bind_text(2, encode(d).dump(), error) && stmt.step(row, error);
}
bool android_resource_catalog::find_descriptor(const std::string & uri, agent_resource_descriptor & out, std::string & error) const {
    std::lock_guard<std::mutex> lock(mutex_); common_sqlite_statement stmt; bool row = false;
    if (!db_.prepare("SELECT descriptor FROM resource_catalog WHERE uri=?1;", stmt, error) ||
            !stmt.bind_text(1, uri, error) || !stmt.step(row, error)) return false;
    if (!row) { error = "resource was not found"; return false; }
    const auto * text = stmt.column_text(0);
    if (!text) { error = "resource descriptor is missing"; return false; }
    return decode(reinterpret_cast<const char *>(text), out, error);
}
bool android_resource_catalog::list_descriptors(std::vector<agent_resource_descriptor> & out, std::string & error) const {
    std::lock_guard<std::mutex> lock(mutex_); common_sqlite_statement stmt; bool row = false;
    if (!db_.prepare("SELECT descriptor FROM resource_catalog ORDER BY uri;", stmt, error)) return false;
    std::vector<agent_resource_descriptor> result;
    while (true) {
        if (!stmt.step(row, error)) return false;
        if (!row) break;
        const auto * text = stmt.column_text(0); agent_resource_descriptor descriptor;
        if (!text || !decode(reinterpret_cast<const char *>(text), descriptor, error)) return false;
        result.push_back(std::move(descriptor));
    }
    out = std::move(result); return true;
}
