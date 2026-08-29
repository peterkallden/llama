#include "agent-learning-lifecycle-store-cozo.h"

extern "C" {
#include <cozo_c.h>
}

#include <nlohmann/json.hpp>

#include <algorithm>

using json = nlohmann::ordered_json;

namespace {

const char * schema = R"COZO(
    {
        ?[event_id, idempotency_key, subject_id, kind, status, namespace_id, project_id, session_id, source_id, content_hash, created_at, payload_json] <- [['__probe__', '__probe__', '__probe__', 'candidate', 'observed', 'local', '', '', '', 'probe', '', '{}']]
        :create agent_learning_lifecycle {
            event_id: String =>
            idempotency_key: String,
            subject_id: String,
            kind: String,
            status: String,
            namespace_id: String,
            project_id: String,
            session_id: String,
            source_id: String,
            content_hash: String,
            created_at: String,
            payload_json: String
        }
    }
    {
        ?[event_id] <- [['__probe__']]
        :delete agent_learning_lifecycle { event_id }
    }
)COZO";

bool relation_exists(const json & relations) {
    if (!relations.is_object() || !relations.contains("rows") || !relations["rows"].is_array()) return false;
    for (const auto & row : relations["rows"]) {
        if (row.is_array() && !row.empty() && row[0].is_string() &&
                row[0].get<std::string>() == "agent_learning_lifecycle") return true;
    }
    return false;
}

bool parse_rows(const std::string & raw,
        std::vector<common_learning_lifecycle_record> & result, std::string & error) {
    const auto value = json::parse(raw, nullptr, false);
    if (!value.is_object() || !value.contains("rows") || !value["rows"].is_array()) {
        error = "Cozo lifecycle query returned invalid rows";
        return false;
    }
    for (const auto & row : value["rows"]) {
        if (!row.is_array() || row.size() < 12 || !row[11].is_string()) {
            error = "Cozo lifecycle row is invalid";
            return false;
        }
        common_learning_lifecycle_record record;
        if (!common_learning_lifecycle_from_json(row[11].get<std::string>(), record, error) ||
                !common_learning_lifecycle_validate(record, 4 * 1024 * 1024, error)) return false;
        result.push_back(std::move(record));
    }
    std::sort(result.begin(), result.end(), [](const auto & left, const auto & right) {
        return left.event_id < right.event_id;
    });
    return true;
}

} // namespace

common_agent_cozo_learning_lifecycle_store::~common_agent_cozo_learning_lifecycle_store() { close(); }

bool common_agent_cozo_learning_lifecycle_store::run(
        const std::string & script, const std::string & params_json,
        std::string & result_json, std::string & error) const {
    if (db_id_ < 0) { error = "Cozo lifecycle store is not open"; return false; }
    char * result = cozo_run_query(db_id_, script.c_str(), params_json.empty() ? "{}" : params_json.c_str(), false);
    if (!result) { error = "Cozo lifecycle query failed without diagnostic output"; return false; }
    result_json = result;
    cozo_free_str(result);
    const auto parsed = json::parse(result_json, nullptr, false);
    if (parsed.is_object() && parsed.value("ok", true) == false) {
        error = parsed.value("message", std::string("Cozo lifecycle query failed"));
        return false;
    }
    return true;
}

bool common_agent_cozo_learning_lifecycle_store::open(const std::string & path, std::string & error) {
    close();
    if (path.empty()) { error = "Cozo lifecycle store requires a path"; return false; }
    char * open_error = cozo_open_db("sqlite", path.c_str(), "{}", &db_id_);
    if (open_error) { error = open_error; cozo_free_str(open_error); db_id_ = -1; return false; }
    std::string relations;
    if (!run("::relations", "{}", relations, error)) { close(); return false; }
    if (!relation_exists(json::parse(relations, nullptr, false))) {
        std::string ignored;
        if (!run(schema, "{}", ignored, error)) { close(); return false; }
    }
    return true;
}

void common_agent_cozo_learning_lifecycle_store::close() {
    if (db_id_ >= 0) { cozo_close_db(db_id_); db_id_ = -1; }
}

bool common_agent_cozo_learning_lifecycle_store::append(
        const common_learning_lifecycle_record & record, std::string & error) {
    if (!common_learning_lifecycle_validate(record, 4 * 1024 * 1024, error)) return false;
    bool exists = false;
    if (!contains_idempotency(record.idempotency_key, exists, error)) return false;
    if (exists) {
        const auto values = list(error);
        if (!error.empty()) return false;
        for (const auto & item : values) if (item.idempotency_key == record.idempotency_key) {
            if (item.event_id != record.event_id || item.payload_json != record.payload_json || item.status != record.status) {
                error = "lifecycle idempotency key conflicts with existing record";
                return false;
            }
            return true;
        }
    }
    const auto params = json{
        {"event_id", record.event_id},
        {"idempotency_key", record.idempotency_key},
        {"subject_id", record.subject_id},
        {"kind", common_learning_lifecycle_kind_name(record.kind)},
        {"status", common_learning_lifecycle_status_name(record.status)},
        {"namespace_id", record.namespace_id},
        {"project_id", record.project_id},
        {"session_id", record.session_id},
        {"source_id", record.source_id},
        {"content_hash", record.content_hash},
        {"created_at", record.created_at},
        {"payload_json", common_learning_lifecycle_to_json(record)},
    };
    std::string result;
    return run("?[event_id, idempotency_key, subject_id, kind, status, namespace_id, project_id, session_id, source_id, content_hash, created_at, payload_json] <- [[$event_id, $idempotency_key, $subject_id, $kind, $status, $namespace_id, $project_id, $session_id, $source_id, $content_hash, $created_at, $payload_json]] :put agent_learning_lifecycle { event_id => idempotency_key, subject_id, kind, status, namespace_id, project_id, session_id, source_id, content_hash, created_at, payload_json }", params.dump(), result, error);
}

bool common_agent_cozo_learning_lifecycle_store::contains_idempotency(
        const std::string & key, bool & contains, std::string & error) const {
    contains = false;
    std::string result;
    const auto params = json({{"idempotency_key", key}}).dump();
    if (!run("?[idempotency_key] := *agent_learning_lifecycle[event_id, idempotency_key, subject_id, kind, status, namespace_id, project_id, session_id, source_id, content_hash, created_at, payload_json], idempotency_key == $idempotency_key", params, result, error)) return false;
    const auto value = json::parse(result, nullptr, false);
    if (!value.is_object() || !value.contains("rows") || !value["rows"].is_array()) { error = "Cozo lifecycle idempotency query returned invalid rows"; return false; }
    contains = !value["rows"].empty();
    return true;
}

std::vector<common_learning_lifecycle_record> common_agent_cozo_learning_lifecycle_store::list(std::string & error) const {
    error.clear();
    std::vector<common_learning_lifecycle_record> result;
    std::string raw;
    if (!run("?[event_id, idempotency_key, subject_id, kind, status, namespace_id, project_id, session_id, source_id, content_hash, created_at, payload_json] := *agent_learning_lifecycle[event_id, idempotency_key, subject_id, kind, status, namespace_id, project_id, session_id, source_id, content_hash, created_at, payload_json]", "{}", raw, error)) return result;
    if (!parse_rows(raw, result, error)) result.clear();
    return result;
}
