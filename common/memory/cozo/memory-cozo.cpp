#include "memory/cozo/memory-cozo.h"

#include "memory/cozo/cozo-schema.h"
#include "memory/memory-in-memory.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <ctime>
#include <unordered_set>

extern "C" {
#include <cozo_c.h>
}

using json = nlohmann::ordered_json;

static json metadata_to_json(const std::unordered_map<std::string, std::string> & metadata) {
    json out = json::object();
    for (const auto & kv : metadata) {
        out[kv.first] = kv.second;
    }
    return out;
}

static common_memory_record record_from_json(const json & row) {
    common_memory_record record;
    record.id = row.at(0).get<std::string>();
    common_memory_kind_parse(row.at(1).get<std::string>(), record.kind);
    record.content = row.at(2).get<std::string>();
    record.summary = row.at(3).get<std::string>();
    if (row.at(4).is_array()) {
        record.embedding = row.at(4).get<std::vector<float>>();
    }
    record.importance = row.at(5).get<float>();
    record.confidence = row.at(6).get<float>();
    record.created_at = row.at(7).get<int64_t>();
    record.accessed_at = row.at(8).get<int64_t>();
    record.access_count = row.at(9).get<uint64_t>();
    record.scope = common_memory_scope::session;
    record.namespace_id = "local";
    record.session_id = "default";
    if (row.size() >= 16) {
        common_memory_scope_parse(row.at(10).get<std::string>(), record.scope);
        record.namespace_id = row.at(11).get<std::string>();
        record.session_id = row.at(12).get<std::string>();
        record.project_id = row.at(13).get<std::string>();
        record.turn_id = row.at(14).get<std::string>();
    }
    const size_t metadata_index = row.size() >= 16 ? 15 : 10;
    if (!row.at(metadata_index).get<std::string>().empty()) {
        const auto metadata = json::parse(row.at(metadata_index).get<std::string>(), nullptr, false);
        if (metadata.is_object()) {
            for (auto it = metadata.begin(); it != metadata.end(); ++it) {
                if (it.value().is_string()) {
                    record.metadata[it.key()] = it.value().get<std::string>();
                }
            }
        }
    }
    return record;
}

static constexpr const char * k_memory_relation = "memory_scoped";

common_memory_cozo_store::common_memory_cozo_store() = default;

common_memory_cozo_store::~common_memory_cozo_store() {
    close();
}

bool common_memory_cozo_store::run(const std::string & script, const std::string & params_json, std::string & result_json, std::string & error) const {
    if (db_id < 0) {
        error = "Cozo memory store is not open";
        return false;
    }
    char * result = cozo_run_query(db_id, script.c_str(), params_json.empty() ? "{}" : params_json.c_str(), false);
    if (result == nullptr) {
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

bool common_memory_cozo_store::open(const std::string & path, std::string & error) {
    close();
    const std::string db_path = path.empty() ? "memory.cozo" : path;
    int32_t opened_db_id = -1;
    char * open_error = cozo_open_db("sqlite", db_path.c_str(), "{}", &opened_db_id);
    if (open_error != nullptr) {
        error = open_error;
        cozo_free_str(open_error);
        return false;
    }
    db_id = opened_db_id;
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

    std::unordered_set<std::string> names;
    for (const auto & row : relations["rows"]) {
        if (row.is_array() && !row.empty() && row[0].is_string()) {
            names.insert(row[0].get<std::string>());
        }
    }

    const bool has_memory = names.count("memory") != 0;
    const bool has_scoped_memory = names.count(k_memory_relation) != 0;
    const bool has_memory_edge = names.count("memory_edge") != 0;
    if ((has_memory || has_scoped_memory) != has_memory_edge) {
        close();
        error = "Cozo memory database has an incomplete schema; create a new PoC database";
        return false;
    }
    if (!has_memory && !has_scoped_memory && !run(common_memory_cozo_schema_script(), "{}", result, error)) {
        close();
        return false;
    }
    if (has_memory && !has_scoped_memory) {
        const char * create_scoped_relation = R"COZO(
            {
                ?[id, kind, content, summary, embedding, importance, confidence, created_at, accessed_at, access_count, scope, namespace_id, session_id, project_id, turn_id, metadata_json] <-
                    [['__schema_probe__', 'fact', '', '', [], 0.0, 0.0, 0, 0, 0, 'session', 'local', 'default', '', '', '{}']]
                :create memory_scoped {
                    id: String =>
                    kind: String,
                    content: String,
                    summary: String,
                    embedding: [Float],
                    importance: Float,
                    confidence: Float,
                    created_at: Int,
                    accessed_at: Int,
                    access_count: Int,
                    scope: String,
                    namespace_id: String,
                    session_id: String,
                    project_id: String,
                    turn_id: String,
                    metadata_json: String
                }
            }
            {
                ?[id] <- [['__schema_probe__']]
                :delete memory_scoped { id }
            }
        )COZO";
        const char * migration =
            "?[id, kind, content, summary, embedding, importance, confidence, created_at, accessed_at, access_count, scope, namespace_id, session_id, project_id, turn_id, metadata_json] := "
            "*memory[id, kind, content, summary, embedding, importance, confidence, created_at, accessed_at, access_count, old_metadata_json], "
            "scope = 'session', namespace_id = 'local', session_id = 'default', project_id = '', turn_id = '', metadata_json = old_metadata_json "
            ":put memory_scoped { id => kind, content, summary, embedding, importance, confidence, created_at, accessed_at, access_count, scope, namespace_id, session_id, project_id, turn_id, metadata_json }";
        if (!run(create_scoped_relation, "{}", result, error) || !run(migration, "{}", result, error)) {
            close();
            return false;
        }
    }
    return true;
}

void common_memory_cozo_store::close() {
    if (db_id >= 0) {
        cozo_close_db(db_id);
        db_id = -1;
    }
}

bool common_memory_cozo_store::put(const common_memory_record & record, std::string & error) {
    if (record.id.empty()) {
        error = "memory id must not be empty";
        return false;
    }
    const json params = {
        {"rows", json::array({json::array({
            record.id,
            common_memory_kind_name(record.kind),
            record.content,
            record.summary,
            record.embedding,
            record.importance,
            record.confidence,
            record.created_at,
            record.accessed_at,
            record.access_count,
            common_memory_scope_name(record.scope),
            record.namespace_id,
            record.session_id,
            record.project_id,
            record.turn_id,
            metadata_to_json(record.metadata).dump(),
        })})},
    };
    std::string result;
    return run("?[id, kind, content, summary, embedding, importance, confidence, created_at, accessed_at, access_count, scope, namespace_id, session_id, project_id, turn_id, metadata_json] <- $rows :put memory_scoped { id => kind, content, summary, embedding, importance, confidence, created_at, accessed_at, access_count, scope, namespace_id, session_id, project_id, turn_id, metadata_json }",
        params.dump(), result, error);
}

std::optional<common_memory_record> common_memory_cozo_store::get(const std::string & id, std::string & error) {
    const json params = {{"id", id}};
    std::string result;
    if (!run("?[id, kind, content, summary, embedding, importance, confidence, created_at, accessed_at, access_count, scope, namespace_id, session_id, project_id, turn_id, metadata_json] := *memory_scoped[id, kind, content, summary, embedding, importance, confidence, created_at, accessed_at, access_count, scope, namespace_id, session_id, project_id, turn_id, metadata_json], id == $id",
            params.dump(), result, error)) {
        return std::nullopt;
    }
    const auto parsed = json::parse(result, nullptr, false);
    if (!parsed.is_object() || !parsed.contains("rows") || parsed["rows"].empty()) {
        return std::nullopt;
    }
    return record_from_json(parsed["rows"][0]);
}

std::vector<common_memory_record> common_memory_cozo_store::list(const common_memory_query & query, std::string & error) {
    std::string result;
    if (!run("?[id, kind, content, summary, embedding, importance, confidence, created_at, accessed_at, access_count, scope, namespace_id, session_id, project_id, turn_id, metadata_json] := *memory_scoped[id, kind, content, summary, embedding, importance, confidence, created_at, accessed_at, access_count, scope, namespace_id, session_id, project_id, turn_id, metadata_json]", "{}", result, error)) return {};
    const auto parsed = json::parse(result, nullptr, false);
    std::vector<common_memory_record> records;
    if (!parsed.is_object() || !parsed.contains("rows")) { error = "Cozo returned an unexpected result shape"; return {}; }
    for (const auto & row : parsed["rows"]) { auto record = record_from_json(row); if ((!query.kind || record.kind == *query.kind) && common_memory_scope_matches(record, query)) records.push_back(std::move(record)); }
    std::sort(records.begin(), records.end(), [](const auto & a, const auto & b) { return a.id < b.id; });
    error.clear(); return records;
}

std::vector<common_memory_hit> common_memory_cozo_store::search(const common_memory_query & query, std::string & error) {
    std::string result;
    if (!run("?[id, kind, content, summary, embedding, importance, confidence, created_at, accessed_at, access_count, scope, namespace_id, session_id, project_id, turn_id, metadata_json] := *memory_scoped[id, kind, content, summary, embedding, importance, confidence, created_at, accessed_at, access_count, scope, namespace_id, session_id, project_id, turn_id, metadata_json]",
            "{}", result, error)) {
        return {};
    }

    const auto parsed = json::parse(result, nullptr, false);
    if (!parsed.is_object() || !parsed.contains("rows")) {
        error = "Cozo returned an unexpected result shape";
        return {};
    }

    common_memory_in_memory_store candidate_store;
    std::string candidate_error;
    candidate_store.open("", candidate_error);
    for (const auto & row : parsed["rows"]) {
        auto record = record_from_json(row);
        if (query.kind && record.kind != *query.kind) {
            continue;
        }
        if (!common_memory_scope_matches(record, query)) {
            continue;
        }
        candidate_store.put(record, candidate_error);
    }
    auto hits = candidate_store.search(query, error);
    for (auto & hit : hits) {
        hit.provenance = "CozoDB candidate scan with C++ scoring";
    }
    return hits;
}

bool common_memory_cozo_store::relate(const std::string & from, const std::string & relation, const std::string & to, float weight, std::string & error) {
    const json params = {
        {"rows", json::array({json::array({from, relation, to, weight, (int64_t) std::time(nullptr)})})},
    };
    std::string result;
    return run("?[from, relation, to, weight, created_at] <- $rows :put memory_edge { from, relation, to => weight, created_at }",
        params.dump(), result, error);
}

bool common_memory_cozo_store::erase(const std::string & id, std::string & error) {
    const json params = {{"id", id}};
    std::string result;
    if (!run("?[id] <- [[$id]] :delete memory_scoped { id }", params.dump(), result, error)) {
        return false;
    }
    return run("?[from, relation, to] := *memory_edge[from, relation, to, weight, created_at], (from == $id || to == $id) :delete memory_edge { from, relation, to }",
        params.dump(), result, error);
}
