#include "memory-sqlite.h"

#include "memory/memory-in-memory.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <sstream>

using json = nlohmann::ordered_json;

namespace {

std::string text_column(common_sqlite_statement & statement, int index) {
    const auto * value = statement.column_text(index);
    return value ? reinterpret_cast<const char *>(value) : std::string();
}

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

float text_score(const std::string & query, const common_memory_record & record) {
    if (query.empty()) return 0.0f;
    const std::string hay = lowercase(record.content + " " + record.summary);
    std::istringstream terms(lowercase(query));
    std::string term;
    size_t total = 0;
    size_t matches = 0;
    while (terms >> term) {
        ++total;
        if (hay.find(term) != std::string::npos) ++matches;
    }
    return total == 0 ? 0.0f : static_cast<float>(matches) / static_cast<float>(total);
}

std::vector<uint8_t> encode_embedding(const std::vector<float> & values) {
    std::vector<uint8_t> result(values.size() * sizeof(float));
    if (!result.empty()) std::memcpy(result.data(), values.data(), result.size());
    return result;
}

std::vector<float> decode_embedding(const void * data, int bytes) {
    if (!data || bytes <= 0 || bytes % static_cast<int>(sizeof(float)) != 0) return {};
    std::vector<float> result(static_cast<size_t>(bytes) / sizeof(float));
    std::memcpy(result.data(), data, static_cast<size_t>(bytes));
    return result;
}

json metadata_json(const std::unordered_map<std::string, std::string> & metadata) {
    json result = json::object();
    for (const auto & entry : metadata) result[entry.first] = entry.second;
    return result;
}

}

common_memory_sqlite_store::~common_memory_sqlite_store() { close(); }

bool common_memory_sqlite_store::open(const std::string & path, std::string & error) {
    close();
    if (!database_.open(path, error)) return false;
    if (!database_.execute("PRAGMA journal_mode = WAL;", error) || !ensure_schema(error)) {
        close();
        return false;
    }
    return true;
}

void common_memory_sqlite_store::close() { database_.close(); }

bool common_memory_sqlite_store::ensure_schema(std::string & error) {
    return database_.execute(R"SQL(
        CREATE TABLE IF NOT EXISTS agent_memory (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            content TEXT NOT NULL,
            summary TEXT NOT NULL,
            embedding BLOB,
            importance REAL NOT NULL,
            confidence REAL NOT NULL,
            created_at INTEGER NOT NULL,
            accessed_at INTEGER NOT NULL,
            access_count INTEGER NOT NULL,
            scope TEXT NOT NULL,
            namespace_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            project_id TEXT NOT NULL,
            turn_id TEXT NOT NULL,
            metadata_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS agent_memory_edge (
            from_id TEXT NOT NULL,
            relation TEXT NOT NULL,
            to_id TEXT NOT NULL,
            weight REAL NOT NULL,
            created_at INTEGER NOT NULL,
            PRIMARY KEY (from_id, relation, to_id),
            FOREIGN KEY (from_id) REFERENCES agent_memory(id) ON DELETE CASCADE,
            FOREIGN KEY (to_id) REFERENCES agent_memory(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS agent_memory_scope_idx
            ON agent_memory(scope, namespace_id, session_id, project_id, turn_id);
    )SQL", error);
}

bool common_memory_sqlite_store::put(const common_memory_record & record, std::string & error) {
    if (!database_.is_open()) { error = "memory store is not open"; return false; }
    if (record.id.empty()) { error = "memory id must not be empty"; return false; }

    common_sqlite_statement statement;
    if (!database_.prepare(R"SQL(
        INSERT INTO agent_memory
        (id, kind, content, summary, embedding, importance, confidence,
         created_at, accessed_at, access_count, scope, namespace_id,
         session_id, project_id, turn_id, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          kind=excluded.kind, content=excluded.content, summary=excluded.summary,
          embedding=excluded.embedding, importance=excluded.importance,
          confidence=excluded.confidence, created_at=excluded.created_at,
          accessed_at=excluded.accessed_at, access_count=excluded.access_count,
          scope=excluded.scope, namespace_id=excluded.namespace_id,
          session_id=excluded.session_id, project_id=excluded.project_id,
          turn_id=excluded.turn_id, metadata_json=excluded.metadata_json;
    )SQL", statement, error)) return false;

    const auto embedding = encode_embedding(record.embedding);
    const std::string metadata = metadata_json(record.metadata).dump();
    if (!statement.bind_text(1, record.id, error) ||
        !statement.bind_text(2, common_memory_kind_name(record.kind), error) ||
        !statement.bind_text(3, record.content, error) ||
        !statement.bind_text(4, record.summary, error) ||
        !statement.bind_blob(5, embedding.data(), embedding.size(), error) ||
        !statement.bind_double(6, record.importance, error) ||
        !statement.bind_double(7, record.confidence, error) ||
        !statement.bind_int64(8, record.created_at, error) ||
        !statement.bind_int64(9, record.accessed_at, error) ||
        !statement.bind_int64(10, static_cast<int64_t>(record.access_count), error) ||
        !statement.bind_text(11, common_memory_scope_name(record.scope), error) ||
        !statement.bind_text(12, record.namespace_id, error) ||
        !statement.bind_text(13, record.session_id, error) ||
        !statement.bind_text(14, record.project_id, error) ||
        !statement.bind_text(15, record.turn_id, error) ||
        !statement.bind_text(16, metadata, error)) return false;
    bool row = false;
    return statement.step(row, error);
}

bool common_memory_sqlite_store::read_record(common_sqlite_statement & statement, common_memory_record & record, std::string & error) const {
    record = {};
    record.id = text_column(statement, 0);
    if (!common_memory_kind_parse(text_column(statement, 1), record.kind)) {
        error = "sqlite memory row has an invalid kind";
        return false;
    }
    record.content = text_column(statement, 2);
    record.summary = text_column(statement, 3);
    record.embedding = decode_embedding(statement.column_blob(4), statement.column_bytes(4));
    record.importance = static_cast<float>(statement.column_double(5));
    record.confidence = static_cast<float>(statement.column_double(6));
    record.created_at = statement.column_int64(7);
    record.accessed_at = statement.column_int64(8);
    record.access_count = static_cast<uint64_t>(statement.column_int64(9));
    if (!common_memory_scope_parse(text_column(statement, 10), record.scope)) {
        error = "sqlite memory row has an invalid scope";
        return false;
    }
    record.namespace_id = text_column(statement, 11);
    record.session_id = text_column(statement, 12);
    record.project_id = text_column(statement, 13);
    record.turn_id = text_column(statement, 14);
    const auto metadata = json::parse(text_column(statement, 15), nullptr, false);
    if (metadata.is_object()) {
        for (auto it = metadata.begin(); it != metadata.end(); ++it) {
            if (it.value().is_string()) record.metadata[it.key()] = it.value().get<std::string>();
        }
    }
    return true;
}

std::optional<common_memory_record> common_memory_sqlite_store::get(const std::string & id, std::string & error) {
    if (!database_.is_open()) { error = "memory store is not open"; return std::nullopt; }
    common_sqlite_statement statement;
    if (!database_.prepare("SELECT id,kind,content,summary,embedding,importance,confidence,created_at,accessed_at,access_count,scope,namespace_id,session_id,project_id,turn_id,metadata_json FROM agent_memory WHERE id = ?;", statement, error) || !statement.bind_text(1, id, error)) return std::nullopt;
    bool row = false;
    if (!statement.step(row, error) || !row) return std::nullopt;
    common_memory_record record;
    return read_record(statement, record, error) ? std::optional<common_memory_record>(std::move(record)) : std::nullopt;
}

std::vector<common_memory_record> common_memory_sqlite_store::list(const common_memory_query & query, std::string & error) {
    std::vector<common_memory_record> result;
    if (!database_.is_open()) { error = "memory store is not open"; return result; }
    common_sqlite_statement statement;
    if (!database_.prepare("SELECT id,kind,content,summary,embedding,importance,confidence,created_at,accessed_at,access_count,scope,namespace_id,session_id,project_id,turn_id,metadata_json FROM agent_memory WHERE scope = ? AND namespace_id = ? AND (session_id = ? OR project_id = ? OR turn_id = ? OR ? = 'global') ORDER BY created_at DESC;", statement, error)) return result;
    if (!statement.bind_text(1, common_memory_scope_name(query.scope), error) || !statement.bind_text(2, query.namespace_id, error) || !statement.bind_text(3, query.session_id, error) || !statement.bind_text(4, query.project_id, error) || !statement.bind_text(5, query.turn_id, error) || !statement.bind_text(6, common_memory_scope_name(query.scope), error)) return result;
    bool row = false;
    while (statement.step(row, error) && row) {
        common_memory_record record;
        if (!read_record(statement, record, error)) return {};
        if (!query.kind || record.kind == *query.kind) result.push_back(std::move(record));
    }
    return error.empty() ? result : std::vector<common_memory_record>{};
}

std::vector<common_memory_hit> common_memory_sqlite_store::search(const common_memory_query & query, std::string & error) {
    std::vector<common_memory_hit> hits;
    for (const auto & record : list(query, error)) {
        const float semantic = !query.embedding.empty() ? common_memory_cosine_similarity(query.embedding, record.embedding) : text_score(query.text, record);
        if (semantic < query.minimum_score) continue;
        common_memory_hit hit;
        hit.memory = record;
        hit.semantic_score = semantic;
        hit.final_score = semantic;
        hit.provenance = "sqlite direct search";
        hits.push_back(std::move(hit));
    }
    std::sort(hits.begin(), hits.end(), [](const auto & a, const auto & b) {
        if (a.final_score != b.final_score) return a.final_score > b.final_score;
        return a.memory.id < b.memory.id;
    });
    if (hits.size() > query.limit) hits.resize(query.limit);
    return hits;
}

bool common_memory_sqlite_store::relate(const std::string & from, const std::string & relation, const std::string & to, float weight, std::string & error) {
    if (!database_.is_open()) { error = "memory store is not open"; return false; }
    if (relation.empty()) { error = "relation must not be empty"; return false; }
    common_sqlite_transaction transaction(database_);
    if (!transaction.begin(error)) return false;
    common_sqlite_statement statement;
    if (!database_.prepare("INSERT INTO agent_memory_edge(from_id,relation,to_id,weight,created_at) VALUES(?,?,?,?,strftime('%s','now')) ON CONFLICT(from_id,relation,to_id) DO UPDATE SET weight=excluded.weight;", statement, error) || !statement.bind_text(1, from, error) || !statement.bind_text(2, relation, error) || !statement.bind_text(3, to, error) || !statement.bind_double(4, weight, error)) return false;
    bool row = false;
    if (!statement.step(row, error) || !transaction.commit(error)) return false;
    return true;
}

bool common_memory_sqlite_store::erase(const std::string & id, std::string & error) {
    if (!database_.is_open()) { error = "memory store is not open"; return false; }
    common_sqlite_statement statement;
    if (!database_.prepare("DELETE FROM agent_memory WHERE id = ?;", statement, error) || !statement.bind_text(1, id, error)) return false;
    bool row = false;
    return statement.step(row, error);
}
