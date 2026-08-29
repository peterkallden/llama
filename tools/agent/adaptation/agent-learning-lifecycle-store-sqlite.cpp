#include "agent-learning-lifecycle-store-sqlite.h"

#include <utility>

namespace {

std::string text_column(common_sqlite_statement & statement, int index) {
    const auto * value = statement.column_text(index);
    return value ? reinterpret_cast<const char *>(value) : std::string();
}

} // namespace

common_agent_sqlite_learning_lifecycle_store::~common_agent_sqlite_learning_lifecycle_store() { close(); }

bool common_agent_sqlite_learning_lifecycle_store::open(const std::string & path, std::string & error) {
    close();
    if (path.empty()) { error = "SQLite lifecycle store requires a path"; return false; }
    if (!database_.open(path, error) || !database_.execute("PRAGMA journal_mode = WAL;", error) || !ensure_schema(error)) {
        close();
        return false;
    }
    return true;
}

void common_agent_sqlite_learning_lifecycle_store::close() { database_.close(); }

bool common_agent_sqlite_learning_lifecycle_store::ensure_schema(std::string & error) {
    return database_.execute(
        "CREATE TABLE IF NOT EXISTS agent_learning_lifecycle("
        "event_id TEXT PRIMARY KEY,"
        "idempotency_key TEXT NOT NULL UNIQUE,"
        "subject_id TEXT NOT NULL, kind TEXT NOT NULL, status TEXT NOT NULL,"
        "namespace_id TEXT NOT NULL, project_id TEXT NOT NULL, session_id TEXT NOT NULL,"
        "source_id TEXT NOT NULL, content_hash TEXT NOT NULL, created_at TEXT NOT NULL,"
        "payload_json TEXT NOT NULL);"
        "CREATE INDEX IF NOT EXISTS agent_learning_lifecycle_subject "
        "ON agent_learning_lifecycle(subject_id,created_at,event_id);",
        error);
}

bool common_agent_sqlite_learning_lifecycle_store::append(
        const common_learning_lifecycle_record & record, std::string & error) {
    if (!common_learning_lifecycle_validate(record, 4 * 1024 * 1024, error)) return false;
    bool contains = false;
    if (!contains_idempotency(record.idempotency_key, contains, error)) return false;
    if (contains) {
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
    common_sqlite_statement statement;
    const auto sql = "INSERT INTO agent_learning_lifecycle("
        "event_id,idempotency_key,subject_id,kind,status,namespace_id,project_id,session_id,"
        "source_id,content_hash,created_at,payload_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?);";
    if (!database_.prepare(sql, statement, error) ||
            !statement.bind_text(1, record.event_id, error) ||
            !statement.bind_text(2, record.idempotency_key, error) ||
            !statement.bind_text(3, record.subject_id, error) ||
            !statement.bind_text(4, common_learning_lifecycle_kind_name(record.kind), error) ||
            !statement.bind_text(5, common_learning_lifecycle_status_name(record.status), error) ||
            !statement.bind_text(6, record.namespace_id, error) ||
            !statement.bind_text(7, record.project_id, error) ||
            !statement.bind_text(8, record.session_id, error) ||
            !statement.bind_text(9, record.source_id, error) ||
            !statement.bind_text(10, record.content_hash, error) ||
            !statement.bind_text(11, record.created_at, error) ||
            !statement.bind_text(12, common_learning_lifecycle_to_json(record), error)) return false;
    bool row = false;
    return statement.step(row, error);
}

bool common_agent_sqlite_learning_lifecycle_store::contains_idempotency(
        const std::string & key, bool & contains, std::string & error) const {
    common_sqlite_statement statement;
    if (!database_.prepare("SELECT 1 FROM agent_learning_lifecycle WHERE idempotency_key=? LIMIT 1;", statement, error) ||
            !statement.bind_text(1, key, error)) return false;
    bool row = false;
    if (!statement.step(row, error)) return false;
    contains = row;
    return true;
}

std::vector<common_learning_lifecycle_record> common_agent_sqlite_learning_lifecycle_store::list(std::string & error) const {
    error.clear();
    std::vector<common_learning_lifecycle_record> result;
    common_sqlite_statement statement;
    if (!database_.prepare("SELECT event_id,idempotency_key,subject_id,kind,status,namespace_id,project_id,session_id,source_id,content_hash,created_at,payload_json FROM agent_learning_lifecycle ORDER BY event_id;", statement, error)) return result;
    bool row = false;
    while (statement.step(row, error) && row) {
        common_learning_lifecycle_record record;
        record.event_id = text_column(statement, 0);
        record.idempotency_key = text_column(statement, 1);
        record.subject_id = text_column(statement, 2);
        const auto kind = text_column(statement, 3);
        if (kind == "candidate") record.kind = common_learning_lifecycle_kind::candidate;
        else if (kind == "corpus_revision") record.kind = common_learning_lifecycle_kind::corpus_revision;
        else if (kind == "training_job") record.kind = common_learning_lifecycle_kind::training_job;
        else if (kind == "training_result") record.kind = common_learning_lifecycle_kind::training_result;
        else if (kind == "adapter") record.kind = common_learning_lifecycle_kind::adapter;
        else { error = "SQLite lifecycle row contains unknown kind"; return {}; }
        const auto status = text_column(statement, 4);
        if (status == "observed") record.status = common_learning_lifecycle_status::observed;
        else if (status == "eligible") record.status = common_learning_lifecycle_status::eligible;
        else if (status == "approved") record.status = common_learning_lifecycle_status::approved;
        else if (status == "rejected") record.status = common_learning_lifecycle_status::rejected;
        else if (status == "revoked") record.status = common_learning_lifecycle_status::revoked;
        else if (status == "queued") record.status = common_learning_lifecycle_status::queued;
        else if (status == "running") record.status = common_learning_lifecycle_status::running;
        else if (status == "succeeded") record.status = common_learning_lifecycle_status::succeeded;
        else if (status == "failed") record.status = common_learning_lifecycle_status::failed;
        else if (status == "cancelled") record.status = common_learning_lifecycle_status::cancelled;
        else if (status == "active") record.status = common_learning_lifecycle_status::active;
        else if (status == "retired") record.status = common_learning_lifecycle_status::retired;
        else { error = "SQLite lifecycle row contains unknown status"; return {}; }
        record.namespace_id = text_column(statement, 5);
        record.project_id = text_column(statement, 6);
        record.session_id = text_column(statement, 7);
        record.source_id = text_column(statement, 8);
        record.content_hash = text_column(statement, 9);
        record.created_at = text_column(statement, 10);
        record.payload_json = text_column(statement, 11);
        if (!common_learning_lifecycle_validate(record, 4 * 1024 * 1024, error)) return {};
        result.push_back(std::move(record));
    }
    return result;
}
