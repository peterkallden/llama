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
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    close();
    if (path.empty()) { error = "SQLite lifecycle store requires a path"; return false; }
    if (!database_.open(path, error) || !database_.execute("PRAGMA journal_mode = WAL;", error) ||
            !database_.execute("PRAGMA synchronous = FULL;", error) || !ensure_schema(error)) {
        close();
        return false;
    }
    return true;
}

void common_agent_sqlite_learning_lifecycle_store::close() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    database_.close();
}

bool common_agent_sqlite_learning_lifecycle_store::ensure_schema(std::string & error) {
    if (!database_.execute(
        "CREATE TABLE IF NOT EXISTS agent_learning_lifecycle("
        "event_id TEXT PRIMARY KEY,"
        "idempotency_key TEXT NOT NULL UNIQUE,"
        "subject_id TEXT NOT NULL, kind TEXT NOT NULL, status TEXT NOT NULL,"
        "namespace_id TEXT NOT NULL, project_id TEXT NOT NULL, session_id TEXT NOT NULL,"
        "source_id TEXT NOT NULL, content_hash TEXT NOT NULL, created_at TEXT NOT NULL,"
        "payload_json TEXT NOT NULL, registration_order INTEGER NOT NULL DEFAULT 0);"
        "CREATE INDEX IF NOT EXISTS agent_learning_lifecycle_subject "
        "ON agent_learning_lifecycle(subject_id,created_at,event_id);",
        error)) return false;
    bool has_order = false;
    common_sqlite_statement columns;
    if (!database_.prepare("PRAGMA table_info(agent_learning_lifecycle);", columns, error)) return false;
    bool row = false;
    while (columns.step(row, error) && row) {
        const auto * name = columns.column_text(1);
        if (name && std::string(reinterpret_cast<const char *>(name)) == "registration_order") has_order = true;
    }
    if (!error.empty()) return false;
    if (!has_order && !database_.execute(
            "ALTER TABLE agent_learning_lifecycle ADD COLUMN registration_order INTEGER NOT NULL DEFAULT 0;",
            error)) return false;
    // SQLite rowid is the durable insertion sequence for this rowid table.
    // It gives pre-upgrade rows an authoritative order without interpreting
    // timestamps or event-id spelling.
    return database_.execute(
        "UPDATE agent_learning_lifecycle SET registration_order=rowid "
        "WHERE registration_order=0;",
        error);
}

bool common_agent_sqlite_learning_lifecycle_store::append(
        const common_learning_lifecycle_record & record, std::string & error) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
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
    common_sqlite_transaction transaction(database_);
    if (!transaction.begin(error)) return false;
    int64_t registration_order = 0;
    {
        common_sqlite_statement maximum;
        if (!database_.prepare(
                "SELECT COALESCE(MAX(registration_order),0)+1 FROM agent_learning_lifecycle;",
                maximum, error)) return false;
        bool has_row = false;
        if (!maximum.step(has_row, error) || !has_row) return false;
        registration_order = maximum.column_int64(0);
    }
    const auto sql = "INSERT INTO agent_learning_lifecycle("
        "event_id,idempotency_key,subject_id,kind,status,namespace_id,project_id,session_id,"
        "source_id,content_hash,created_at,payload_json,registration_order) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?);";
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
            !statement.bind_text(12, record.payload_json, error) ||
            !statement.bind_int64(13, registration_order, error)) return false;
    bool row = false;
    if (!statement.step(row, error)) return false;
    return transaction.commit(error);
}

bool common_agent_sqlite_learning_lifecycle_store::contains_idempotency(
        const std::string & key, bool & contains, std::string & error) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    common_sqlite_statement statement;
    if (!database_.prepare("SELECT 1 FROM agent_learning_lifecycle WHERE idempotency_key=? LIMIT 1;", statement, error) ||
            !statement.bind_text(1, key, error)) return false;
    bool row = false;
    if (!statement.step(row, error)) return false;
    contains = row;
    return true;
}

std::vector<common_learning_lifecycle_record> common_agent_sqlite_learning_lifecycle_store::list(std::string & error) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    error.clear();
    std::vector<common_learning_lifecycle_record> result;
    common_sqlite_statement statement;
    if (!database_.prepare("SELECT event_id,idempotency_key,subject_id,kind,status,namespace_id,project_id,session_id,source_id,content_hash,created_at,payload_json,registration_order FROM agent_learning_lifecycle ORDER BY registration_order,event_id;", statement, error)) return result;
    bool row = false;
    while (statement.step(row, error) && row) {
        common_learning_lifecycle_record record;
        record.schema_version = 2;
        record.event_id = text_column(statement, 0);
        record.idempotency_key = text_column(statement, 1);
        record.subject_id = text_column(statement, 2);
        if (!parse_common_learning_lifecycle_kind(text_column(statement, 3), record.kind, error) ||
                !parse_common_learning_lifecycle_status(text_column(statement, 4), record.status, error)) {
            if (error.find("kind") != std::string::npos) error = "SQLite lifecycle row contains unknown kind";
            else error = "SQLite lifecycle row contains unknown status";
            return {};
        }
        record.namespace_id = text_column(statement, 5);
        record.project_id = text_column(statement, 6);
        record.session_id = text_column(statement, 7);
        record.source_id = text_column(statement, 8);
        record.content_hash = text_column(statement, 9);
        record.created_at = text_column(statement, 10);
        record.payload_json = text_column(statement, 11);
        record.registration_order = static_cast<uint64_t>(statement.column_int64(12));
        if (!common_learning_lifecycle_validate(record, 4 * 1024 * 1024, error)) return {};
        result.push_back(std::move(record));
    }
    return result;
}
