#include "agent-learning-transaction-store-sqlite.h"

namespace {

std::string text_column(common_sqlite_statement & statement, int index) {
    const auto * value = statement.column_text(index);
    return value ? reinterpret_cast<const char *>(value) : std::string();
}

} // namespace

common_agent_sqlite_learning_transaction_store::~common_agent_sqlite_learning_transaction_store() { close(); }

bool common_agent_sqlite_learning_transaction_store::open(const std::string & path, std::string & error) {
    close();
    if (path.empty()) { error = "SQLite learning transaction store requires a path"; return false; }
    if (!database_.open(path, error) || !database_.execute("PRAGMA journal_mode = WAL;", error) || !ensure_schema(error)) {
        close();
        return false;
    }
    return true;
}

void common_agent_sqlite_learning_transaction_store::close() { database_.close(); }

bool common_agent_sqlite_learning_transaction_store::ensure_schema(std::string & error) {
    return database_.execute(
        "CREATE TABLE IF NOT EXISTS agent_learning_transactions("
        "transaction_id TEXT PRIMARY KEY,"
        "idempotency_key TEXT NOT NULL UNIQUE,"
        "namespace_id TEXT NOT NULL, session_id TEXT NOT NULL,"
        "project_id TEXT NOT NULL, turn_id TEXT NOT NULL,"
        "created_at TEXT NOT NULL, transaction_json TEXT NOT NULL);"
        "CREATE INDEX IF NOT EXISTS agent_learning_transactions_scope "
        "ON agent_learning_transactions(namespace_id,session_id,project_id,turn_id);",
        error);
}

bool common_agent_sqlite_learning_transaction_store::append(
        const common_learning_transaction & transaction, std::string & error) {
    error.clear();
    bool exists = false;
    if (!contains_idempotency(transaction.observation.idempotency_key, exists, error)) return false;
    if (exists) return true;
    common_sqlite_statement statement;
    const auto sql = "INSERT INTO agent_learning_transactions("
        "transaction_id,idempotency_key,namespace_id,session_id,project_id,turn_id,created_at,transaction_json) "
        "VALUES(?,?,?,?,?,?,?,?);";
    if (!database_.prepare(sql, statement, error) ||
            !statement.bind_text(1, transaction.id, error) ||
            !statement.bind_text(2, transaction.observation.idempotency_key, error) ||
            !statement.bind_text(3, transaction.observation.scope.namespace_id, error) ||
            !statement.bind_text(4, transaction.observation.scope.session_id, error) ||
            !statement.bind_text(5, transaction.observation.scope.project_id, error) ||
            !statement.bind_text(6, transaction.observation.scope.turn_id, error) ||
            !statement.bind_text(7, transaction.created_at, error) ||
            !statement.bind_text(8, common_learning_transaction_to_json(transaction), error)) return false;
    bool row = false;
    return statement.step(row, error);
}

bool common_agent_sqlite_learning_transaction_store::contains_idempotency(
        const std::string & key, bool & contains, std::string & error) const {
    common_sqlite_statement statement;
    if (!database_.prepare("SELECT 1 FROM agent_learning_transactions WHERE idempotency_key=? LIMIT 1;", statement, error) ||
            !statement.bind_text(1, key, error)) return false;
    bool row = false;
    if (!statement.step(row, error)) return false;
    contains = row;
    return true;
}

std::vector<common_learning_transaction> common_agent_sqlite_learning_transaction_store::list(std::string & error) const {
    error.clear();
    std::vector<common_learning_transaction> result;
    common_sqlite_statement statement;
    if (!database_.prepare("SELECT transaction_json FROM agent_learning_transactions ORDER BY transaction_id;", statement, error)) return result;
    bool row = false;
    while (statement.step(row, error) && row) {
        common_learning_transaction transaction;
        if (!common_learning_transaction_from_json(text_column(statement, 0), transaction, error)) return {};
        result.push_back(std::move(transaction));
    }
    return result;
}
