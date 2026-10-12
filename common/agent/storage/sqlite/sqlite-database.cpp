#include "sqlite-database.h"

#include <sqlite3.h>

#include <utility>

namespace {

std::string sqlite_error(sqlite3 * db, int code, const char * operation) {
    const char * message = db ? sqlite3_errmsg(db) : sqlite3_errstr(code);
    return std::string(operation) + ": " + (message ? message : "unknown SQLite error");
}

bool check_bind(sqlite3_stmt * statement, int code, const char * operation, std::string & error) {
    if (code == SQLITE_OK) return true;
    sqlite3 * db = statement ? sqlite3_db_handle(statement) : nullptr;
    error = sqlite_error(db, code, operation);
    return false;
}

}

common_sqlite_database::~common_sqlite_database() { close(); }

bool common_sqlite_database::open(const std::string & path, std::string & error) {
    close();
    const std::string actual_path = path.empty() ? ":memory:" : path;
    const int code = sqlite3_open_v2(actual_path.c_str(), &db_, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nullptr);
    if (code != SQLITE_OK) {
        error = sqlite_error(db_, code, "sqlite open");
        close();
        return false;
    }
    sqlite3_busy_timeout(db_, 5000);
    if (!execute("PRAGMA foreign_keys = ON;", error)) {
        close();
        return false;
    }
    return true;
}

void common_sqlite_database::close() {
    if (db_) {
        sqlite3_close(db_);
        db_ = nullptr;
    }
}

bool common_sqlite_database::execute(const std::string & sql, std::string & error) const {
    if (!db_) {
        error = "sqlite execute: database is not open";
        return false;
    }
    char * message = nullptr;
    const int code = sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &message);
    if (code != SQLITE_OK) {
        error = "sqlite execute: ";
        error += message ? message : sqlite3_errmsg(db_);
        sqlite3_free(message);
        return false;
    }
    return true;
}

bool common_sqlite_database::prepare(const std::string & sql, common_sqlite_statement & statement, std::string & error) const {
    if (!db_) {
        error = "sqlite prepare: database is not open";
        return false;
    }
    statement.close();
    sqlite3_stmt * prepared = nullptr;
    const int code = sqlite3_prepare_v2(db_, sql.c_str(), -1, &prepared, nullptr);
    if (code != SQLITE_OK) {
        error = sqlite_error(db_, code, "sqlite prepare");
        return false;
    }
    statement.statement_ = prepared;
    return true;
}

bool common_sqlite_database::begin(std::string & error) const { return execute("BEGIN IMMEDIATE;", error); }
bool common_sqlite_database::commit(std::string & error) const { return execute("COMMIT;", error); }
bool common_sqlite_database::rollback(std::string & error) const { return execute("ROLLBACK;", error); }

common_sqlite_statement::~common_sqlite_statement() { close(); }

common_sqlite_statement::common_sqlite_statement(common_sqlite_statement && other) noexcept : statement_(other.statement_) {
    other.statement_ = nullptr;
}

common_sqlite_statement & common_sqlite_statement::operator=(common_sqlite_statement && other) noexcept {
    if (this != &other) {
        close();
        statement_ = other.statement_;
        other.statement_ = nullptr;
    }
    return *this;
}

void common_sqlite_statement::close() {
    if (statement_) {
        sqlite3_finalize(statement_);
        statement_ = nullptr;
    }
}

bool common_sqlite_statement::bind_null(int index, std::string & error) {
    return check_bind(statement_, sqlite3_bind_null(statement_, index), "sqlite bind null", error);
}

bool common_sqlite_statement::bind_text(int index, const std::string & value, std::string & error) {
    return check_bind(statement_, sqlite3_bind_text(statement_, index, value.data(), static_cast<int>(value.size()), SQLITE_TRANSIENT), "sqlite bind text", error);
}

bool common_sqlite_statement::bind_int64(int index, int64_t value, std::string & error) {
    return check_bind(statement_, sqlite3_bind_int64(statement_, index, value), "sqlite bind int64", error);
}

bool common_sqlite_statement::bind_double(int index, double value, std::string & error) {
    return check_bind(statement_, sqlite3_bind_double(statement_, index, value), "sqlite bind double", error);
}

bool common_sqlite_statement::bind_blob(int index, const void * data, size_t size, std::string & error) {
    return check_bind(statement_, sqlite3_bind_blob(statement_, index, data, static_cast<int>(size), SQLITE_TRANSIENT), "sqlite bind blob", error);
}

bool common_sqlite_statement::step(bool & row, std::string & error) {
    const int code = sqlite3_step(statement_);
    if (code == SQLITE_ROW) {
        row = true;
        return true;
    }
    if (code == SQLITE_DONE) {
        row = false;
        return true;
    }
    row = false;
    error = sqlite_error(sqlite3_db_handle(statement_), code, "sqlite step");
    return false;
}

bool common_sqlite_statement::reset(std::string & error) {
    const int code = sqlite3_reset(statement_);
    if (code != SQLITE_OK) {
        error = sqlite_error(sqlite3_db_handle(statement_), code, "sqlite reset");
        return false;
    }
    sqlite3_clear_bindings(statement_);
    return true;
}

int common_sqlite_statement::column_count() const { return sqlite3_column_count(statement_); }
int common_sqlite_statement::column_type(int index) const { return sqlite3_column_type(statement_, index); }
const unsigned char * common_sqlite_statement::column_text(int index) const { return sqlite3_column_text(statement_, index); }
int64_t common_sqlite_statement::column_int64(int index) const { return sqlite3_column_int64(statement_, index); }
double common_sqlite_statement::column_double(int index) const { return sqlite3_column_double(statement_, index); }
const void * common_sqlite_statement::column_blob(int index) const { return sqlite3_column_blob(statement_, index); }
int common_sqlite_statement::column_bytes(int index) const { return sqlite3_column_bytes(statement_, index); }

common_sqlite_transaction::~common_sqlite_transaction() { rollback(); }

bool common_sqlite_transaction::begin(std::string & error) {
    if (active_) return true;
    active_ = database_.begin(error);
    return active_;
}

bool common_sqlite_transaction::commit(std::string & error) {
    if (!active_) return true;
    if (!database_.commit(error)) return false;
    active_ = false;
    return true;
}

void common_sqlite_transaction::rollback() {
    if (active_) {
        std::string ignored;
        database_.rollback(ignored);
        active_ = false;
    }
}
