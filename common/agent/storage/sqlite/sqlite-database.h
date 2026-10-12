#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

struct sqlite3;
struct sqlite3_stmt;

// Small backend-internal SQLite seam. Store implementations must keep SQLite
// details behind their existing common_*_store contracts.
class common_sqlite_statement;

class common_sqlite_database {
public:
    common_sqlite_database() = default;
    ~common_sqlite_database();

    common_sqlite_database(const common_sqlite_database &) = delete;
    common_sqlite_database & operator=(const common_sqlite_database &) = delete;

    bool open(const std::string & path, std::string & error);
    void close();
    bool is_open() const { return db_ != nullptr; }

    bool execute(const std::string & sql, std::string & error) const;
    bool prepare(const std::string & sql, common_sqlite_statement & statement, std::string & error) const;

    bool begin(std::string & error) const;
    bool commit(std::string & error) const;
    bool rollback(std::string & error) const;

    sqlite3 * native_handle() const { return db_; }

private:
    sqlite3 * db_ = nullptr;
};

class common_sqlite_statement {
public:
    common_sqlite_statement() = default;
    ~common_sqlite_statement();

    common_sqlite_statement(const common_sqlite_statement &) = delete;
    common_sqlite_statement & operator=(const common_sqlite_statement &) = delete;
    common_sqlite_statement(common_sqlite_statement && other) noexcept;
    common_sqlite_statement & operator=(common_sqlite_statement && other) noexcept;

    bool bind_null(int index, std::string & error);
    bool bind_text(int index, const std::string & value, std::string & error);
    bool bind_int64(int index, int64_t value, std::string & error);
    bool bind_double(int index, double value, std::string & error);
    bool bind_blob(int index, const void * data, size_t size, std::string & error);

    // row=true means a row is available; row=false means the statement is done.
    bool step(bool & row, std::string & error);
    bool reset(std::string & error);

    int column_count() const;
    int column_type(int index) const;
    const unsigned char * column_text(int index) const;
    int64_t column_int64(int index) const;
    double column_double(int index) const;
    const void * column_blob(int index) const;
    int column_bytes(int index) const;

private:
    friend class common_sqlite_database;
    explicit common_sqlite_statement(sqlite3_stmt * statement) : statement_(statement) {}
    void close();

    sqlite3_stmt * statement_ = nullptr;
};

class common_sqlite_transaction {
public:
    explicit common_sqlite_transaction(const common_sqlite_database & database) : database_(database) {}
    ~common_sqlite_transaction();

    common_sqlite_transaction(const common_sqlite_transaction &) = delete;
    common_sqlite_transaction & operator=(const common_sqlite_transaction &) = delete;

    bool begin(std::string & error);
    bool commit(std::string & error);
    void rollback();

private:
    const common_sqlite_database & database_;
    bool active_ = false;
};
