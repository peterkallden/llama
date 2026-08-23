#pragma once

#include "memory/memory-store.h"

#include "agent/storage/sqlite/sqlite-database.h"

class common_memory_sqlite_store final : public common_memory_store {
public:
    common_memory_sqlite_store() = default;
    ~common_memory_sqlite_store() override;

    bool open(const std::string & path, std::string & error) override;
    void close() override;

    bool put(const common_memory_record & record, std::string & error) override;
    std::optional<common_memory_record> get(const std::string & id, std::string & error) override;
    std::vector<common_memory_record> list(const common_memory_query & query, std::string & error) override;
    std::vector<common_memory_hit> search(const common_memory_query & query, std::string & error) override;
    bool relate(const std::string & from, const std::string & relation, const std::string & to, float weight, std::string & error) override;
    bool erase(const std::string & id, std::string & error) override;

private:
    bool ensure_schema(std::string & error);
    bool read_record(common_sqlite_statement & statement, common_memory_record & record, std::string & error) const;

    common_sqlite_database database_;
};
