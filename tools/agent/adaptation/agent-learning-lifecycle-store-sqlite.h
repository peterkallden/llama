#pragma once

#include "agent/adaptation/lifecycle-store.h"
#include "agent/storage/sqlite/sqlite-database.h"

class common_agent_sqlite_learning_lifecycle_store final
    : public common_learning_lifecycle_store {
public:
    ~common_agent_sqlite_learning_lifecycle_store() override;

    bool open(const std::string & path, std::string & error);
    void close();
    bool append(const common_learning_lifecycle_record & record, std::string & error) override;
    bool contains_idempotency(const std::string & key, bool & contains, std::string & error) const override;
    std::vector<common_learning_lifecycle_record> list(std::string & error) const override;

private:
    bool ensure_schema(std::string & error);
    common_sqlite_database database_;
};
