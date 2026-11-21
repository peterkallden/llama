#pragma once

#include "agent/adaptation/learning-transaction.h"
#include "agent/storage/sqlite/sqlite-database.h"

class common_agent_sqlite_learning_transaction_store final
    : public common_learning_transaction_store {
public:
    ~common_agent_sqlite_learning_transaction_store() override;

    bool open(const std::string & path, std::string & error);
    void close();
    bool append(const common_learning_transaction & transaction, std::string & error) override;
    bool contains_idempotency(const std::string & key, bool & contains, std::string & error) const override;
    std::vector<common_learning_transaction> list(std::string & error) const override;

private:
    bool ensure_schema(std::string & error);
    common_sqlite_database database_;
};
