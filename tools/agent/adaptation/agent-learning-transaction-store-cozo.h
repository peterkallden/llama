#pragma once

#include "agent/adaptation/learning-transaction.h"

#include <cstdint>

class common_agent_cozo_learning_transaction_store final
    : public common_learning_transaction_store {
public:
    ~common_agent_cozo_learning_transaction_store() override;

    bool open(const std::string & path, std::string & error);
    void close();
    bool append(const common_learning_transaction & transaction, std::string & error) override;
    bool contains_idempotency(const std::string & key, bool & contains, std::string & error) const override;
    std::vector<common_learning_transaction> list(std::string & error) const override;

private:
    bool run(const std::string & script, const std::string & params_json,
             std::string & result_json, std::string & error) const;

    int32_t db_id_ = -1;
};
