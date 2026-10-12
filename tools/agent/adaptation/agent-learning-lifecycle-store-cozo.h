#pragma once

#include "agent/adaptation/lifecycle-store.h"

#include <cstdint>

class common_agent_cozo_learning_lifecycle_store final
    : public common_learning_lifecycle_store {
public:
    ~common_agent_cozo_learning_lifecycle_store() override;

    bool open(const std::string & path, std::string & error);
    void close();
    bool append(const common_learning_lifecycle_record & record, std::string & error) override;
    bool contains_idempotency(const std::string & key, bool & contains, std::string & error) const override;
    std::vector<common_learning_lifecycle_record> list(std::string & error) const override;

private:
    bool run(const std::string & script, const std::string & params_json,
             std::string & result_json, std::string & error) const;

    int32_t db_id_ = -1;
};
