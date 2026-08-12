#pragma once

#include "agent/data-store.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <string>
#include <vector>

// Cozo-backed structured data store. Rows are kept in one host-owned relation
// with JSON payloads so the semantic tool layer is independent of relation
// layout and can later use another backend without changing its contract.
class common_agent_cozo_data_store final : public common_agent_data_store {
public:
    common_agent_cozo_data_store() = default;
    ~common_agent_cozo_data_store() override;

    bool open(const std::string & path, std::string & error);
    void close();
    bool put_row(const std::string & dataset, const std::string & row_id, const std::string & row_json, std::string & error) override;
    bool put_dataset_descriptor(const common_agent_dataset_descriptor & descriptor, std::string & error) override;
    bool get_dataset_descriptor(const std::string & dataset_uri, common_agent_dataset_descriptor & descriptor, std::string & error) override;
    bool execute(const std::string & operation, const std::string & request_json, std::string & result_json, std::string & error) override;

private:
    int32_t db_id_ = -1;
    bool run(const std::string & script, const std::string & params_json, std::string & result_json, std::string & error) const;
};
