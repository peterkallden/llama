#pragma once

#include "agent/data-store.h"
#include "agent/storage/sqlite/sqlite-database.h"

class common_agent_sqlite_data_store final : public common_agent_data_store {
public:
    ~common_agent_sqlite_data_store() override;

    bool open(const std::string & path, std::string & error);
    void close();
    bool put_row(const std::string & dataset, const std::string & row_id, const std::string & row_json, std::string & error) override;
    bool put_dataset_descriptor(const common_agent_dataset_descriptor & descriptor, std::string & error) override;
    bool get_dataset_descriptor(const std::string & dataset_uri, common_agent_dataset_descriptor & descriptor, std::string & error) override;
    bool list_dataset_descriptors(std::vector<common_agent_dataset_descriptor> & descriptors, std::string & error) override;
    bool find_dataset_by_name(const std::string & name, common_agent_dataset_descriptor & descriptor, std::string & error) override;
    bool execute(const std::string & operation, const std::string & request_json, std::string & result_json, std::string & error) override;

private:
    bool ensure_schema(std::string & error);
    common_sqlite_database database_;
};
