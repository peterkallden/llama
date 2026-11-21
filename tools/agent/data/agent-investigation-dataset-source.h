#pragma once

#include "agent/data-store.h"
#include "tools/agent/host/agent-investigation-config.h"

#include <string>
#include <vector>
#include <cstddef>

bool agent_investigation_search_page_complete(bool total_exact, bool timed_out,
    bool shard_status_known, size_t failed_shards, size_t offset,
    size_t returned_rows, size_t total);

class agent_investigation_virtual_dataset_source final : public common_agent_virtual_dataset_source {
public:
    explicit agent_investigation_virtual_dataset_source(agent_host_investigation_config config);

    bool register_datasets(common_agent_data_store & store, std::string & error) const;
    bool authorize_source(const std::string & source_id,
        const common_agent_scope & scope, std::string & error) const override;
    bool execute(const common_agent_dataset_descriptor & descriptor,
        const std::string & operation, const std::string & request_json,
        const common_agent_scope & scope, bool allow_network,
        std::string & result_json, std::string & error) override;

private:
    agent_host_investigation_config config_;
};
