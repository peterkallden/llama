#pragma once

#include "agent/data-store.h"
#include "tools/agent/host/agent-investigation-config.h"

#include <string>
#include <vector>

class agent_investigation_virtual_dataset_source final : public common_agent_virtual_dataset_source {
public:
    explicit agent_investigation_virtual_dataset_source(agent_host_investigation_config config);

    bool register_datasets(common_agent_data_store & store, std::string & error) const;
    bool execute(const common_agent_dataset_descriptor & descriptor,
        const std::string & operation, const std::string & request_json,
        const common_agent_scope & scope, bool allow_network,
        std::string & result_json, std::string & error) override;

private:
    agent_host_investigation_config config_;
};
