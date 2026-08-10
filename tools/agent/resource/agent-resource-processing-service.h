#pragma once

#include "resource/resource-contract.h"
#include "agent/agent-contract.h"

#include <string>

struct agent_resource_processing_service_request {
    std::string source_uri;
    agent_resource_read_authority authority;
    common_runtime_resource_media_type media_type;
    std::string target_representation = "text";
    std::optional<size_t> page;
    std::optional<agent_resource_byte_range> range;
    agent_resource_processing_limits limits;
    common_agent_event_sink event_sink;
};

class agent_resource_processing_service final : public agent_resource_processing_provider {
public:
    agent_resource_processing_service(
        agent_resource_store & store,
        const agent_resource_processor_registry & registry);

    agent_resource_processing_result process(
        const agent_resource_processing_binding_request & request) const override;

    agent_resource_processing_result process(
        const agent_resource_processing_service_request & request) const;

private:
    agent_resource_store & store_;
    const agent_resource_processor_registry & registry_;
};
