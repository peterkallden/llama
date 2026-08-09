#pragma once

#include "resource/resource-contract.h"

#include <string>

struct agent_resource_media_type_resolution_request {
    std::string declared_type;
    std::string sample_bytes;
    bool allow_text_heuristic = true;
};

common_runtime_resource_media_type resolve_agent_resource_media_type(
    const agent_resource_media_type_resolution_request & request);
