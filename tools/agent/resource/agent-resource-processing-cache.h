#pragma once

#include "resource/resource-contract.h"

// Builds a deterministic identity for one derived representation request.
// The key deliberately includes the source content identity and typed request
// selectors, so a page/range/representation change cannot reuse stale output.
std::string make_agent_resource_processing_cache_key(
        const agent_resource_descriptor & source,
        const common_runtime_resource_media_type & media_type,
        const agent_resource_processor & processor,
        const std::string & target_representation,
        const std::string & target_media_type,
        agent_resource_processing_purpose purpose,
        const std::optional<size_t> & page,
        const std::optional<agent_resource_byte_range> & range,
        const agent_resource_processing_limits & limits);
