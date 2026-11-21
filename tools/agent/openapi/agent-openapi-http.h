#pragma once

#include "agent-openapi-provider.h"
#include "../host/agent-investigation-config.h"

// Creates the default bounded HTTP executor for an OpenAPI provider. Network
// access remains host-controlled through agent_tool_context::allow_network.
agent_openapi_executor make_agent_openapi_http_executor(
    agent_host_openapi_provider_config config);

// Investigation searches use the same URL validation, secret references,
// TLS, timeouts and bounded response handling as ordinary OpenAPI providers.
bool agent_investigation_http_search(
    const agent_tool_context & context,
    const agent_host_investigation_source_config & source,
    const std::string & index,
    const nlohmann::ordered_json & query,
    std::string & response_json,
    std::string & error);
