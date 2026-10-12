#pragma once

#include "agent-openapi-provider.h"

// Creates the default bounded HTTP executor for an OpenAPI provider. Network
// access remains host-controlled through agent_tool_context::allow_network.
agent_openapi_executor make_agent_openapi_http_executor(
    agent_host_openapi_provider_config config);
