#pragma once

#include "agent-runtime-tooling.h"
#include "agent-route-compiler.h"
#include "agent/contracts/agent-request.h"

common_agent_workflow_continuation_provider
make_agent_resource_document_workflow_continuation(
        common_agent_route_candidate route,
        std::vector<common_agent_tool_argument_binding> bindings);
