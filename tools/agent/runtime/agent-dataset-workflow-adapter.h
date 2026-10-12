#pragma once

#include "agent-runtime-tooling.h"
#include "agent-route-compiler.h"
#include "agent/contracts/agent-request.h"
#include "agent/learning/blueprint-selector.h"
#include "plan/plan-store.h"

// Host-specific dataset binding and verification for the generic workflow
// materialization seam. Other workflow families register their own adapter.
common_blueprint_instance_materializer make_agent_dataset_workflow_materializer(
        common_plan_store & plan_store,
        const common_agent_runtime_tooling * tooling);

common_agent_workflow_continuation_provider
make_agent_resource_document_workflow_continuation(
        common_agent_route_candidate route,
        std::vector<common_agent_tool_argument_binding> bindings);
