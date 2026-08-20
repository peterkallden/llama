#pragma once

#include "agent/thinking/research/research-contract.h"
#include "agent/workspace-contract.h"

inline common_agent_workspace_context common_agent_workspace_context_from_research(
        const common_agent_research_workspace & workspace) {
    common_agent_workspace_context context;
    context.workspace_id = workspace.workspace_id;
    context.project_id = workspace.scope.project_id;
    context.namespace_id = workspace.scope.namespace_id;
    context.session_id = workspace.scope.session_id;
    context.turn_id = workspace.scope.turn_id.empty() ? workspace.turn_id : workspace.scope.turn_id;
    for (const auto & source : workspace.sources) {
        if (source.resource_ref) context.input_resources.push_back(*source.resource_ref);
    }
    return context;
}
