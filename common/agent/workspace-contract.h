#pragma once

#include "agent/agent-scope.h"
#include "resource/resource-contract.h"
#include "runtime/runtime-state.h"

#include <string>
#include <vector>

// Shared identity carried by research, developer and data-analysis work.
// The workspace kind-specific state remains owned by its specialized layer.
struct common_agent_workspace_context {
    std::string workspace_id;
    std::string project_id;
    std::string namespace_id;
    std::string session_id;
    std::string turn_id;
    std::vector<common_runtime_resource_ref> input_resources;
};

inline common_agent_state_descriptor describe_common_agent_workspace_context(
        const common_agent_workspace_context & context) {
    common_agent_state_descriptor descriptor;
    descriptor.state_id = context.workspace_id;
    descriptor.state_type = "agent_workspace";
    descriptor.state_class = common_agent_state_class::turn_workspace;
    descriptor.lifetime = common_agent_state_lifetime::turn;
    descriptor.persistence = common_agent_state_persistence::none;
    descriptor.identity.namespace_id = context.namespace_id;
    descriptor.identity.project_id = context.project_id;
    descriptor.identity.session_id = context.session_id;
    descriptor.identity.turn_id = context.turn_id;
    descriptor.owner = "agent workspace host";
    descriptor.source_of_truth = "workspace context";
    return descriptor;
}

inline bool validate_common_agent_workspace_context(
        const common_agent_workspace_context & context,
        std::string & error) {
    if (context.workspace_id.empty() || context.namespace_id.empty() ||
            context.session_id.empty() || context.turn_id.empty()) {
        error = "workspace context requires workspace, namespace, session and turn identities";
        return false;
    }
    for (const auto & resource : context.input_resources) {
        if (resource.uri.empty()) {
            error = "workspace input resource requires a URI";
            return false;
        }
    }
    error.clear();
    return true;
}

struct common_agent_workspace_roots {
    std::string workspace_root;
    std::string artifact_root;
    std::string operation_mode = "ephemeral";
    std::string project_mode = "persistent";
};

struct common_agent_workspace_operation {
    common_agent_workspace_context context;
    std::string operation_id;
    std::string operation_root;
    std::string source_path;
    std::string writable_path;
    std::string artifact_path;
};
