#pragma once

#include "chat.h"
#include "agent/dataset-contracts.h"
#include "resource/resource-contract.h"

#include <memory>
#include <map>
#include <vector>

class agent_tool_view;

struct common_agent_runtime_tooling {
    std::vector<common_chat_tool> tools;
    // Host-owned descriptions for model-facing family preflight. This is
    // presentation metadata only and does not grant tools or change names.
    std::map<std::string, std::string> family_descriptions;
    // Host-resolved semantic capabilities used for pre-selection eligibility.
    std::vector<std::string> capabilities;
    // Host-resolved hard constraints that are unavailable for the active turn.
    std::vector<std::string> blocked_constraint_ids;
    bool profile_tools_active = false;
    agent_tool_view * tool_view = nullptr;
    std::vector<std::shared_ptr<void>> owned_resources;
    // Host-owned resource store used by controller-side bounded chunk planning.
    // Tool execution remains bound through the resolved tool view.
    agent_resource_runtime resource_runtime;
    // Host-owned dataset inventory captured from the active scoped data store.
    // The runtime copies this into each turn request; it is never a model alias
    // unless the planner explicitly uses the documented host inventory form.
    std::vector<common_agent_dataset_descriptor> available_datasets;
};
