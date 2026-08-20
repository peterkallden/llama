#pragma once

#include "chat.h"
#include "resource/resource-contract.h"

#include <memory>
#include <vector>

class agent_tool_view;

struct common_agent_runtime_tooling {
    std::vector<common_chat_tool> tools;
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
};
