#pragma once

#include "chat.h"
#include "agent/dataset-contracts.h"
#include "agent/investigation-contracts.h"
#include "resource/resource-contract.h"

#include <memory>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

class agent_tool_view;
struct common_agent_template_library_snapshot;

struct common_agent_runtime_tooling {
    std::vector<common_chat_tool> tools;
    std::vector<common_chat_tool> app_tools;

    // Host presentation includes app-only tools without exposing them to the model.
    const common_chat_tool * find_presented_tool(const std::string & name) const {
        for (const auto * list : {&tools, &app_tools}) {
            for (const auto & tool : *list) if (tool.name == name) return &tool;
        }
        return nullptr;
    }

    // Host-owned descriptions for model-facing family preflight. This is
    // presentation metadata only and does not grant tools or change names.
    std::map<std::string, std::string> family_descriptions;
    std::map<std::string, std::vector<std::string>> additional_tool_families;
    // Host-resolved semantic capabilities used for pre-selection eligibility.
    std::vector<std::string> capabilities;
    // Host facts that workflows may require for materialization. Unlike
    // capabilities, these do not resolve to tools or grant execution rights.
    std::vector<std::string> available_context;
    // Host-resolved capability-to-tool bindings. Workflow definitions request
    // semantic capabilities; only tools in this active view can satisfy them.
    std::map<std::string, std::vector<std::string>> capability_tools;
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
    std::function<bool(const std::string &, nlohmann::ordered_json &, std::string &)> grounding_dataset_reader;
    // Sanitized host projection of a typed investigation binding. This is
    // request-scoped and contains only case metadata and selected evidence.
    std::optional<std::string> investigation_context_text;
    std::optional<common_investigation_context_binding> investigation_context;
    std::optional<std::vector<common_investigation_source_kind>> investigation_source_kinds;
    std::shared_ptr<const common_agent_template_library_snapshot> template_libraries;
};
