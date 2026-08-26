#pragma once

#include "../mcp/agent-mcp-auth.h"

#include <nlohmann/json.hpp>

// Bind daemon requests to the authenticated caller's authority. Resource
// commands must use the same namespace/project as run_turn, even when the
// client omits those optional fields or sends a different value.
inline void common_agent_daemon_bind_caller_scope(
        nlohmann::ordered_json & request,
        const agent_mcp_caller_policy & policy) {
    const auto command = request.value("command", std::string());
    const bool resource_command = command == "put_resource" ||
        command == "read_resource" || command == "list_resources";
    if (request.contains("namespace_id") || resource_command || command == "run_turn" || command == "execute_tool") {
        request["namespace_id"] = policy.namespace_id;
    }
    if (request.contains("project_id") || resource_command || command == "run_turn" || command == "execute_tool") {
        request["project_id"] = policy.project_id;
    }
    if (command == "run_turn") {
        if (request.value("session_id", std::string()).empty()) {
            request["session_id"] = policy.caller_id + "-session";
        }
    }
    if (command == "execute_tool" && !policy.tool_profile.empty()) {
        request["tool_profile"] = policy.tool_profile;
    }
}
