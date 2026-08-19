#pragma once

#include "agent/tooling/catalog/tool-catalog.h"
#include "agent/tooling/registry/tool-registry.h"
#include "chat.h"

#include <cstddef>
#include <string>
#include <vector>

// Converts only tools that are both declared by the profile and registered by
// native code. This is the model-facing half of the bridge.
bool common_tool_profile_to_chat_tools(
    const common_tool_catalog & catalog,
    const std::string & profile_id,
    const common_tool_registry & registry,
    std::vector<common_chat_tool> & tools,
    std::string & error);

struct common_tool_chat_dispatch_result {
    std::vector<common_chat_msg> tool_messages;
    size_t executed = 0;
};

// Runs a bounded assistant tool-call batch through the native registry and
// produces ordinary `role: tool` messages for the next template application.
// Generated call ids are written back to assistant_message when a template did
// not supply one.
bool common_tool_dispatch_chat_calls(
    common_chat_msg & assistant_message,
    const common_tool_registry & registry,
    size_t max_calls,
    common_tool_chat_dispatch_result & result,
    std::string & error);

