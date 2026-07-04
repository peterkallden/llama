#pragma once

#include "chat.h"

#include <vector>

class agent_tool_view;

struct common_agent_runtime_tooling {
    std::vector<common_chat_tool> tools;
    bool profile_tools_active = false;
    agent_tool_view * tool_view = nullptr;
};
