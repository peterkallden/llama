#pragma once

#include "../runtime/agent-runtime-assembly.h"

#include <string>
#include <vector>

std::string make_agent_cli_generation_trace_id(
    const common_agent_request & request,
    common_agent_generation_purpose purpose);

std::string describe_agent_cli_generation_failure(
    const char * label,
    const common_agent_generation_result & result);

common_agent_generation_options make_agent_cli_generation_options(
    const common_agent_generation_config & generation_config,
    int n_predict);

common_agent_generation_request make_agent_cli_generation_request(
    const common_agent_request & request,
    common_agent_generation_purpose purpose,
    std::vector<common_chat_msg> messages,
    common_agent_generation_options options,
    std::string json_schema = {},
    std::vector<common_chat_tool> tools = {},
    common_chat_tool_choice tool_choice = COMMON_CHAT_TOOL_CHOICE_NONE);
