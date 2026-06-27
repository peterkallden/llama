#pragma once

#include "chat.h"
#include "common/cli-config.h"

#include <string>
#include <vector>

struct common_agent_inference_request {
    std::vector<common_chat_msg> messages;
    std::vector<common_chat_tool> tools;
    common_chat_tool_choice tool_choice = COMMON_CHAT_TOOL_CHOICE_NONE;
    args options;
    std::string json_schema;
};

struct common_agent_inference_result {
    std::string output;
    common_chat_params chat_params;
    int n_decode = 0;
};

class common_agent_inference {
public:
    virtual ~common_agent_inference() = default;
    virtual bool infer(
        const common_agent_inference_request & request,
        common_agent_inference_result & result) = 0;
};
