#pragma once

#include "chat.h"
#include "llama.h"
#include "common/cli-config.h"

#include <string>
#include <vector>

bool generate_chat_turn(
    llama_model * model,
    const common_chat_templates * chat_templates,
    const std::vector<common_chat_msg> & messages,
    const std::vector<common_chat_tool> & tools,
    common_chat_tool_choice tool_choice,
    const args & a,
    std::string & output,
    common_chat_params & chat_params,
    int & n_decode,
    const std::string & json_schema = {});
