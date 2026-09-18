#pragma once

#include "agent/agent-inference.h"
#include "agent/agent-prepared-generation.h"
#include "agent/adaptation/flydelta/flydelta-contracts.h"
#include "agent/adaptation/flydelta/flydelta-decision-margin.h"
#include "chat.h"
#include "llama.h"

#include <string>
#include <vector>

bool generate_chat_turn_result(
    llama_model * model,
    const common_chat_templates * chat_templates,
    const std::vector<common_chat_msg> & messages,
    const std::vector<common_chat_tool> & tools,
    common_chat_tool_choice tool_choice,
    const common_agent_generation_options & options,
    common_agent_generation_result & result,
    common_chat_params * chat_params = nullptr,
    const std::string & json_schema = {},
    const std::vector<llama_adapter_lora *> & adapters = {},
    const std::vector<float> & adapter_scales = {},
    const common_flydelta_static_overlay & flydelta_overlay = {},
    const std::shared_ptr<const common_flydelta_hidden_state_capture_request> & flydelta_capture = {});

// Scores two bounded model-facing continuations after choice_prefix. The
// alternatives are teacher-forced on fresh contexts with the same prompt and
// overlay. They may be just tool names or complete behavior-specific call
// tails. The result contains total and length-normalized logprob data; it is
// a diagnostic ranking signal and never a host verdict.
bool score_chat_choice_margin(
    llama_model * model,
    const common_chat_templates * chat_templates,
    const std::vector<common_chat_msg> & messages,
    const std::vector<common_chat_tool> & tools,
    common_chat_tool_choice tool_choice,
    const common_agent_generation_options & options,
    const std::string & choice_prefix,
    const std::string & positive_choice,
    const std::string & negative_choice,
    common_flydelta_decision_margin & margin,
    common_chat_params * chat_params = nullptr,
    const std::string & json_schema = {},
    const std::vector<llama_adapter_lora *> & adapters = {},
    const std::vector<float> & adapter_scales = {},
    const common_flydelta_static_overlay & flydelta_overlay = {},
    std::string * error = nullptr);

bool generate_chat_turn(
    llama_model * model,
    const common_chat_templates * chat_templates,
    const std::vector<common_chat_msg> & messages,
    const std::vector<common_chat_tool> & tools,
    common_chat_tool_choice tool_choice,
    const common_agent_generation_options & options,
    std::string & output,
    common_chat_params & chat_params,
    int & n_decode,
    const std::string & json_schema = {});
