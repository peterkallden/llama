#pragma once

#include "agent-runtime-turn.h"

#include <functional>

class agent_tool_view;

using common_agent_runtime_host_post_run = std::function<bool(
    const common_agent_result & result,
    std::string & error)>;

struct common_agent_runtime_host_inputs {
    common_agent_runtime_host_mode mode = common_agent_runtime_host_mode::chat;
    common_memory_store & memory_store;
    common_plan_store * plan_store = nullptr;
    common_agent_runtime_turn_request turn_request;
    std::string * current_plan_id = nullptr;
    const std::vector<common_blueprint_candidate> * installed_blueprint_candidates = nullptr;
    const std::vector<common_memory_hit> * memories = nullptr;
    const std::vector<common_chat_tool> & tools;
    bool profile_tools_active = false;
    agent_tool_view * tool_view = nullptr;
    const common_tool_registry * tool_registry = nullptr;
    common_agent_chat_tool_handler tool_handler;
    bool reset_session_on_completion = false;
    common_agent_runtime_host_post_run post_run;
};

struct common_agent_runtime_host_execution {
    common_agent_runtime_host_mode mode = common_agent_runtime_host_mode::chat;
    common_memory_store & memory_store;
    common_plan_store * plan_store = nullptr;
    common_agent_inference & inference;
    common_agent_runtime_turn_request turn_request;
    std::string * current_plan_id = nullptr;
    const std::vector<common_blueprint_candidate> * installed_blueprint_candidates = nullptr;
    const std::vector<common_memory_hit> * memories = nullptr;
    const std::vector<common_chat_tool> & tools;
    bool profile_tools_active = false;
    agent_tool_view * tool_view = nullptr;
    const common_tool_registry * tool_registry = nullptr;
    common_agent_chat_tool_handler tool_handler;
};

struct common_agent_runtime_host_build_context {
    common_memory_store & memory_store;
    common_plan_store * plan_store = nullptr;
    common_agent_runtime_turn_request turn_request;
    std::string * current_plan_id = nullptr;
    const std::vector<common_blueprint_candidate> * installed_blueprint_candidates = nullptr;
    const std::vector<common_memory_hit> & memories;
    const std::vector<common_chat_tool> & tools;
    bool profile_tools_active = false;
    agent_tool_view * tool_view = nullptr;
    const common_tool_registry * tool_registry = nullptr;
    common_agent_chat_tool_handler tool_handler;
};

common_agent_runtime_host_inputs make_agent_runtime_host_chat_inputs(
    common_agent_runtime_host_build_context & context);

common_agent_runtime_host_inputs make_agent_runtime_host_mini_inputs(
    common_agent_runtime_host_build_context & context,
    const common_agent_orchestration_config & orchestration_config);

common_agent_runtime_host_execution make_agent_runtime_host_execution(
    common_agent_runtime_host_inputs & inputs,
    common_agent_inference & inference);
