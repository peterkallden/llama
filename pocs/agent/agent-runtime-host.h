#pragma once

#include "agent-runtime-turn.h"

#include <functional>
#include <memory>
#include <string>

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
    const common_tool_registry * tool_registry = nullptr;
    common_agent_chat_tool_handler tool_handler;
};

common_agent_runtime_host_execution make_agent_runtime_host_execution(
    common_agent_runtime_host_inputs & inputs,
    common_agent_inference & inference);

bool run_agent_runtime_host_session(
    common_agent_runtime_host_inputs & inputs,
    common_agent_runtime_session & session,
    common_agent_result & result,
    std::string & error);

bool complete_agent_runtime_host_turn(
    common_agent_runtime_host_inputs & inputs,
    common_agent_runtime_session & session,
    const common_agent_result & result,
    std::string & error);

bool run_agent_runtime_host_turn(
    common_agent_runtime_host_inputs & inputs,
    common_agent_runtime_session & session,
    common_agent_result & result,
    std::string & error);

bool run_agent_runtime_host(
    common_agent_runtime_host_execution & execution,
    common_agent_result & result,
    std::string & error);

class common_agent_runtime_resident_host {
public:
    bool run_turn(
        common_agent_runtime_host_inputs & inputs,
        common_agent_result & result,
        std::string & error);

    void reset();

    const common_agent_runtime_session & session() const { return runtime_session; }
    common_agent_runtime_session & session() { return runtime_session; }

private:
    common_agent_runtime_session runtime_session;
};

common_agent_runtime_turn_request make_agent_runtime_resident_base_turn_request(
    const common_agent_runtime_resident_request_config & config);

common_agent_runtime_turn_request make_agent_runtime_resident_turn_request(
    const common_agent_runtime_turn_request & base_turn_request,
    const std::string & prompt,
    const std::string & turn_id);

struct common_agent_runtime_resident_runtime_config {
    common_memory_store & memory_store;
    common_plan_store * plan_store = nullptr;
    common_agent_runtime_turn_request base_turn_request;
    std::string current_plan_id;
    std::vector<common_blueprint_candidate> installed_blueprint_candidates;
    std::vector<common_chat_tool> tools;
    bool profile_tools_active = false;
    const common_tool_registry * tool_registry = nullptr;
};

common_agent_runtime_resident_runtime_config make_agent_runtime_resident_runtime_config(
    common_memory_store & memory_store,
    common_plan_store * plan_store,
    common_agent_runtime_turn_request base_turn_request,
    std::string current_plan_id = {},
    std::vector<common_blueprint_candidate> installed_blueprint_candidates = {},
    std::vector<common_chat_tool> tools = {},
    bool profile_tools_active = false,
    const common_tool_registry * tool_registry = nullptr);

class common_agent_runtime_resident_runtime {
public:
    explicit common_agent_runtime_resident_runtime(common_agent_runtime_resident_runtime_config config);

    bool run_chat_prompt(
        const std::string & prompt,
        const std::string & turn_id,
        common_agent_result & result,
        std::string & error);

    bool run_mini_prompt(
        const std::string & prompt,
        const std::string & turn_id,
        common_agent_result & result,
        std::string & error);

    void reset();

    const std::string & current_plan_id() const { return resident_current_plan_id; }
    const common_agent_runtime_resident_host & runtime_host() const { return host; }
    common_agent_runtime_resident_host & runtime_host() { return host; }

private:
    common_memory_store & memory_store;
    common_plan_store * plan_store = nullptr;
    common_agent_runtime_turn_request base_turn_request;
    std::string resident_current_plan_id;
    std::vector<common_blueprint_candidate> installed_blueprint_candidates;
    std::vector<common_chat_tool> tools;
    bool profile_tools_active = false;
    const common_tool_registry * tool_registry = nullptr;
    common_agent_runtime_resident_host host;
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
    const common_tool_registry * tool_registry = nullptr;
    common_agent_chat_tool_handler tool_handler;
};

common_agent_runtime_host_inputs make_agent_runtime_host_chat_inputs(
    common_agent_runtime_host_build_context & context);

common_agent_runtime_host_inputs make_agent_runtime_host_mini_inputs(
    common_agent_runtime_host_build_context & context,
    const common_agent_orchestration_config & orchestration_config);
