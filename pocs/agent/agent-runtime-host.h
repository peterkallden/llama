#pragma once

#include "../common/cli-config.h"

#include "agent-runtime-chat-driver.h"
#include "agent-runtime-execution.h"

#include <functional>
#include <memory>
#include <string>

enum class common_agent_runtime_host_mode {
    chat,
    mini,
};

using common_agent_runtime_host_post_run = std::function<bool(
    const common_agent_result & result,
    std::string & error)>;

struct common_agent_runtime_turn_request {
    common_agent_request request;
    common_agent_scope scope;
    common_agent_inference_options inference_options;
    common_agent_runtime_policy policy;
    common_agent_runtime_config runtime_config;
    common_agent_orchestration_config orchestration_config;
    common_agent_generation_options generation_options;
    common_memory_scope memory_scope = common_memory_scope::session;
    bool memory_enabled = false;
    std::string fallback_reason;
};

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

struct common_agent_runtime_resident_request_config {
    std::string prompt;
    std::string session_id;
    std::string namespace_id;
    std::string project_id;
    std::string model;
    int n_predict = 0;
    int n_gpu_layers = 0;
    bool fit_params = true;
    std::string inference_backend = "server-context";
    common_memory_scope memory_scope = common_memory_scope::session;
    common_plan_scope plan_scope = common_plan_scope::turn;
};

common_agent_runtime_turn_request make_agent_runtime_resident_base_turn_request(
    const common_agent_runtime_resident_request_config & config);

common_agent_runtime_turn_request make_agent_runtime_resident_turn_request(
    const common_agent_runtime_turn_request & base_turn_request,
    const std::string & prompt,
    const std::string & turn_id);

struct common_agent_runtime_resident_chat_host_config {
    common_memory_store & memory_store;
    common_agent_runtime_turn_request base_turn_request;
};

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

class common_agent_runtime_resident_chat_host {
public:
    explicit common_agent_runtime_resident_chat_host(common_agent_runtime_resident_chat_host_config config);

    bool run_prompt(
        const std::string & prompt,
        const std::string & turn_id,
        common_agent_result & result,
        std::string & error);

    void reset();

    const common_agent_runtime_resident_host & runtime_host() const { return runtime.runtime_host(); }
    common_agent_runtime_resident_host & runtime_host() { return runtime.runtime_host(); }

private:
    common_agent_runtime_resident_runtime runtime;
};

struct common_agent_runtime_session_host_turn_request {
    common_agent_runtime_host_mode mode = common_agent_runtime_host_mode::chat;
    std::string prompt;
    std::string session_id;
    std::string namespace_id;
    std::string project_id;
    std::string turn_id;
    common_memory_scope memory_scope = common_memory_scope::session;
    common_plan_scope plan_scope = common_plan_scope::turn;
    int n_predict = 0;
};

struct common_agent_runtime_session_host_turn_result {
    bool ok = false;
    bool runtime_reused = false;
    bool limit_reached = false;
    bool reflected = false;
    bool revised = false;
    std::string response;
    std::string plan_id;
    int total_decoded_tokens = 0;
    size_t event_count = 0;
    size_t memory_learning_related_count = 0;
    std::string memory_learning_summary;
    std::string error;
};

struct common_agent_runtime_session_host_config {
    common_memory_store & memory_store;
    common_plan_store & plan_store;
    common_agent_runtime_resident_request_config resident_request;
    common_agent_runtime_policy policy;
    common_agent_runtime_config runtime_config;
    common_agent_orchestration_config orchestration_config;
    common_memory_scope memory_scope = common_memory_scope::session;
    bool memory_enabled = false;
    std::vector<common_blueprint_candidate> installed_blueprint_candidates;
    std::vector<common_chat_tool> tools;
    bool profile_tools_active = false;
    const common_tool_registry * tool_registry = nullptr;
};

struct common_agent_runtime_session_host_build_config {
    common_memory_store & memory_store;
    common_plan_store & plan_store;
    common_agent_runtime_resident_request_config resident_request;
    common_agent_runtime_policy policy;
    common_agent_runtime_config runtime_config;
    common_agent_orchestration_config orchestration_config;
    common_memory_scope memory_scope = common_memory_scope::session;
    bool memory_enabled = false;
    std::vector<common_blueprint_candidate> installed_blueprint_candidates;
    std::vector<common_chat_tool> tools;
    bool profile_tools_active = false;
    const common_tool_registry * tool_registry = nullptr;
};

common_agent_runtime_session_host_config make_agent_runtime_session_host_config(
    common_agent_runtime_session_host_build_config config);

class common_agent_runtime_session_host {
public:
    explicit common_agent_runtime_session_host(common_agent_runtime_session_host_config config);

    bool run_turn(
        const common_agent_runtime_session_host_turn_request & request,
        common_agent_runtime_session_host_turn_result & result,
        std::string & error);

    void reset();

    const common_agent_runtime_session * session() const;
    common_agent_runtime_session * session();

private:
    bool ensure_runtime(
        const common_agent_runtime_session_host_turn_request & request,
        bool & reused,
        std::string & error);

    common_agent_runtime_turn_request make_base_turn_request(
        const common_agent_runtime_session_host_turn_request & request) const;

    common_agent_runtime_session_host_config config;
    std::string active_session_id;
    std::string active_namespace_id;
    std::string active_project_id;
    common_memory_scope active_memory_scope = common_memory_scope::session;
    common_plan_scope active_plan_scope = common_plan_scope::turn;
    int active_n_predict = 0;
    std::unique_ptr<common_agent_runtime_resident_runtime> runtime;
    uint64_t generated_turn_counter = 0;
};

using common_agent_runtime_daemon_turn_request = common_agent_runtime_session_host_turn_request;
using common_agent_runtime_daemon_turn_result = common_agent_runtime_session_host_turn_result;
using common_agent_runtime_daemon_config = common_agent_runtime_session_host_config;
using common_agent_runtime_daemon_build_config = common_agent_runtime_session_host_build_config;
using common_agent_runtime_daemon_host = common_agent_runtime_session_host;

inline common_agent_runtime_daemon_config make_agent_runtime_daemon_config(
        common_agent_runtime_daemon_build_config config) {
    return make_agent_runtime_session_host_config(std::move(config));
}

struct common_agent_runtime_resident_mini_host_config {
    common_memory_store & memory_store;
    common_plan_store & plan_store;
    common_agent_runtime_turn_request base_turn_request;
    std::string current_plan_id;
    std::vector<common_blueprint_candidate> installed_blueprint_candidates;
    std::vector<common_chat_tool> tools;
    bool profile_tools_active = false;
    const common_tool_registry * tool_registry = nullptr;
};

class common_agent_runtime_resident_mini_host {
public:
    explicit common_agent_runtime_resident_mini_host(common_agent_runtime_resident_mini_host_config config);

    bool run_prompt(
        const std::string & prompt,
        const std::string & turn_id,
        common_agent_result & result,
        std::string & error);

    void reset();

    const std::string & current_plan_id() const { return runtime.current_plan_id(); }
    const common_agent_runtime_resident_host & runtime_host() const { return runtime.runtime_host(); }
    common_agent_runtime_resident_host & runtime_host() { return runtime.runtime_host(); }

private:
    common_agent_runtime_resident_runtime runtime;
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
