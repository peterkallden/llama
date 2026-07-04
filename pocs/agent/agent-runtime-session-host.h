#pragma once

#include "agent-runtime-turn.h"
#include "agent-runtime-tooling.h"

#include <memory>
#include <string>
#include <vector>

class common_agent_runtime_resident_runtime;

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
    bool cancelled = false;
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
    common_agent_runtime_tooling tooling;
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
    common_agent_runtime_tooling tooling;
};

common_agent_runtime_session_host_config make_agent_runtime_session_host_config(
    common_agent_runtime_session_host_build_config config);

class common_agent_runtime_session_host {
public:
    explicit common_agent_runtime_session_host(common_agent_runtime_session_host_config config);
    ~common_agent_runtime_session_host();

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
    std::unique_ptr<common_agent_runtime_resident_runtime> runtime;
    uint64_t generated_turn_counter = 0;
};
