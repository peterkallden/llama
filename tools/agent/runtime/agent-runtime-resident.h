#pragma once

#include "../runtime/agent-runtime-host-contracts.h"
#include "../runtime/agent-runtime-tooling.h"
#include "../runtime/agent-runtime-turn.h"

#include <string>
#include <vector>

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
    const std::string & turn_id,
    int n_predict_override = 0);

struct common_agent_runtime_resident_runtime_config {
    common_memory_store & memory_store;
    common_plan_store * plan_store = nullptr;
    common_agent_runtime_turn_request base_turn_request;
    std::string current_plan_id;
    std::vector<common_blueprint_candidate> installed_blueprint_candidates;
    common_agent_runtime_tooling tooling;
};

common_agent_runtime_resident_runtime_config make_agent_runtime_resident_runtime_config(
    common_memory_store & memory_store,
    common_plan_store * plan_store,
    common_agent_runtime_turn_request base_turn_request,
    std::string current_plan_id = {},
    std::vector<common_blueprint_candidate> installed_blueprint_candidates = {},
    common_agent_runtime_tooling tooling = {});

class common_agent_runtime_resident_runtime {
public:
    explicit common_agent_runtime_resident_runtime(common_agent_runtime_resident_runtime_config config);

    void set_tooling(common_agent_runtime_tooling tooling);
    void set_execution_control(common_agent_runtime_execution_control execution_control);
    void set_policy_pack(std::optional<common_memory_policy_pack> policy_pack);

    // Load the configured inference model without executing a user turn.
    bool prepare_model(std::string & error);

    bool run_chat_prompt(
        const std::string & prompt,
        const std::string & turn_id,
        int n_predict,
        common_agent_result & result,
        std::string & error);

    bool run_agent_prompt(
        const std::string & prompt,
        const std::string & turn_id,
        int n_predict,
        common_agent_result & result,
        std::string & error);

    void reset();

    const std::string & current_plan_id() const { return resident_current_plan_id; }
    std::string & current_plan_id() { return resident_current_plan_id; }
    const common_agent_runtime_resident_host & runtime_host() const { return host; }
    common_agent_runtime_resident_host & runtime_host() { return host; }

private:
    common_memory_store & memory_store;
    common_plan_store * plan_store = nullptr;
    common_agent_runtime_turn_request base_turn_request;
    std::string resident_current_plan_id;
    std::vector<common_blueprint_candidate> installed_blueprint_candidates;
    common_agent_runtime_tooling tooling;
    common_agent_runtime_resident_host host;
};
