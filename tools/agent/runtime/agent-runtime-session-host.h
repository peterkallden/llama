#pragma once

#include "../runtime/agent-runtime-control.h"
#include "../runtime/agent-runtime-tooling.h"
#include "../runtime/agent-runtime-turn.h"
#include "agent/contracts/agent-events.h"
#include "agent/contracts/agent-failures.h"
#include "agent/contracts/agent-result.h"
#include "agent-model-residency.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

class common_agent_runtime_resident_runtime;

struct common_agent_runtime_session_host_runtime_key {
    std::string session_id;
    std::string namespace_id;
    std::string project_id;
    common_memory_scope memory_scope = common_memory_scope::session;
    common_plan_scope plan_scope = common_plan_scope::turn;
    std::string model_profile_id;

    bool operator==(const common_agent_runtime_session_host_runtime_key & other) const {
        return session_id == other.session_id &&
               namespace_id == other.namespace_id &&
               project_id == other.project_id &&
               memory_scope == other.memory_scope &&
               plan_scope == other.plan_scope &&
               model_profile_id == other.model_profile_id;
    }
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
    std::optional<common_memory_policy_pack> policy_pack;
    common_agent_runtime_execution_control execution_control;
    std::optional<bool> allow_policy_gated_writes;
    std::vector<std::string> allowed_exposed_tool_names;
    std::optional<common_agent_deliberation_policy> deliberation_policy_override;
    std::vector<common_agent_input_resource> input_resources;
    common_agent_event_sink event_sink;
    // Daemon request identity is kept separate from the model turn identity,
    // but travels with the host-owned turn for checkpoint validation.
    std::string request_id;
    std::string model_profile_id;
};

struct common_agent_runtime_session_host_turn_result {
    bool ok = false;
    bool cancelled = false;
    bool runtime_reused = false;
    bool limit_reached = false;
    std::optional<common_agent_continuation_checkpoint> continuation_checkpoint;
    bool reflected = false;
    bool revised = false;
    common_agent_failure_class failure_class = common_agent_failure_class::execution;
    std::string response;
    std::string plan_id;
    int total_decoded_tokens = 0;
    common_agent_generation_status response_generation_status = common_agent_generation_status::errored;
    common_agent_generation_stop_reason response_stop_reason = common_agent_generation_stop_reason::error;
    size_t event_count = 0;
    size_t trace_count = 0;
    size_t memory_learning_related_count = 0;
    std::string memory_learning_summary;
    std::vector<common_agent_event> events;
    std::vector<common_runtime_trace_entry> trace;
    std::string error;
};

struct common_agent_runtime_session_host_descriptor {
    std::string namespace_id;
    std::string session_id;
    std::string project_id;
    common_memory_scope memory_scope = common_memory_scope::session;
    common_plan_scope plan_scope = common_plan_scope::turn;
    std::string policy_pack_id;
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
    std::function<bool(
        const common_agent_runtime_resident_runtime * runtime,
        const common_agent_runtime_session_host_turn_request & request,
        common_agent_runtime_tooling & tooling,
        std::string & error)> tooling_resolver;
    std::shared_ptr<common_agent_runtime_model_residency> model_residency;
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
    std::function<bool(
        const common_agent_runtime_resident_runtime * runtime,
        const common_agent_runtime_session_host_turn_request & request,
        common_agent_runtime_tooling & tooling,
        std::string & error)> tooling_resolver;
    std::shared_ptr<common_agent_runtime_model_residency> model_residency;
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

    bool prepare_model(
        const common_agent_runtime_session_host_turn_request & request,
        std::string & error);

    void reset();

    const common_agent_runtime_session * session() const;
    common_agent_runtime_session * session();
    common_agent_runtime_session_host_descriptor describe_session() const;

private:
    bool ensure_runtime(
        const common_agent_runtime_session_host_turn_request & request,
        bool & reused,
        std::string & error);

    common_agent_runtime_session_host_runtime_key make_runtime_key(
        const common_agent_runtime_session_host_turn_request & request) const;

    common_agent_runtime_turn_request make_base_turn_request(
        const common_agent_runtime_session_host_turn_request & request) const;

    std::optional<common_memory_policy_pack> resolve_policy_pack(
        const common_agent_runtime_session_host_turn_request & request) const;

    void update_session_policy_pack(
        const common_agent_runtime_session_host_turn_request & request);

    void compact_session_policy_pack_after_reflection(
        const common_agent_result & agent_result,
        common_agent_runtime_session_host_turn_result & result);

    bool resolve_tooling(
        const common_agent_runtime_resident_runtime * runtime,
        const common_agent_runtime_session_host_turn_request & request,
        common_agent_runtime_tooling & tooling,
        std::string & error) const;

    common_agent_runtime_session_host_config config;
    common_agent_runtime_session_host_runtime_key active_runtime_key;
    std::optional<common_memory_policy_pack> active_policy_pack;
    std::unique_ptr<common_agent_runtime_resident_runtime> runtime;
    uint64_t generated_turn_counter = 0;
};
