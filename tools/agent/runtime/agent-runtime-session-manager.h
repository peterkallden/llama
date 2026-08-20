#pragma once

#include "../daemon/agent-daemon-events.h"
#include "../runtime/agent-runtime-session-host.h"
#include "../runtime/agent-runtime-turn-driver.h"
#include "../runtime/agent-inference-executor.h"
#include "../runtime/agent-runtime-turn-execution.h"
#include "runtime/runtime-state.h"

#include <functional>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <condition_variable>
#include <vector>

class common_agent_inference_capacity_gate;

struct common_agent_runtime_session_key {
    std::string namespace_id;
    std::string session_id;

    bool operator<(const common_agent_runtime_session_key & other) const {
        if (namespace_id != other.namespace_id) return namespace_id < other.namespace_id;
        return session_id < other.session_id;
    }
};

struct common_agent_runtime_session_descriptor {
    common_agent_runtime_session_key key;
    std::string project_id;
    common_memory_scope memory_scope = common_memory_scope::session;
    common_plan_scope plan_scope = common_plan_scope::turn;
    std::string policy_pack_id;
    std::string lane_state;
    size_t queued_turn_count = 0;
    bool has_active_turn = false;
    std::string active_request_id;
    std::string active_turn_id;
    std::string active_turn_phase;
    std::string active_turn_disposition;
    bool active_cancel_requested = false;
    std::string pending_operation_kind;
    std::string pending_operation_detail;
    std::string last_turn_id;
    std::string last_turn_phase;
    std::string last_turn_disposition;
};

inline common_agent_state_descriptor describe_agent_runtime_session(
        const common_agent_runtime_session_descriptor & session) {
    common_agent_state_descriptor descriptor;
    descriptor.state_id = session.key.namespace_id + ":" + session.key.session_id;
    descriptor.state_type = "session_lane";
    descriptor.state_class = common_agent_state_class::resident_runtime;
    descriptor.lifetime = common_agent_state_lifetime::session;
    descriptor.persistence = common_agent_state_persistence::none;
    descriptor.identity.namespace_id = session.key.namespace_id;
    descriptor.identity.session_id = session.key.session_id;
    descriptor.identity.project_id = session.project_id;
    descriptor.owner = "common_agent_runtime_session_manager";
    descriptor.source_of_truth = "session lane";
    return descriptor;
}

struct common_agent_runtime_session_manager_turn_request {
    std::string request_id;
    common_agent_runtime_session_host_turn_request turn;
};
using common_agent_runtime_session_manager_turn_result = common_agent_runtime_session_host_turn_result;

struct common_agent_runtime_session_manager_config {
    common_agent_runtime_session_host_config host_config;
    std::shared_ptr<common_agent_inference_capacity_gate> inference_gate;
    std::shared_ptr<common_agent_runtime_inference_executor> inference_executor;
    std::function<bool(
        const common_agent_runtime_session_host_turn_request & request,
        std::optional<common_agent_runtime_session_manager_pending_operation> & pending_operation,
        std::string & error)> pending_operation_resolver;
};

struct common_agent_runtime_session_manager_build_config {
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
    std::function<bool(
        const common_agent_runtime_session_host_turn_request & request,
        std::optional<common_agent_runtime_session_manager_pending_operation> & pending_operation,
        std::string & error)> pending_operation_resolver;
    std::shared_ptr<common_agent_inference_capacity_gate> inference_gate;
    std::shared_ptr<common_agent_runtime_inference_executor> inference_executor;
};

struct common_agent_runtime_active_turn_descriptor {
    common_agent_runtime_session_key key;
    std::string project_id;
    std::string request_id;
    std::string turn_id;
    std::string phase;
    std::string disposition;
    bool cancellation_requested = false;
    std::string pending_operation_kind;
    std::string pending_operation_detail;
};

inline common_agent_runtime_session_manager_config make_agent_runtime_session_manager_config(
        common_agent_runtime_session_manager_build_config config) {
    common_agent_runtime_session_host_build_config host_build_config = {
        config.memory_store,
        config.plan_store,
        std::move(config.resident_request),
        std::move(config.policy),
        std::move(config.runtime_config),
        std::move(config.orchestration_config),
        config.memory_scope,
        config.memory_enabled,
        std::move(config.installed_blueprint_candidates),
        std::move(config.tooling),
        std::move(config.tooling_resolver),
    };
    return {
        make_agent_runtime_session_host_config(std::move(host_build_config)),
        std::move(config.inference_gate),
        std::move(config.inference_executor),
        std::move(config.pending_operation_resolver),
    };
}

enum class common_agent_runtime_session_lane_state {
    idle,
    running,
    running_with_waiters,
    resetting,
    closing,
};

inline const char * common_agent_runtime_session_lane_state_name(
        common_agent_runtime_session_lane_state state) {
    switch (state) {
        case common_agent_runtime_session_lane_state::idle:                 return "idle";
        case common_agent_runtime_session_lane_state::running:              return "running";
        case common_agent_runtime_session_lane_state::running_with_waiters: return "running_with_waiters";
        case common_agent_runtime_session_lane_state::resetting:            return "resetting";
        case common_agent_runtime_session_lane_state::closing:              return "closing";
    }
    return "idle";
}

class common_agent_runtime_session_manager {
public:
    explicit common_agent_runtime_session_manager(common_agent_runtime_session_manager_config config);

    bool run_turn(
        const common_agent_runtime_session_manager_turn_request & request,
        common_agent_runtime_session_manager_turn_result & result,
        std::string & error);

    bool reset_session(
        const common_agent_runtime_session_key & key,
        std::string & error);

    bool close_session(
        const common_agent_runtime_session_key & key,
        std::string & error);

    bool request_cancel_active_turn(
        const std::string & target_request_id,
        const std::string & target_turn_id,
        common_agent_runtime_active_turn_descriptor & active_turn,
        std::string & error);

    void set_event_sink(common_agent_daemon_event_sink sink);

    std::optional<common_agent_runtime_active_turn_descriptor> describe_active_turn() const;

    std::vector<common_agent_runtime_session_descriptor> list_sessions() const;

    void reset_all();

private:
    struct common_agent_runtime_session_lane_message {
        size_t id = 0;
        common_agent_runtime_session_manager_turn_request request;
        common_agent_runtime_session_manager_turn_result result;
        std::string error;
        bool ok = false;
        bool completed = false;
        mutable std::mutex mutex;
        std::condition_variable condition;
    };

    struct common_agent_runtime_session_lane {
        std::unique_ptr<common_agent_runtime_session_host> host;
        std::deque<std::shared_ptr<common_agent_runtime_session_lane_message>> mailbox;
        std::shared_ptr<common_agent_runtime_session_lane_message> current_message;
        std::optional<common_agent_runtime_turn_execution> active_turn;
        std::optional<common_agent_runtime_session_manager_pending_operation> pending_operation;
        std::string last_turn_id;
        common_agent_runtime_turn_phase last_turn_phase = common_agent_runtime_turn_phase::queued;
        common_agent_runtime_turn_disposition last_turn_disposition = common_agent_runtime_turn_disposition::continue_immediately;
        size_t next_message_id = 1;
        common_agent_runtime_session_lane_state state = common_agent_runtime_session_lane_state::idle;
        mutable std::mutex mutex;
    };

    common_agent_runtime_session_key make_session_key(
        const common_agent_runtime_session_manager_turn_request & request) const;

    common_agent_runtime_session_lane & ensure_session_lane(
        const common_agent_runtime_session_key & key);

    std::shared_ptr<common_agent_runtime_session_lane_message> enqueue_lane_turn(
        common_agent_runtime_session_lane & lane,
        const common_agent_runtime_session_manager_turn_request & request,
        common_agent_runtime_session_manager_turn_result & result,
        std::string & error);

    bool wait_for_message_completion(
        const std::shared_ptr<common_agent_runtime_session_lane_message> & message,
        std::string & error) const;

    void complete_lane_message(
        const std::shared_ptr<common_agent_runtime_session_lane_message> & message,
        bool ok,
        const std::string & error,
        bool cancelled = false) const;

    void reconcile_lane_state(
        common_agent_runtime_session_lane & lane) const;

    bool run_lane_turn(
        common_agent_runtime_session_lane & lane,
        const std::shared_ptr<common_agent_runtime_session_lane_message> & message);

    common_agent_runtime_turn_disposition advance_lane_turn(
        common_agent_runtime_session_lane & lane,
        const std::shared_ptr<common_agent_runtime_session_lane_message> & message);

    bool drain_lane(
        common_agent_runtime_session_lane & lane,
        const std::shared_ptr<common_agent_runtime_session_lane_message> & target_message,
        std::string & error);

    bool prepare_lane_transition(
        common_agent_runtime_session_lane & lane,
        common_agent_runtime_session_lane_state target_state,
        const char * pending_error,
        std::shared_ptr<common_agent_runtime_session_lane_message> & current_message,
        std::string & error);

    common_agent_event_emitter make_lane_emitter(
        const common_agent_runtime_session_key & key,
        const std::string & project_id,
        const std::string & request_id,
        const std::string & turn_id,
        const std::string & operation_id = {}) const;

    common_agent_runtime_session_manager_config config;
    common_runtime_operation_manager operation_manager;
    std::map<common_agent_runtime_session_key, common_agent_runtime_session_lane> lanes;
    common_agent_daemon_event_sink event_sink;
};
