#pragma once

#include "agent-runtime-session-host.h"
#include "agent-runtime-turn-execution.h"

#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <condition_variable>
#include <vector>

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
    size_t queued_turn_count = 0;
    bool has_active_turn = false;
    std::string active_request_id;
    std::string active_turn_id;
    std::string active_turn_phase;
    std::string active_turn_disposition;
    bool active_cancel_requested = false;
    std::string last_turn_id;
    std::string last_turn_phase;
    std::string last_turn_disposition;
};

struct common_agent_runtime_session_manager_turn_request {
    std::string request_id;
    common_agent_runtime_session_host_turn_request turn;
};
using common_agent_runtime_session_manager_turn_result = common_agent_runtime_session_host_turn_result;
using common_agent_runtime_session_manager_config = common_agent_runtime_session_host_config;
using common_agent_runtime_session_manager_build_config = common_agent_runtime_session_host_build_config;

struct common_agent_runtime_active_turn_descriptor {
    common_agent_runtime_session_key key;
    std::string project_id;
    std::string request_id;
    std::string turn_id;
    std::string phase;
    std::string disposition;
    bool cancellation_requested = false;
};

inline common_agent_runtime_session_manager_config make_agent_runtime_session_manager_config(
        common_agent_runtime_session_manager_build_config config) {
    return make_agent_runtime_session_host_config(std::move(config));
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

    std::optional<common_agent_runtime_active_turn_descriptor> describe_active_turn() const;

    std::vector<common_agent_runtime_session_descriptor> list_sessions() const;

    void reset_all();

private:
    struct common_agent_runtime_session_lane_message {
        size_t id = 0;
        common_agent_runtime_session_manager_turn_request request;
        common_agent_runtime_session_manager_turn_result * result = nullptr;
        std::string * error = nullptr;
        bool ok = false;
        bool completed = false;
        mutable std::mutex mutex;
        std::condition_variable condition;
    };

    struct common_agent_runtime_session_lane {
        std::unique_ptr<common_agent_runtime_session_host> host;
        std::deque<std::shared_ptr<common_agent_runtime_session_lane_message>> mailbox;
        std::optional<common_agent_runtime_turn_execution> active_turn;
        std::string last_turn_id;
        common_agent_runtime_turn_phase last_turn_phase = common_agent_runtime_turn_phase::queued;
        common_agent_runtime_turn_disposition last_turn_disposition = common_agent_runtime_turn_disposition::continue_immediately;
        size_t next_message_id = 1;
        bool draining = false;
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

    bool run_lane_turn(
        common_agent_runtime_session_lane & lane,
        const common_agent_runtime_session_manager_turn_request & request,
        common_agent_runtime_session_manager_turn_result & result,
        std::string & error);

    common_agent_runtime_turn_disposition advance_lane_turn(
        common_agent_runtime_session_lane & lane,
        const common_agent_runtime_session_manager_turn_request & request,
        common_agent_runtime_session_manager_turn_result & result,
        std::string & error);

    bool drain_lane(
        common_agent_runtime_session_lane & lane,
        const std::shared_ptr<common_agent_runtime_session_lane_message> & target_message,
        std::string & error);

    common_agent_runtime_session_manager_config config;
    std::map<common_agent_runtime_session_key, common_agent_runtime_session_lane> lanes;
};
