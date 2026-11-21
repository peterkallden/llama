#pragma once

#include "../daemon/agent-daemon-events.h"
#include "agent-runtime-session-host.h"
#include "agent-runtime-turn-execution.h"
#include "agent-inference-executor.h"

#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

class common_agent_inference_capacity_gate;

struct common_agent_runtime_session_manager_pending_operation {
    common_agent_runtime_pending_operation pending_operation;
    common_agent_runtime_turn_phase waiting_phase = common_agent_runtime_turn_phase::awaiting_tool;
    common_agent_runtime_turn_disposition waiting_disposition =
        common_agent_runtime_turn_disposition::wait_for_tool;
    std::function<bool(bool & ready, std::string & error)> poll;
    std::function<bool(std::string & error)> cancel;

    common_runtime_operation_manager::poll_callback take_poll_callback() {
        return std::move(poll);
    }

    common_runtime_operation_manager::cancel_callback take_cancel_callback() {
        return std::move(cancel);
    }
};

common_agent_runtime_turn_disposition poll_common_agent_runtime_pending_operation(
        common_runtime_operation_manager & operation_manager,
        std::optional<common_agent_runtime_session_manager_pending_operation> & pending_operation,
        std::optional<common_agent_runtime_turn_execution> & active_turn,
        std::mutex & lane_mutex,
        const common_agent_runtime_session_host_turn_request & request,
        common_agent_runtime_session_host_turn_result & result,
        std::string & error,
        common_agent_runtime_turn_phase phase,
        const common_agent_event_emitter & turn_events);

common_agent_runtime_turn_disposition advance_common_agent_runtime_turn(
        common_agent_runtime_session_host * host,
        std::optional<common_agent_runtime_session_manager_pending_operation> & pending_operation,
        std::optional<common_agent_runtime_turn_execution> & active_turn,
        std::mutex & lane_mutex,
        common_runtime_operation_manager & operation_manager,
        const std::function<bool(
            const common_agent_runtime_session_host_turn_request & request,
            std::optional<common_agent_runtime_session_manager_pending_operation> & pending_operation,
            std::string & error)> & pending_operation_resolver,
        const std::shared_ptr<common_agent_inference_capacity_gate> & inference_gate,
        const std::shared_ptr<common_agent_runtime_inference_executor> & inference_executor,
        const std::string & request_id,
        const common_agent_runtime_session_host_turn_request & request,
        common_agent_runtime_session_host_turn_result & result,
        std::string & error,
        const common_agent_event_emitter & turn_events);
