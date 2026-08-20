#include "agent-daemon-client-status.h"

#include <string>

namespace {

std::string bool_name(bool value) {
    return value ? "yes" : "no";
}

} // namespace

agent_daemon_client_status_summary make_agent_daemon_client_status_summary(
        const agent_daemon_jsonl_status_response & response) {
    agent_daemon_client_status_summary summary;
    summary.headline =
        "state=" + response.state +
        " live=" + bool_name(response.live) +
        " ready=" + bool_name(response.ready) +
        " worker=" + bool_name(response.worker_running) +
        " accepting=" + bool_name(response.accepting_commands) +
        " shutdown=" + bool_name(response.shutdown_requested) +
        " queued=" + std::to_string(response.queued_commands) +
        "/" + std::to_string(response.max_queue_size) +
        " capacity=" + std::to_string(response.queue_capacity_remaining) +
        " sessions=" + std::to_string(response.sessions);
    if (!response.active_request_id.empty()) {
        summary.headline += " active_request=" + response.active_request_id;
    }
    if (!response.active_turn_id.empty()) {
        summary.headline += " active_turn=" + response.active_turn_id;
        if (!response.active_turn_phase.empty()) {
            summary.headline += "/" + response.active_turn_phase;
        }
        if (!response.active_turn_disposition.empty()) {
            summary.headline += ":" + response.active_turn_disposition;
        }
        if (!response.active_pending_operation_kind.empty()) {
            summary.headline += " pending=" + response.active_pending_operation_kind;
            if (!response.active_pending_operation_detail.empty()) {
                summary.headline += "(" + response.active_pending_operation_detail + ")";
            }
        }
    }

    summary.session_bindings.reserve(response.session_keys.size());
    for (const auto & session : response.session_keys) {
        std::string binding = session.namespace_id + "/" + session.session_id;
        if (!session.project_id.empty()) {
            binding += "@" + session.project_id;
        }
        if (!session.policy_pack_id.empty()) {
            binding += "#" + session.policy_pack_id;
        }
        if (!session.lane_state.empty()) {
            binding += "{state=" + session.lane_state + "}";
        }
        if (session.has_active_turn) {
            binding += "[active=" + session.active_turn_id;
            if (!session.active_turn_phase.empty()) {
                binding += "/" + session.active_turn_phase;
            }
            if (!session.active_turn_disposition.empty()) {
                binding += ":" + session.active_turn_disposition;
            }
            if (!session.pending_operation_kind.empty()) {
                binding += " pending=" + session.pending_operation_kind;
                if (!session.pending_operation_detail.empty()) {
                    binding += "(" + session.pending_operation_detail + ")";
                }
            }
            binding += "]";
        }
        summary.session_bindings.push_back(std::move(binding));
    }

    return summary;
}

std::string render_agent_daemon_client_status_summary(
        const agent_daemon_client_status_summary & summary) {
    std::string rendered = summary.headline;
    if (!summary.session_bindings.empty()) {
        rendered += " session_bindings=";
        for (size_t i = 0; i < summary.session_bindings.size(); ++i) {
            if (i > 0) {
                rendered += ",";
            }
            rendered += summary.session_bindings[i];
        }
    }
    return rendered;
}

std::string render_agent_daemon_client_status_verbose(
        const agent_daemon_jsonl_status_response & response) {
    if (response.payload.is_object()) {
        return response.payload.dump(2);
    }

    return nlohmann::ordered_json{{
        "event", response.event,
    }, {
        "state", response.state,
    }, {
        "live", response.live,
    }, {
        "ready", response.ready,
    }, {
        "readiness", response.readiness,
    }}.dump(2);
}

agent_daemon_client_lifecycle_summary make_agent_daemon_client_lifecycle_summary(
        const agent_daemon_jsonl_lifecycle_response & response) {
    agent_daemon_client_lifecycle_summary summary;
    summary.event = response.event;
    summary.status = make_agent_daemon_client_status_summary(response.status);
    return summary;
}

std::string render_agent_daemon_client_lifecycle_summary(
        const agent_daemon_client_lifecycle_summary & summary) {
    return summary.event + " " + render_agent_daemon_client_status_summary(summary.status);
}

agent_daemon_client_turn_failure_summary make_agent_daemon_client_turn_failure_summary(
        const agent_daemon_jsonl_turn_response & response,
        const std::string & fallback_error) {
    agent_daemon_client_turn_failure_summary summary;
    summary.headline = response.event.empty() ? "turn_failed" : response.event;

    const std::string error =
        !response.error.empty() ? response.error : fallback_error;
    if (!response.failure_class.empty()) {
        summary.headline += " class=" + response.failure_class;
    }
    if (!response.response_generation_status.empty()) {
        summary.headline += " status=" + response.response_generation_status;
    }
    if (!response.response_stop_reason.empty()) {
        summary.headline += " stop=" + response.response_stop_reason;
    }
    if (response.cancelled) {
        summary.headline += " cancelled=yes";
    }
    if (!error.empty()) {
        summary.headline += " error=" + error;
    }
    return summary;
}

std::string render_agent_daemon_client_turn_failure_summary(
        const agent_daemon_client_turn_failure_summary & summary) {
    return summary.headline;
}
