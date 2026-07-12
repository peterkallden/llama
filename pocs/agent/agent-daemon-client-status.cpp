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
