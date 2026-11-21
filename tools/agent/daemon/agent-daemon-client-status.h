#pragma once

#include "../daemon/agent-daemon-jsonl-protocol.h"

#include <string>
#include <vector>

struct agent_daemon_client_status_summary {
    std::string headline;
    std::vector<std::string> session_bindings;
};

struct agent_daemon_client_lifecycle_summary {
    std::string event;
    agent_daemon_client_status_summary status;
};

struct agent_daemon_client_turn_failure_summary {
    std::string headline;
};

agent_daemon_client_status_summary make_agent_daemon_client_status_summary(
    const agent_daemon_jsonl_status_response & response);

std::string render_agent_daemon_client_status_summary(
    const agent_daemon_client_status_summary & summary);

std::string render_agent_daemon_client_status_verbose(
    const agent_daemon_jsonl_status_response & response);

agent_daemon_client_lifecycle_summary make_agent_daemon_client_lifecycle_summary(
    const agent_daemon_jsonl_lifecycle_response & response);

std::string render_agent_daemon_client_lifecycle_summary(
    const agent_daemon_client_lifecycle_summary & summary);

agent_daemon_client_turn_failure_summary make_agent_daemon_client_turn_failure_summary(
    const agent_daemon_jsonl_turn_response & response,
    const std::string & fallback_error = {});

std::string render_agent_daemon_client_turn_failure_summary(
    const agent_daemon_client_turn_failure_summary & summary);
