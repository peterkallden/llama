#pragma once

#include <cstdio>
#include <nlohmann/json.hpp>

#include <string>
#include <vector>

struct agent_daemon_jsonl_turn_request {
    std::string prompt;
    std::string session_id;
    std::string namespace_id;
    std::string project_id;
    std::string turn_id;
    std::string memory_scope;
    std::string plan_scope;
    int n_predict = 0;
    std::string mode = "chat";
};

struct agent_daemon_jsonl_ready_response {
    std::string default_mode;
    int protocol_version = 0;
    std::vector<std::string> capabilities;
};

bool read_agent_daemon_jsonl_message(
    FILE * stream,
    nlohmann::ordered_json & out,
    std::string & error);

bool write_agent_daemon_jsonl_message(
    FILE * stream,
    const nlohmann::ordered_json & message,
    std::string & error);

nlohmann::ordered_json make_agent_daemon_jsonl_turn_request(
    const agent_daemon_jsonl_turn_request & request);

nlohmann::ordered_json make_agent_daemon_jsonl_shutdown_request();

bool parse_agent_daemon_jsonl_ready_response(
    const nlohmann::ordered_json & message,
    agent_daemon_jsonl_ready_response & response,
    std::string & error);

bool parse_agent_daemon_jsonl_event_response(
    const nlohmann::ordered_json & message,
    const std::string & expected_event,
    std::string & error);
