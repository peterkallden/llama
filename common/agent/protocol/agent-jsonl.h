#pragma once

#include <cstdio>
#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <vector>

class common_agent_jsonl_stream {
public:
    virtual ~common_agent_jsonl_stream() = default;

    virtual bool read(nlohmann::ordered_json & message, std::string & error) = 0;
    virtual bool write(const nlohmann::ordered_json & message, std::string & error) = 0;
    virtual bool eof() const = 0;
};

bool common_agent_jsonl_parse_line(
        const std::string & line,
        nlohmann::ordered_json & out,
        std::string & error);

std::string common_agent_jsonl_make_line(
        const nlohmann::ordered_json & message);

// Transport-neutral wire contracts shared by daemon, TCP and Android hosts.
// Daemon administration commands remain in tools/agent/daemon; this header
// contains only the request/result/event surface needed by a runtime client.
struct common_agent_jsonl_turn_request {
    std::string prompt;
    std::string session_id;
    std::string namespace_id;
    std::string project_id;
    std::string turn_id;
    std::string memory_scope;
    std::string plan_scope;
    int n_predict = 0;
    std::string mode = "chat";
    std::vector<std::string> resource_refs;
    bool include_summary = false;
    std::optional<size_t> turn_timeout_ms;
    std::optional<uint32_t> inference_step_timeout_ms;
    std::optional<uint32_t> tool_timeout_ms;
    std::optional<uint32_t> mcp_connect_timeout_ms;
    std::optional<uint32_t> mcp_request_timeout_ms;
    std::optional<uint32_t> mcp_shutdown_timeout_ms;
};

struct common_agent_jsonl_event_entry {
    std::string type;
    std::string request_id;
    std::string turn_id;
    std::string namespace_id;
    std::string project_id;
    std::string session_id;
    std::string operation_id;
    std::string detail;
    std::string event_type;
    uint64_t sequence = 0;
    std::string memory_id;
    std::string plan_id;
    std::string step_id;
    std::string observation_id;
    std::string tool_name;
    std::string resource_uri;
};

struct common_agent_jsonl_turn_result {
    std::string request_id;
    bool ok = false;
    bool cancelled = false;
    std::string response;
    std::string plan_id;
    std::string error;
    std::string failure_class;
    int event_count = 0;
};

bool common_agent_jsonl_read_message(
        FILE * stream,
        nlohmann::ordered_json & out,
        std::string & error);

bool common_agent_jsonl_write_message(
        FILE * stream,
        const nlohmann::ordered_json & message,
        std::string & error);

nlohmann::ordered_json common_agent_jsonl_make_turn_request(
        const common_agent_jsonl_turn_request & request);

nlohmann::ordered_json common_agent_jsonl_make_turn_result(
        const common_agent_jsonl_turn_result & result);

nlohmann::ordered_json common_agent_jsonl_make_event_message(
        const common_agent_jsonl_event_entry & event);
