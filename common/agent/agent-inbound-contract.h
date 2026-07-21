#pragma once

#include "agent-contract.h"

#include <optional>
#include <string>
#include <vector>

// Transport-neutral contract for an agent request received from another
// agent, an IDE adapter, or an MCP client. The transport remains responsible
// for authentication; this contract carries the already identified caller
// into the existing runtime/dispatcher path.
struct common_agent_inbound_limits {
    std::optional<int> max_reflection_rounds;
    std::optional<int> max_plan_revisions;
    std::optional<size_t> max_research_iterations;
    std::optional<size_t> max_tool_rounds;
};

struct common_agent_inbound_request {
    std::string request_id;
    std::string caller_id;
    std::string parent_request_id;

    std::string task;
    common_agent_thinking_request thinking_request =
        common_agent_thinking_request::auto_select;

    std::string namespace_id;
    std::string project_id;
    std::string session_id;
    std::string turn_id;

    std::vector<common_agent_input_resource> input_resources;
    common_agent_inbound_limits limits;
    int delegation_depth = 0;
};

struct common_agent_inbound_result {
    bool accepted = false;
    bool ok = false;
    bool cancelled = false;

    std::string request_id;
    std::string operation_id;
    std::string response;
    std::string plan_id;
    std::string thinking_mode;

    common_agent_failure_class failure_class = common_agent_failure_class::execution;
    std::string failure_code;
    std::string error;

    size_t event_count = 0;
    size_t trace_count = 0;
};
