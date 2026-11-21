#pragma once

#include <nlohmann/json.hpp>

#include <string>

// HTTP/SSE is only a framing adapter for the existing agent wire message.
// The JSON object is deliberately kept unchanged so clients can share the
// JSONL and SSE event contracts.
std::string common_agent_sse_format_message(
        const nlohmann::ordered_json & message,
        const std::string & event_name = "agent");

