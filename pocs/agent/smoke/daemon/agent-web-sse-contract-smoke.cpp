#include "common/agent/protocol/agent-sse.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <string>

using json = nlohmann::ordered_json;

int main() {
    const json event = {
        {"message_type", "event"},
        {"delivery_kind", "event"},
        {"cursor", {{"after_sequence", 17}}},
        {"event", {
            {"type", "tool.completed"},
            {"sequence", 17},
            {"tool_name", "data.select"},
        }},
    };
    const auto formatted = common_agent_sse_format_message(event);
    if (formatted.find("id: 17\n") == std::string::npos ||
            formatted.find("event: agent\n") == std::string::npos ||
            formatted.find("data: " + event.dump() + "\n\n") == std::string::npos) {
        std::fprintf(stderr, "SSE formatter did not preserve the JSONL payload\n");
        return 1;
    }
    if (formatted.find("tool.completed") == std::string::npos ||
            formatted.find("data.select") == std::string::npos) {
        std::fprintf(stderr, "SSE formatter omitted event content\n");
        return 1;
    }
    const auto heartbeat_payload = common_agent_sse_format_message({{"heartbeat", true}}, "");
    if (heartbeat_payload != "data: {\"heartbeat\":true}\n\n") {
        std::fprintf(stderr, "SSE formatter heartbeat contract mismatch\n");
        return 1;
    }
    return 0;
}

