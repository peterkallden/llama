#include "agent-sse.h"

#include <cstdint>

namespace {

uint64_t event_sequence(const nlohmann::ordered_json & message) {
    if (message.contains("event") && message["event"].is_object()) {
        return message["event"].value("sequence", uint64_t(0));
    }
    if (message.contains("cursor") && message["cursor"].is_object()) {
        return message["cursor"].value("after_sequence", uint64_t(0));
    }
    return 0;
}

} // namespace

std::string common_agent_sse_format_message(
        const nlohmann::ordered_json & message,
        const std::string & event_name) {
    std::string result;
    const auto sequence = event_sequence(message);
    if (sequence != 0) {
        result += "id: " + std::to_string(sequence) + "\n";
    }
    if (!event_name.empty()) {
        result += "event: " + event_name + "\n";
    }
    result += "data: " + message.dump() + "\n\n";
    return result;
}

