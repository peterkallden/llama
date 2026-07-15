#include "agent-daemon-event-collector.h"

#include <utility>

void common_agent_daemon_event_collector::append(
        common_agent_daemon_event event) {
    std::lock_guard<std::mutex> lock(mutex);
    event.type = common_agent_daemon_event_type_name(event.event_type);
    event.sequence = next_sequence++;
    pending_events.push_back(std::move(event));
}

std::vector<common_agent_daemon_event> common_agent_daemon_event_collector::take() {
    std::lock_guard<std::mutex> lock(mutex);
    auto out = std::move(pending_events);
    pending_events.clear();
    return out;
}
