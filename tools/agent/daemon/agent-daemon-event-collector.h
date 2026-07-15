#pragma once

#include "agent-daemon-events.h"

#include <cstdint>
#include <mutex>
#include <vector>

class common_agent_daemon_event_collector {
public:
    void append(common_agent_daemon_event event);

    std::vector<common_agent_daemon_event> take();

private:
    std::mutex mutex;
    std::vector<common_agent_daemon_event> pending_events;
    uint64_t next_sequence = 1;
};
