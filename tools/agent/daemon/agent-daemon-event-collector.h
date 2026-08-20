#pragma once

#include "agent-daemon-events.h"

#include <cstdint>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

class common_agent_daemon_event_collector {
public:
    explicit common_agent_daemon_event_collector(size_t max_history_events = 1024);

    void append(common_agent_daemon_event event);

    std::vector<common_agent_daemon_event> take();

    std::string subscribe(common_agent_event_stream_subscription subscription);

    uint64_t latest_sequence() const;

    void unsubscribe(const std::string & subscription_id);

    common_agent_event_stream_wait_status wait_next(
        const std::string & subscription_id,
        common_agent_event_stream_delivery & delivery,
        std::chrono::milliseconds timeout);

private:
    struct subscription_state {
        common_agent_event_stream_subscription subscription;
        std::deque<common_agent_event_stream_delivery> pending;
        bool active = true;
    };

    mutable std::mutex mutex;
    std::condition_variable condition;
    std::vector<common_agent_daemon_event> pending_events;
    std::deque<common_agent_daemon_event> history;
    std::unordered_map<std::string, subscription_state> subscriptions;
    size_t max_history_events = 1024;
    uint64_t next_sequence = 1;
    uint64_t next_subscription_id = 1;
};
