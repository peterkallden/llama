// Thread-safe transport for structured agent events.
//
// The queue is deliberately separate from the event semantics.  Native
// hosts, including JNI, can enqueue events from a worker thread and let the
// UI/client poll them without callbacks crossing an unpredictable lifecycle.
#pragma once

#include "agent/contracts/agent-events.h"

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>

class common_agent_event_queue {
public:
    bool push(common_agent_event event) {
        std::lock_guard<std::mutex> lock(mutex);
        if (closed_value) return false;
        events.push_back(std::move(event));
        condition.notify_one();
        return true;
    }

    bool try_pop(common_agent_event & event) {
        std::lock_guard<std::mutex> lock(mutex);
        if (events.empty()) return false;
        event = std::move(events.front());
        events.pop_front();
        return true;
    }

    bool wait_pop(
            common_agent_event & event,
            std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex);
        condition.wait_for(lock, timeout, [this] {
            return closed_value || !events.empty();
        });
        if (events.empty()) return false;
        event = std::move(events.front());
        events.pop_front();
        return true;
    }

    void close() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            closed_value = true;
        }
        condition.notify_all();
    }

    bool closed() const {
        std::lock_guard<std::mutex> lock(mutex);
        return closed_value;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex);
        return events.size();
    }

private:
    mutable std::mutex mutex;
    std::condition_variable condition;
    std::deque<common_agent_event> events;
    bool closed_value = false;
};
