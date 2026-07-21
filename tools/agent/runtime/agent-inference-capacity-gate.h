#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

enum class common_agent_inference_priority {
    background,
    normal,
    interactive,
};

struct common_agent_inference_wait_request {
    std::string waiter_id;
    common_agent_inference_priority priority = common_agent_inference_priority::normal;
    std::chrono::steady_clock::time_point deadline{};
};

// Host-owned capacity gate for active model executions. It deliberately does
// not know about daemon workers, sessions, server slots, or GPU details.
class common_agent_inference_capacity_gate {
public:
    explicit common_agent_inference_capacity_gate(size_t capacity = 1);

    // Legacy anonymous lease API. New production callers should use the
    // named waiter API below so admission remains fair and cancellable.
    bool try_acquire();
    void release();

    bool enqueue(
            common_agent_inference_wait_request request,
            std::string & error);
    bool try_acquire(const std::string & waiter_id);
    bool cancel(const std::string & waiter_id, std::string & error);
    void release(const std::string & waiter_id);

    size_t capacity() const;
    size_t active() const;
    size_t waiting() const;

private:
    struct waiter {
        common_agent_inference_wait_request request;
        uint64_t sequence = 0;
        std::chrono::steady_clock::time_point enqueued_at;
    };

    std::vector<waiter>::iterator select_next_waiter();

    mutable std::mutex mutex_;
    size_t capacity_ = 1;
    size_t active_ = 0;
    uint64_t next_sequence_ = 1;
    std::vector<waiter> waiters_;
    std::unordered_set<std::string> active_waiters_;
};
