#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

struct common_agent_runtime_timeout_policy {
    size_t turn_timeout_ms = 0;
    uint32_t inference_step_timeout_ms = 0;
    uint32_t tool_timeout_ms = 0;
    uint32_t mcp_connect_timeout_ms = 0;
    uint32_t mcp_request_timeout_ms = 0;
    uint32_t mcp_shutdown_timeout_ms = 0;

    bool has_any_timeout() const {
        return turn_timeout_ms > 0 ||
               inference_step_timeout_ms > 0 ||
               tool_timeout_ms > 0 ||
               mcp_connect_timeout_ms > 0 ||
               mcp_request_timeout_ms > 0 ||
               mcp_shutdown_timeout_ms > 0;
    }
};

class common_agent_runtime_cancellation_state {
public:
    bool request_cancel(std::string reason = "cancelled") {
        const bool already_cancelled = cancelled.exchange(true);
        if (!already_cancelled) {
            std::lock_guard<std::mutex> lock(mutex);
            reason_value = std::move(reason);
        }
        return !already_cancelled;
    }

    bool is_cancelled() const {
        return cancelled.load();
    }

    std::string reason() const {
        std::lock_guard<std::mutex> lock(mutex);
        return reason_value;
    }

private:
    std::atomic<bool> cancelled = false;
    mutable std::mutex mutex;
    std::string reason_value;
};

struct common_agent_runtime_execution_control {
    common_agent_runtime_timeout_policy timeout_policy;
    std::shared_ptr<common_agent_runtime_cancellation_state> cancellation;
    std::optional<std::chrono::steady_clock::time_point> deadline;

    bool has_deadline() const {
        return deadline.has_value();
    }

    bool is_cancel_requested() const {
        return cancellation && cancellation->is_cancelled();
    }

    bool is_deadline_exceeded() const {
        return deadline.has_value() &&
               std::chrono::steady_clock::now() >= *deadline;
    }

    bool should_stop() const {
        return is_cancel_requested() || is_deadline_exceeded();
    }

    std::string stop_reason() const {
        if (is_cancel_requested()) {
            const std::string reason_value = cancellation->reason();
            return reason_value.empty() ? "turn cancelled" : reason_value;
        }
        if (is_deadline_exceeded()) {
            return "turn deadline exceeded";
        }
        return {};
    }
};

inline common_agent_runtime_execution_control make_common_agent_runtime_execution_control(
        const common_agent_runtime_timeout_policy & timeout_policy) {
    common_agent_runtime_execution_control control;
    control.timeout_policy = timeout_policy;
    control.cancellation = std::make_shared<common_agent_runtime_cancellation_state>();
    if (timeout_policy.turn_timeout_ms > 0) {
        control.deadline =
            std::chrono::steady_clock::now() +
            std::chrono::milliseconds(timeout_policy.turn_timeout_ms);
    }
    return control;
}
