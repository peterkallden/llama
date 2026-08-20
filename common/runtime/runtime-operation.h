#pragma once

#include "runtime-state.h"

#include <chrono>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <vector>

enum class common_runtime_operation_kind {
    inference,
    tool,
};

inline const char * common_runtime_operation_kind_name(
        common_runtime_operation_kind kind) {
    switch (kind) {
        case common_runtime_operation_kind::inference: return "inference";
        case common_runtime_operation_kind::tool:      return "tool";
    }
    return "tool";
}

struct common_runtime_operation {
    std::string operation_id;
    common_runtime_operation_kind kind = common_runtime_operation_kind::tool;
    std::string detail;
    std::chrono::steady_clock::time_point deadline{};
};

struct common_runtime_operation_ref {
    std::string operation_id;
    common_runtime_operation_kind kind = common_runtime_operation_kind::tool;
    std::string subject_name;
    // The producer's effective deadline is copied into the operation-manager
    // entry when an async operation is registered.
    std::chrono::steady_clock::time_point deadline{};
};

enum class common_runtime_operation_state {
    running,
    completed,
    failed,
    cancelled,
    timed_out,
};

inline const char * common_runtime_operation_state_name(
        common_runtime_operation_state state) {
    switch (state) {
        case common_runtime_operation_state::running:   return "running";
        case common_runtime_operation_state::completed: return "completed";
        case common_runtime_operation_state::failed:    return "failed";
        case common_runtime_operation_state::cancelled: return "cancelled";
        case common_runtime_operation_state::timed_out: return "timed_out";
    }
    return "failed";
}

struct common_runtime_operation_status {
    common_runtime_operation operation;
    common_runtime_operation_state state = common_runtime_operation_state::running;
    std::string error;
};

inline common_agent_state_descriptor describe_common_runtime_operation(
        const common_runtime_operation_status & status) {
    common_agent_state_descriptor descriptor;
    descriptor.state_id = status.operation.operation_id;
    descriptor.state_type = "pending_operation";
    descriptor.state_class = common_agent_state_class::resident_runtime;
    descriptor.lifetime = common_agent_state_lifetime::operation;
    descriptor.persistence = common_agent_state_persistence::none;
    descriptor.identity.operation_id = status.operation.operation_id;
    descriptor.owner = "common_runtime_operation_manager";
    descriptor.source_of_truth = "operation status";
    return descriptor;
}

// Host-neutral registry for pending work. Session lanes decide when to poll;
// this type owns identity, deadlines, terminal state, and cleanup.
class common_runtime_operation_manager {
public:
    using poll_callback = std::function<bool(bool & ready, std::string & error)>;
    using cancel_callback = std::function<bool(std::string & error)>;

    bool begin(
            common_runtime_operation operation,
            poll_callback poll,
            cancel_callback cancel,
            std::string & error) {
        if (operation.operation_id.empty()) {
            error = "operation id is required";
            return false;
        }
        if (!poll) {
            error = "operation poll callback is required";
            return false;
        }
        const std::string operation_id = operation.operation_id;
        std::lock_guard<std::mutex> lock(mutex_);
        if (operations_.find(operation_id) != operations_.end()) {
            error = "operation id is already registered: " + operation_id;
            return false;
        }
        operations_.emplace(
            operation_id,
            entry{std::move(operation), std::move(poll), std::move(cancel), {}});
        error.clear();
        return true;
    }

    bool poll(
            const std::string & operation_id,
            bool & ready,
            std::string & error) {
        ready = false;
        poll_callback callback;
        cancel_callback timeout_callback;
        bool timed_out = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = operations_.find(operation_id);
            if (it == operations_.end()) {
                error = "operation is unknown: " + operation_id;
                return false;
            }
            if (it->second.status.state != common_runtime_operation_state::running) {
                error = it->second.status.error;
                return it->second.status.state == common_runtime_operation_state::completed;
            }
            if (it->second.status.operation.deadline != std::chrono::steady_clock::time_point{} &&
                    std::chrono::steady_clock::now() >= it->second.status.operation.deadline) {
                it->second.status.state = common_runtime_operation_state::timed_out;
                it->second.status.error = "operation deadline exceeded";
                error = it->second.status.error;
                timeout_callback = it->second.cancel;
                timed_out = true;
            }
            if (!timed_out) {
                callback = it->second.poll;
            }
        }

        if (timed_out) {
            if (timeout_callback) {
                std::string ignored_error;
                timeout_callback(ignored_error);
            }
            return false;
        }

        if (!callback(ready, error)) {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = operations_.find(operation_id);
            if (it != operations_.end() &&
                    it->second.status.state == common_runtime_operation_state::running) {
                it->second.status.state = common_runtime_operation_state::failed;
                it->second.status.error = error;
            }
            return false;
        }
        if (ready) {
            complete(operation_id, {});
        }
        return true;
    }

    bool cancel(const std::string & operation_id, std::string & error) {
        cancel_callback callback;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = operations_.find(operation_id);
            if (it == operations_.end()) {
                error = "operation is unknown: " + operation_id;
                return false;
            }
            if (it->second.status.state != common_runtime_operation_state::running) {
                error = it->second.status.error;
                return it->second.status.state == common_runtime_operation_state::cancelled;
            }
            callback = it->second.cancel;
        }
        if (callback && !callback(error)) {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = operations_.find(operation_id);
            if (it != operations_.end()) {
                it->second.status.state = common_runtime_operation_state::failed;
                it->second.status.error = error;
            }
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = operations_.find(operation_id);
        if (it == operations_.end()) {
            error = "operation disappeared during cancellation: " + operation_id;
            return false;
        }
        if (it->second.status.state != common_runtime_operation_state::running) {
            error = it->second.status.error;
            return it->second.status.state == common_runtime_operation_state::cancelled;
        }
        it->second.status.state = common_runtime_operation_state::cancelled;
        it->second.status.error = error;
        return true;
    }

    bool complete(const std::string & operation_id, std::string error) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = operations_.find(operation_id);
        if (it == operations_.end()) return false;
        if (it->second.status.state != common_runtime_operation_state::running) {
            return false;
        }
        it->second.status.state = error.empty()
            ? common_runtime_operation_state::completed
            : common_runtime_operation_state::failed;
        it->second.status.error = std::move(error);
        return true;
    }

    bool describe(
            const std::string & operation_id,
            common_runtime_operation_status & out) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = operations_.find(operation_id);
        if (it == operations_.end()) return false;
        out = it->second.status;
        return true;
    }

    std::vector<common_runtime_operation_status> list() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<common_runtime_operation_status> result;
        result.reserve(operations_.size());
        for (const auto & item : operations_) result.push_back(item.second.status);
        return result;
    }

    size_t cleanup_terminal() {
        std::vector<entry> removed;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (auto it = operations_.begin(); it != operations_.end();) {
                if (it->second.status.state != common_runtime_operation_state::running) {
                    removed.push_back(std::move(it->second));
                    it = operations_.erase(it);
                } else {
                    ++it;
                }
            }
        }
        return removed.size();
    }

private:
    struct entry {
        common_runtime_operation_status status;
        poll_callback poll;
        cancel_callback cancel;
        std::string reserved;

        entry(
                common_runtime_operation operation,
                poll_callback poll_callback_value,
                cancel_callback cancel_callback_value,
                std::string reserved_value)
            : status{std::move(operation), common_runtime_operation_state::running, {}}
            , poll(std::move(poll_callback_value))
            , cancel(std::move(cancel_callback_value))
            , reserved(std::move(reserved_value)) {}
    };

    mutable std::mutex mutex_;
    std::map<std::string, entry> operations_;
};
