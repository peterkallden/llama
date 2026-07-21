#include "agent-inference-capacity-gate.h"

#include <algorithm>
#include <limits>
#include <utility>

common_agent_inference_capacity_gate::common_agent_inference_capacity_gate(size_t capacity)
    : capacity_(std::max<size_t>(1, capacity)) {}

bool common_agent_inference_capacity_gate::try_acquire() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_ >= capacity_) return false;
    ++active_;
    return true;
}

void common_agent_inference_capacity_gate::release() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_ > 0) --active_;
}

bool common_agent_inference_capacity_gate::enqueue(
        common_agent_inference_wait_request request,
        std::string & error) {
    if (request.waiter_id.empty()) {
        error = "inference waiter id is required";
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (active_waiters_.find(request.waiter_id) != active_waiters_.end() ||
            std::any_of(waiters_.begin(), waiters_.end(), [&](const waiter & item) {
                return item.request.waiter_id == request.waiter_id;
            })) {
        error = "inference waiter id is already registered: " + request.waiter_id;
        return false;
    }

    waiters_.push_back(waiter{
        std::move(request),
        next_sequence_++,
        std::chrono::steady_clock::now(),
    });
    error.clear();
    return true;
}

std::vector<common_agent_inference_capacity_gate::waiter>::iterator
common_agent_inference_capacity_gate::select_next_waiter() {
    const auto now = std::chrono::steady_clock::now();
    auto best = waiters_.end();
    int best_score = std::numeric_limits<int>::min();
    uint64_t best_sequence = std::numeric_limits<uint64_t>::max();

    for (auto it = waiters_.begin(); it != waiters_.end();) {
        if (it->request.deadline != std::chrono::steady_clock::time_point{} &&
                now >= it->request.deadline) {
            it = waiters_.erase(it);
            continue;
        }

        const int base_score = static_cast<int>(it->request.priority);
        const auto waited = std::chrono::duration_cast<std::chrono::seconds>(
            now - it->enqueued_at).count();
        const int aging_score = static_cast<int>(std::min<int64_t>(2, waited / 5));
        const int score = std::min(2, base_score + aging_score);
        if (score > best_score || (score == best_score && it->sequence < best_sequence)) {
            best = it;
            best_score = score;
            best_sequence = it->sequence;
        }
        ++it;
    }
    return best;
}

bool common_agent_inference_capacity_gate::try_acquire(const std::string & waiter_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_waiters_.find(waiter_id) != active_waiters_.end()) return true;
    if (active_ >= capacity_) return false;

    const auto next = select_next_waiter();
    if (next == waiters_.end() || next->request.waiter_id != waiter_id) return false;

    active_waiters_.insert(waiter_id);
    waiters_.erase(next);
    ++active_;
    return true;
}

bool common_agent_inference_capacity_gate::cancel(
        const std::string & waiter_id,
        std::string & error) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = std::find_if(waiters_.begin(), waiters_.end(), [&](const waiter & item) {
        return item.request.waiter_id == waiter_id;
    });
    if (it == waiters_.end()) {
        error = "inference waiter is not queued: " + waiter_id;
        return false;
    }
    waiters_.erase(it);
    error.clear();
    return true;
}

void common_agent_inference_capacity_gate::release(const std::string & waiter_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_waiters_.erase(waiter_id) > 0 && active_ > 0) --active_;
}

size_t common_agent_inference_capacity_gate::capacity() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return capacity_;
}

size_t common_agent_inference_capacity_gate::active() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return active_;
}

size_t common_agent_inference_capacity_gate::waiting() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return waiters_.size();
}
