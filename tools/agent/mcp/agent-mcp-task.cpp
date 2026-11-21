#include "agent-mcp-task.h"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

const char * agent_mcp_task_status_name(agent_mcp_task_status status) {
    switch (status) {
        case agent_mcp_task_status::working: return "working";
        case agent_mcp_task_status::completed: return "completed";
        case agent_mcp_task_status::failed: return "failed";
        case agent_mcp_task_status::cancelled: return "cancelled";
    }
    return "failed";
}

std::string agent_mcp_task_store::now_iso8601() {
    const auto now = std::chrono::system_clock::now();
    const auto time = std::chrono::system_clock::to_time_t(now);
    std::tm utc{};
#ifdef _WIN32
    gmtime_s(&utc, &time);
#else
    gmtime_r(&time, &utc);
#endif
    std::ostringstream out;
    out << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return out.str();
}

agent_mcp_task_store::agent_mcp_task_store(
        uint64_t default_ttl_ms,
        uint64_t default_poll_interval_ms)
    : default_ttl_ms_(default_ttl_ms)
    , default_poll_interval_ms_(default_poll_interval_ms) {}

agent_mcp_task_store::~agent_mcp_task_store() {
    std::vector<std::shared_ptr<entry>> tasks;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto & [id, task] : tasks_) {
            (void) id;
            task->cancellation_requested = true;
            tasks.push_back(task);
        }
    }
    condition_.notify_all();
    for (const auto & task : tasks) {
        if (task->worker.joinable()) task->worker.join();
    }
}

std::string agent_mcp_task_store::create(work operation, uint64_t requested_ttl_ms) {
    auto task = std::make_shared<entry>();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        task->snapshot.task_id = "task-" + std::to_string(next_task_id_++);
        task->snapshot.created_at = now_iso8601();
        task->snapshot.last_updated_at = task->snapshot.created_at;
        task->snapshot.ttl_ms = requested_ttl_ms == 0 ? default_ttl_ms_ : requested_ttl_ms;
        task->snapshot.poll_interval_ms = default_poll_interval_ms_;
        task->operation = std::move(operation);
        tasks_.emplace(task->snapshot.task_id, task);
    }
    task->worker = std::thread([this, task]() { run(task); });
    return task->snapshot.task_id;
}

void agent_mcp_task_store::run(const std::shared_ptr<entry> & task) {
    agent_mcp_json result;
    try {
        result = task->operation();
    } catch (const std::exception & exception) {
        result = agent_mcp_json{{"isError", true}, {"error", exception.what()}};
    } catch (...) {
        result = agent_mcp_json{{"isError", true}, {"error", "task operation failed"}};
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        task->snapshot.last_updated_at = now_iso8601();
        if (task->cancellation_requested) {
            task->snapshot.status = agent_mcp_task_status::cancelled;
            task->snapshot.status_message = "task cancelled";
        } else if (result.value("isError", false)) {
            task->snapshot.status = agent_mcp_task_status::failed;
            task->snapshot.status_message = result.value("error", "task operation failed");
            task->snapshot.result = std::move(result);
        } else {
            task->snapshot.status = agent_mcp_task_status::completed;
            task->snapshot.result = std::move(result);
        }
    }
    condition_.notify_all();
}

bool agent_mcp_task_store::get(const std::string & task_id, agent_mcp_task_snapshot & snapshot) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = tasks_.find(task_id);
    if (it == tasks_.end()) return false;
    snapshot = it->second->snapshot;
    return true;
}

bool agent_mcp_task_store::wait_result(const std::string & task_id, agent_mcp_task_snapshot & snapshot) {
    std::unique_lock<std::mutex> lock(mutex_);
    const auto it = tasks_.find(task_id);
    if (it == tasks_.end()) return false;
    const auto task = it->second;
    condition_.wait(lock, [&task]() {
        return task->snapshot.status != agent_mcp_task_status::working;
    });
    snapshot = task->snapshot;
    return true;
}

bool agent_mcp_task_store::cancel(const std::string & task_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = tasks_.find(task_id);
    if (it == tasks_.end()) return false;
    if (it->second->snapshot.status != agent_mcp_task_status::working) return true;
    it->second->cancellation_requested = true;
    it->second->snapshot.status = agent_mcp_task_status::cancelled;
    it->second->snapshot.status_message = "task cancelled";
    it->second->snapshot.last_updated_at = now_iso8601();
    condition_.notify_all();
    return true;
}

std::vector<agent_mcp_task_snapshot> agent_mcp_task_store::list() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<agent_mcp_task_snapshot> result;
    for (const auto & [id, task] : tasks_) {
        (void) id;
        result.push_back(task->snapshot);
    }
    return result;
}

agent_mcp_json agent_mcp_render_task_snapshot(const agent_mcp_task_snapshot & snapshot) {
    agent_mcp_json result = {
        {"taskId", snapshot.task_id},
        {"status", agent_mcp_task_status_name(snapshot.status)},
        {"createdAt", snapshot.created_at},
        {"lastUpdatedAt", snapshot.last_updated_at},
        {"ttl", snapshot.ttl_ms},
        {"pollInterval", snapshot.poll_interval_ms},
    };
    if (!snapshot.status_message.empty()) result["statusMessage"] = snapshot.status_message;
    return result;
}
