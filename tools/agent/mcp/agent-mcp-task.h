#pragma once

#include "agent-mcp-server-protocol.h"

#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

enum class agent_mcp_task_status {
    working,
    completed,
    failed,
    cancelled,
};

const char * agent_mcp_task_status_name(agent_mcp_task_status status);

struct agent_mcp_task_snapshot {
    std::string task_id;
    agent_mcp_task_status status = agent_mcp_task_status::working;
    std::string status_message;
    std::string created_at;
    std::string last_updated_at;
    uint64_t ttl_ms = 60000;
    uint64_t poll_interval_ms = 250;
    agent_mcp_json result;
};

class agent_mcp_task_store {
public:
    using work = std::function<agent_mcp_json()>;

    agent_mcp_task_store(uint64_t default_ttl_ms, uint64_t default_poll_interval_ms);
    ~agent_mcp_task_store();

    std::string create(work operation, uint64_t requested_ttl_ms);
    bool get(const std::string & task_id, agent_mcp_task_snapshot & snapshot) const;
    bool wait_result(const std::string & task_id, agent_mcp_task_snapshot & snapshot);
    bool cancel(const std::string & task_id);
    std::vector<agent_mcp_task_snapshot> list() const;

private:
    struct entry {
        agent_mcp_task_snapshot snapshot;
        work operation;
        std::thread worker;
        bool cancellation_requested = false;
    };

    static std::string now_iso8601();
    void run(const std::shared_ptr<entry> & task);

    uint64_t default_ttl_ms_;
    uint64_t default_poll_interval_ms_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::unordered_map<std::string, std::shared_ptr<entry>> tasks_;
    uint64_t next_task_id_ = 1;
};

agent_mcp_json agent_mcp_render_task_snapshot(const agent_mcp_task_snapshot & snapshot);
