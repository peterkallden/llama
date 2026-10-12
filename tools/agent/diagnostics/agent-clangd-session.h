#pragma once

#include "agent-clangd-protocol.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

struct agent_clangd_session_config {
    std::string executable = "clangd";
    std::string repository_root;
    std::string compile_commands_dir;
    uint32_t request_timeout_ms = 10000;
    uint32_t shutdown_timeout_ms = 1000;
};

class agent_clangd_session {
public:
    explicit agent_clangd_session(agent_clangd_session_config config);
    ~agent_clangd_session();

    bool request(
            const std::string & method,
            const agent_clangd_json & params,
            agent_clangd_json & response,
            std::string & error);

    bool initialized() const;

private:
    struct impl;
    bool ensure_started(std::string & error);
    bool send_notification(const std::string & method, const agent_clangd_json & params, std::string & error);
    bool send_request(const std::string & method, const agent_clangd_json & params, agent_clangd_json & response, std::string & error);
    bool read_message(agent_clangd_json & message, std::string & error);
    void shutdown();

    agent_clangd_session_config config_;
    std::unique_ptr<impl> state_;
    mutable std::mutex mutex_;
};
