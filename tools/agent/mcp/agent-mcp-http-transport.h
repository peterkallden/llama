#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>

struct agent_mcp_http_request {
    std::string url;
    std::map<std::string, std::string> headers;
    std::string body;
    uint32_t connect_timeout_ms = 5000;
    uint32_t request_timeout_ms = 30000;
    size_t max_result_bytes = 1024 * 1024;
    std::optional<std::chrono::steady_clock::time_point> deadline;
};

struct agent_mcp_http_response {
    int status = 0;
    std::map<std::string, std::string> headers;
    std::string body;
};

class agent_mcp_http_transport {
public:
    virtual ~agent_mcp_http_transport() = default;

    virtual bool post(
        const agent_mcp_http_request & request,
        agent_mcp_http_response & response,
        std::string & error) = 0;
};

std::shared_ptr<agent_mcp_http_transport> make_agent_mcp_httplib_transport();
