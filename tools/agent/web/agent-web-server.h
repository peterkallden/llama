#pragma once

#include <cstddef>
#include <string>

struct agent_web_server_options {
    std::string listen_address = "127.0.0.1";
    int port = 8090;
    std::string daemon_address = "127.0.0.1";
    int daemon_port = 0;
    std::string web_bearer_token;
    std::string daemon_authorization;
    std::string allowed_origin;
    size_t max_body_bytes = 1024 * 1024;
};

bool run_agent_web_server(
        const agent_web_server_options & options,
        std::string & error);

