#pragma once

#include "agent-runtime-assembly.h"

#include "common.h"

#include <memory>
#include <string>
#include <thread>

struct server_context;

struct common_agent_server_context_load_key {
    std::string model;
    int n_gpu_layers = 0;
    bool fit_params = true;
};

common_agent_server_context_load_key make_agent_server_context_load_key(
    const common_agent_inference_options & options);

common_params make_agent_server_context_load_params(
    const common_agent_inference_options & options);

class common_agent_server_context_host {
public:
    common_agent_server_context_host();
    ~common_agent_server_context_host();

    bool start(const common_agent_inference_options & options, std::string & error);
    void stop();

    const common_agent_server_context_load_key & load_key() const { return current_load_key; }
    const common_params & params() const { return params_base; }
    server_context & server();
    const server_context & server() const;

private:
    std::unique_ptr<server_context> server_ptr;
    common_agent_server_context_load_key current_load_key;
    common_params params_base;
    std::unique_ptr<std::thread> loop;
    bool running = false;
};

bool make_server_context_inference_session(
    const common_agent_inference_options & options,
    common_agent_inference_session & session,
    std::string & error);
