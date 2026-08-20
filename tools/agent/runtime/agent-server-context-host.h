#pragma once

#include "../runtime/agent-runtime-assembly.h"

#include "common.h"
#include "log.h"

#include <memory>
#include <string>
#include <thread>

struct server_context;

struct common_agent_server_context_load_key {
    std::string model;
    int n_gpu_layers = 0;
    bool fit_params = true;
    std::string mmproj;
};

struct common_agent_server_context_context_key {
    common_agent_server_context_load_key load_key;
    int n_parallel = 1;
    int n_sequences = 1;
    int n_ctx = 0;
    int n_threads = 2;
};

struct common_agent_server_context_host_config {
    common_agent_server_context_context_key context_key;
    int verbosity = LOG_LEVEL_WARN;
};

struct common_agent_server_context_running_instance {
    std::unique_ptr<server_context> server;
    common_params params;
    std::unique_ptr<std::thread> loop;
    bool running = false;
};

common_agent_server_context_load_key make_agent_server_context_load_key(
    const common_agent_inference_options & options);

common_agent_server_context_context_key make_agent_server_context_context_key(
    const common_agent_inference_options & options);

common_agent_server_context_host_config make_agent_server_context_host_config(
    const common_agent_inference_options & options);

common_params make_agent_server_context_params(
    const common_agent_server_context_host_config & config);

common_params make_agent_server_context_params(
    const common_agent_inference_options & options);

class common_agent_server_context_host {
public:
    common_agent_server_context_host();
    ~common_agent_server_context_host();

    bool start(const common_agent_server_context_host_config & config, std::string & error);
    bool build_inference_session(common_agent_inference_session & session, std::string & error) const;
    void stop();

    const common_agent_server_context_load_key & load_key() const { return current_load_key; }
    const common_agent_server_context_context_key & context_key() const { return current_context_key; }
    const common_agent_server_context_host_config & config() const { return current_config; }
    const common_params & params() const { return instance->params; }
    server_context & server();
    const server_context & server() const;

private:
    common_agent_server_context_load_key current_load_key;
    common_agent_server_context_context_key current_context_key;
    common_agent_server_context_host_config current_config;
    std::unique_ptr<common_agent_server_context_running_instance> instance;
};
