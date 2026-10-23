#pragma once

#include "agent-daemon-adapter.h"

#include <algorithm>

// Project daemon options into the shared resident request without overriding
// the host's automatic model-fitting policy through positional arguments.
inline common_agent_runtime_resident_request_config make_daemon_resident_model_config(
        const daemon_options & options) {
    common_agent_runtime_resident_request_config config;
    config.model = options.model;
    config.n_predict = options.n_predict;
    config.n_gpu_layers = options.n_gpu_layers;
    config.fit_params = true;
    config.inference_backend = "server-context";
    config.n_threads = options.n_threads;
    config.context_size_tokens = static_cast<size_t>(std::max(0, options.context_size));
    config.mmproj = options.mmproj;
    return config;
}
