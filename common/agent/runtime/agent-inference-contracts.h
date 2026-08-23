// Host-neutral model and inference configuration.
//
// CLI, daemon and Android adapters may translate their platform-specific
// configuration into these values.  Keep this header free of CLI argument
// parsing, filesystem policy and Android/JNI types.
#pragma once

#include <cstddef>
#include <string>

// These names describe runtime providers, not user-interface entrypoints.
// `cli` is retained for compatibility with the existing direct llama.cpp
// provider; a future rename can be handled at the adapter boundary.
enum class agent_inference_backend {
    cli,
    server_context,
};

struct common_agent_inference_options {
    // The host resolves this to an accessible model location before creating
    // a session.  Android may therefore populate it from app-private storage.
    std::string model;
    int n_predict = -1;
    int n_gpu_layers = 0;
    bool fit_params = true;
    int n_threads = 2;
    size_t context_size_tokens = 0;
    std::string mmproj;
};
