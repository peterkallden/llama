#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Agent-owned wrapper around llama.cpp's staging layer-input extraction API.
// This is deliberately capture-only: it never changes the graph or injects
// into the context that produced the sample.
struct common_flydelta_hidden_state_capture_request {
    int schema_version = 1;
    bool enabled = false;
    std::vector<uint32_t> layer_indices;
    // -1 selects the last prompt row. Positive values are prompt-row indices.
    int32_t token_index = -1;
    size_t max_bytes = 4U * 1024U * 1024U;
    std::string model_profile_fingerprint;
    std::string capture_layout_revision;
};

struct common_flydelta_hidden_state_capture {
    int schema_version = 1;
    bool captured = false;
    std::string model_profile_fingerprint;
    std::string capture_layout_revision;
    std::vector<uint32_t> layer_indices;
    uint32_t n_embd = 0;
    int32_t token_index = -1;
    std::vector<float> values; // layer-major: layer 0 row, layer 1 row, ...
    std::string failure_reason;
};

bool common_flydelta_hidden_state_capture_request_validate(
        const common_flydelta_hidden_state_capture_request & request,
        size_t model_n_layers,
        size_t max_bytes,
        std::string & error);

bool common_flydelta_hidden_state_capture_validate(
        const common_flydelta_hidden_state_capture & capture,
        size_t max_bytes,
        std::string & error);

