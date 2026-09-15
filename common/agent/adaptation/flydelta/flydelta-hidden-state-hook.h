#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

// Agent-owned wrapper around llama.cpp's staging layer-input extraction API.
// This is deliberately capture-only: it never changes the graph or injects
// into the context that produced the sample.
enum class common_flydelta_capture_position {
    // token_index is an absolute prompt row and must align exactly.
    prompt_row,
    // The final prompt row, identified semantically as the state immediately
    // before generation. Repair prompts may have different absolute lengths.
    generation_boundary,
};

const char * common_flydelta_capture_position_name(
        common_flydelta_capture_position position);

struct common_flydelta_hidden_state_capture_request {
    int schema_version = 1;
    bool enabled = false;
    std::vector<uint32_t> layer_indices;
    // -1 selects the last prompt row. Positive values are prompt-row indices.
    int32_t token_index = -1;
    common_flydelta_capture_position position = common_flydelta_capture_position::prompt_row;
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
    common_flydelta_capture_position position = common_flydelta_capture_position::prompt_row;
    int32_t token_index = -1;
    std::vector<float> values; // layer-major: layer 0 row, layer 1 row, ...
    std::string failure_reason;
};

// The llama.cpp staging API is only safe for graph implementations that
// publish layer inputs. Unknown architectures must fail closed before graph
// reservation rather than reaching an internal assertion.
bool common_flydelta_hidden_state_capture_architecture_supported(
        std::string_view architecture);

bool common_flydelta_hidden_state_capture_request_validate(
        const common_flydelta_hidden_state_capture_request & request,
        size_t model_n_layers,
        size_t max_bytes,
        std::string & error);

bool common_flydelta_hidden_state_capture_validate(
        const common_flydelta_hidden_state_capture & capture,
        size_t max_bytes,
        std::string & error);
