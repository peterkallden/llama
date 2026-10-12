#include "agent/adaptation/flydelta/flydelta-hidden-state-hook.h"

#include <limits>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

int main() {
    std::string error;
    common_flydelta_hidden_state_capture_request request;
    request.enabled = true;
    request.layer_indices = {1, 4};
    request.max_bytes = 4096;
    request.model_profile_fingerprint = "model:qwen:v1";
    request.capture_layout_revision = "layer-input:v1";
    CHECK(common_flydelta_hidden_state_capture_request_validate(request, 8, 4096, error));
    CHECK(common_flydelta_hidden_state_capture_architecture_supported("qwen2"));
    CHECK(!common_flydelta_hidden_state_capture_architecture_supported("unknown"));

    request.layer_indices = {4, 1};
    CHECK(!common_flydelta_hidden_state_capture_request_validate(request, 8, 4096, error));
    request.layer_indices = {1, 4};
    request.layer_indices = {1, 8};
    CHECK(!common_flydelta_hidden_state_capture_request_validate(request, 8, 4096, error));
    request.layer_indices = {1, 4};
    request.max_bytes = 8192;
    CHECK(!common_flydelta_hidden_state_capture_request_validate(request, 8, 4096, error));
    request.max_bytes = 4096;
    request.layer_indices.clear();
    for (uint32_t layer = 0; layer < 16; ++layer) request.layer_indices.push_back(layer);
    CHECK(common_flydelta_hidden_state_capture_request_validate(request, 16, 4096, error));
    request.layer_indices = {1, 4};

    common_flydelta_hidden_state_capture capture;
    capture.captured = true;
    capture.model_profile_fingerprint = request.model_profile_fingerprint;
    capture.capture_layout_revision = request.capture_layout_revision;
    capture.layer_indices = request.layer_indices;
    capture.n_embd = 2;
    capture.token_index = 3;
    capture.values = {1.0f, -1.0f, 0.25f, 0.5f};
    CHECK(common_flydelta_hidden_state_capture_validate(capture, 4096, error));

    capture.values[0] = std::numeric_limits<float>::quiet_NaN();
    CHECK(!common_flydelta_hidden_state_capture_validate(capture, 4096, error));

    capture = {};
    CHECK(common_flydelta_hidden_state_capture_validate(capture, 4096, error));
    capture.values = {1.0f};
    CHECK(!common_flydelta_hidden_state_capture_validate(capture, 4096, error));
    return 0;
}
