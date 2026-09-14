#include "agent/adaptation/flydelta/flydelta-hidden-state-hook.h"

#include <cmath>

namespace {

constexpr size_t MAX_CAPTURE_LAYERS = 8;
constexpr size_t MAX_STRING_SIZE = 512;

bool nonempty_bounded(const std::string & value) {
    return !value.empty() && value.size() <= MAX_STRING_SIZE;
}

} // namespace

bool common_flydelta_hidden_state_capture_architecture_supported(
        std::string_view architecture) {
    // Keep this allow-list aligned with llama.cpp graph implementations that
    // assign llm_graph_result::t_layer_inp. Unknown architectures fail closed.
    static constexpr std::string_view supported[] = {
        "deepseek4", "gemma3n", "gemma4", "lfm2", "llama",
        "minimax-m2", "muse-glimmer", "nemotron-h", "openai-moe",
        "qwen2", "qwen3", "qwen3moe", "qwen3next", "qwen35",
        "qwen35moe",
    };
    for (const std::string_view candidate : supported) {
        if (candidate == architecture) return true;
    }
    return false;
}

bool common_flydelta_hidden_state_capture_request_validate(
        const common_flydelta_hidden_state_capture_request & request,
        size_t model_n_layers,
        size_t max_bytes,
        std::string & error) {
    error.clear();
    if (request.schema_version != 1) {
        error = "unsupported FlyDelta hidden-state capture schema";
        return false;
    }
    if (!request.enabled) {
        return true;
    }
    if (model_n_layers == 0 || request.layer_indices.empty() ||
            request.layer_indices.size() > MAX_CAPTURE_LAYERS) {
        error = "FlyDelta hidden-state capture layer selection is invalid";
        return false;
    }
    for (const uint32_t layer : request.layer_indices) {
        if (layer >= model_n_layers) {
            error = "FlyDelta hidden-state capture layer is out of range";
            return false;
        }
    }
    for (size_t i = 1; i < request.layer_indices.size(); ++i) {
        if (request.layer_indices[i - 1] >= request.layer_indices[i]) {
            error = "FlyDelta hidden-state capture layers must be sorted and unique";
            return false;
        }
    }
    if (request.token_index < -1) {
        error = "FlyDelta hidden-state capture token index is invalid";
        return false;
    }
    if (request.max_bytes == 0 || request.max_bytes > max_bytes) {
        error = "FlyDelta hidden-state capture byte bound is invalid";
        return false;
    }
    if (!nonempty_bounded(request.model_profile_fingerprint) ||
            !nonempty_bounded(request.capture_layout_revision)) {
        error = "FlyDelta hidden-state capture identity is incomplete";
        return false;
    }
    return true;
}

bool common_flydelta_hidden_state_capture_validate(
        const common_flydelta_hidden_state_capture & capture,
        size_t max_bytes,
        std::string & error) {
    error.clear();
    if (capture.schema_version != 1) {
        error = "unsupported FlyDelta hidden-state capture schema";
        return false;
    }
    if (!capture.captured) {
        if (!capture.values.empty()) {
            error = "uncaptured FlyDelta hidden-state result contains values";
            return false;
        }
        return true;
    }
    if (capture.layer_indices.empty() || capture.n_embd == 0 ||
            capture.token_index < 0 ||
            capture.values.size() != capture.layer_indices.size() * capture.n_embd ||
            capture.values.size() * sizeof(float) > max_bytes ||
            capture.model_profile_fingerprint.empty() ||
            capture.capture_layout_revision.empty()) {
        error = "FlyDelta hidden-state capture result is invalid";
        return false;
    }
    for (size_t i = 1; i < capture.layer_indices.size(); ++i) {
        if (capture.layer_indices[i - 1] >= capture.layer_indices[i]) {
            error = "FlyDelta hidden-state capture result layers are not sorted";
            return false;
        }
    }
    for (const float value : capture.values) {
        if (!std::isfinite(value)) {
            error = "FlyDelta hidden-state capture contains a non-finite value";
            return false;
        }
    }
    return true;
}
