#include "agent/adaptation/flydelta/flydelta-layer-discovery.h"

#include <algorithm>
#include <cmath>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

static common_flydelta_hidden_state_capture capture(
        bool positive, float late_signal, int32_t token_index) {
    common_flydelta_hidden_state_capture value;
    value.captured = true;
    value.model_profile_fingerprint = "model:qwen:test";
    value.capture_layout_revision = "layer-input:generation-boundary:v1";
    value.layer_indices = {1, 2, 3, 4, 5, 6};
    value.n_embd = 2;
    value.position = common_flydelta_capture_position::generation_boundary;
    value.token_index = token_index;
    for (size_t layer = 0; layer < value.layer_indices.size(); ++layer) {
        const float signal = layer == 4 ? late_signal : (positive ? 0.1f : 0.0f);
        value.values.push_back(signal);
        value.values.push_back(0.0f);
    }
    return value;
}

int main() {
    std::string error;
    common_flydelta_layer_discovery_config config;
    config.max_anchors = 4;
    config.min_anchor_separation = 2;
    std::vector<common_flydelta_hidden_state_capture> positive = {
        capture(true, 3.0f, 4), capture(true, 3.2f, 5)};
    std::vector<common_flydelta_hidden_state_capture> negative = {
        capture(false, 0.0f, 2), capture(false, 0.1f, 3)};

    common_flydelta_layer_discovery_result result;
    CHECK(common_flydelta_discover_layer_regions(
        positive, negative, config, result, error));
    CHECK(result.positive_samples == 2 && result.negative_samples == 2);
    CHECK(result.scores.size() == 6);
    CHECK(result.scores[4].layer_index == 5);
    CHECK(result.scores[4].separation > result.scores[0].separation);
    CHECK(result.scores[4].fisher_score > result.scores[0].fisher_score);
    CHECK(result.anchor_layers.size() == 4);
    CHECK(std::find(result.anchor_layers.begin(), result.anchor_layers.end(), 5) !=
        result.anchor_layers.end());
    CHECK(common_flydelta_layer_discovery_result_validate(result, config, error));

    positive[1].capture_layout_revision = "different-layout";
    CHECK(!common_flydelta_discover_layer_regions(
        positive, negative, config, result, error));
    return 0;
}
