#pragma once

#include "agent/adaptation/flydelta/flydelta-hidden-state-hook.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Host-side, capture-only discovery of where a model-facing distinction is
// represented. It does not run inference, compose an overlay or classify a
// host outcome.
struct common_flydelta_layer_discovery_config {
    int schema_version = 1;
    size_t max_samples_per_class = 32;
    size_t max_layers = 64;
    size_t max_anchors = 6;
    uint32_t min_anchor_separation = 2;
    float variance_ridge = 0.001f;
    size_t max_capture_bytes = 4U * 1024U * 1024U;
};

struct common_flydelta_layer_score {
    uint32_t layer_index = 0;
    float separation = 0.0f;
    float projected_variance = 0.0f;
    float fisher_score = 0.0f;
    float slope = 0.0f;
    std::vector<float> direction;
};

struct common_flydelta_layer_discovery_result {
    int schema_version = 1;
    size_t positive_samples = 0;
    size_t negative_samples = 0;
    uint32_t n_embd = 0;
    std::vector<common_flydelta_layer_score> scores;
    std::vector<uint32_t> anchor_layers;
};

bool common_flydelta_layer_discovery_config_validate(
        const common_flydelta_layer_discovery_config & config,
        std::string & error);

bool common_flydelta_layer_discovery_result_validate(
        const common_flydelta_layer_discovery_result & result,
        const common_flydelta_layer_discovery_config & config,
        std::string & error);

// Computes a dense layer profile from matched positive/negative generation-
// boundary captures. The output contains a normalized mean-difference
// direction per layer, separation, projected within-class variance and local
// slope. Anchor selection is signal-driven and only chooses bounded layer
// indices for the existing intervention search.
bool common_flydelta_discover_layer_regions(
        const std::vector<common_flydelta_hidden_state_capture> & positive,
        const std::vector<common_flydelta_hidden_state_capture> & negative,
        const common_flydelta_layer_discovery_config & config,
        common_flydelta_layer_discovery_result & result,
        std::string & error);
