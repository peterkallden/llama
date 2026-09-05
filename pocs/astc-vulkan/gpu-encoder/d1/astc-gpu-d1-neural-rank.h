#pragma once

// Exact decoded D1 candidate ranking in activation space.
//
// This stage deliberately runs after the CPU astcenc finisher. It therefore
// scores the legal decoded ASTC candidate, never its source RGBA field or a
// GPU proposer surrogate. It is a local bounded-ranking primitive; global
// conflict-aware selection remains a later layer above this API.

#include "astc-gpu-d1-candidates.h"
#include "astc-gpu-encoder-finisher.h"

#include <cstdint>
#include <vector>

struct astc_gpu_d1_activation_rank_request {
    uint32_t tensor_width = 0;
    uint32_t tensor_height = 0;
    uint32_t source_blocks_x = 0;
    // Row-major [sample][tensor_width]. The source builder's raster block IDs
    // define the mapping back into this logical tensor.
    std::vector<float> activations;
};

struct astc_gpu_d1_finished_candidate_score {
    uint32_t candidate_source_block_id = 0;
    uint32_t logical_source_block_id = 0;
    astc_gpu_d1_candidate_family family = astc_gpu_d1_candidate_family::scalar;
    double activation_error = 0.0;
};

// Scores finished candidates exactly after ASTC decode. Scalar is mandatory in
// the returned set; remaining slots per logical block are sorted by decoded
// activation error. The caller can retain more than one alternative for a
// later global conflict-aware selector.
bool astc_gpu_d1_rank_finished_candidates_activation(
    const astc_gpu_d1_candidate_bank & bank,
    const std::vector<astc_gpu_encoder_finished_block> & finished,
    const astc_gpu_d1_activation_rank_request & request,
    uint32_t max_candidates_per_logical,
    std::vector<astc_gpu_encoder_finished_block> & selected,
    std::vector<astc_gpu_d1_finished_candidate_score> & scores);
