#pragma once

// Exact decoded D2 candidate ranking in activation space.
//
// The ranker is deliberately downstream of GPU proposal and CPU astcenc
// finishing. It reconstructs both paired logical rows using each D2 record's
// layout/semantic contract, so physical RGBA or proposal-error rankings never
// become a hidden neural objective.

#include "astc-gpu-d2-candidates.h"
#include "astc-gpu-encoder-finisher.h"

#include <cstdint>
#include <vector>

struct astc_gpu_d2_activation_rank_request {
    uint32_t tensor_width = 0;
    uint32_t tensor_height = 0;
    uint32_t source_blocks_x = 0;
    // Row-major [sample][tensor_width].
    std::vector<float> activations;
};

struct astc_gpu_d2_finished_candidate_score {
    uint32_t candidate_source_block_id = 0;
    uint32_t logical_source_block_id = 0;
    astc_gpu_d2_candidate_family family = astc_gpu_d2_candidate_family::direct_neutral;
    astc_vulkan_paired_layout layout = astc_vulkan_paired_layout::rg_b;
    astc_vulkan_paired_semantic semantic = astc_vulkan_paired_semantic::direct_rgb;
    double activation_error = 0.0;
};

// Direct-neutral is mandatory per logical block; remaining candidates are
// ranked by exact decoded activation error and retained up to the budget.
bool astc_gpu_d2_rank_finished_candidates_activation(
    const astc_gpu_d2_candidate_bank & bank,
    const std::vector<astc_gpu_encoder_finished_block> & finished,
    const astc_gpu_d2_activation_rank_request & request,
    uint32_t max_candidates_per_logical,
    std::vector<astc_gpu_encoder_finished_block> & selected,
    std::vector<astc_gpu_d2_finished_candidate_score> & scores);
