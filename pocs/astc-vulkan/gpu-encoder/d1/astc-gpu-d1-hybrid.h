#pragma once

// D1 frontend hand-off for the shared offline hybrid seam. D1 supplies the
// scalar/gauge decoded-weight interpretation; physical proposal, legal finish,
// payload verification, and global selector remain shared contracts.

#include "astc-gpu-d1-candidates.h"
#include "astc-gpu-d1-neural-rank.h"
#include "astc-gpu-encoder-finisher.h"
#include "astc-gpu-selector-adapter.h"

#include <cstdint>
#include <string>
#include <vector>

using astc_gpu_d1_selector_delta_request = astc_gpu_selector_delta_request;

struct astc_gpu_d1_hybrid_options {
    // Bound CPU astcenc work after the physical proposer. Scalar candidate
    // zero remains mandatory for every logical block.
    uint32_t max_finish_candidates_per_logical = 2;
    // Bound the exact decoded activation bank passed to a later global D1
    // selector. It may be smaller than the finisher budget.
    uint32_t max_ranked_candidates_per_logical = 2;
    float astcenc_quality = ASTCENC_PRE_FAST;
    // Quality mode always retains a separately encoded scalar thorough
    // reference; speed mode retains the historical bounded CPU finish path.
    astc_gpu_encoder_candidate_profile profile = astc_gpu_encoder_candidate_profile::speed;
    float reference_astcenc_quality = ASTCENC_PRE_THOROUGH;
    uint32_t worker_count = 1;
};

struct astc_gpu_d1_hybrid_result {
    std::vector<astc_gpu_encoder_proposal> retained_proposals;
    std::vector<astc_gpu_encoder_finished_block> finished_blocks;
    std::vector<astc_gpu_encoder_finished_block> ranked_blocks;
    std::vector<astc_gpu_d1_finished_candidate_score> activation_scores;
};

// Retains a bounded number of proposals per logical D1 block and sends them
// through the exact CPU/libastc finisher. In neural_quality profile, scalar
// is separately encoded at reference quality as a mandatory candidate. This
// is the reusable proposer -> finisher seam; no global selection is implied.
bool astc_gpu_d1_finish_candidate_bank(
    const astc_gpu_d1_candidate_bank & bank,
    const std::vector<astc_gpu_encoder_proposal> & proposals,
    const astc_gpu_d1_hybrid_options & options,
    astc_gpu_d1_hybrid_result & result,
    std::string & error);

// Runs the D1 CPU half over proposals from either Vulkan or the deterministic
// CPU proposer. Candidate retention is bounded before astcenc finish.
bool astc_gpu_d1_finish_and_rank(
    const astc_gpu_d1_candidate_bank & bank,
    const std::vector<astc_gpu_encoder_proposal> & proposals,
    const astc_gpu_d1_activation_rank_request & rank_request,
    const astc_gpu_d1_hybrid_options & options,
    astc_gpu_d1_hybrid_result & result,
    std::string & error);

// Converts exact decoded D1 candidates to the common global selector format.
// Candidate zero in every physical block is the scalar decoded baseline.
bool astc_gpu_d1_make_selector_candidates(
    const astc_gpu_d1_candidate_bank & bank,
    const std::vector<astc_gpu_encoder_finished_block> & finished,
    const astc_gpu_d1_selector_delta_request & request,
    std::vector<std::vector<astc_vulkan_paired_candidate_delta>> & candidates,
    std::string & error);
