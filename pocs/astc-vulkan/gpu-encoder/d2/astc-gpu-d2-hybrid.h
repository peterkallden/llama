#pragma once

// D2 hybrid offline bridge: GPU proposal -> CPU legal ASTC finish -> exact
// decoded activation ranking. The physical proposer remains representation
// agnostic; this frontend owns the point where paired-row semantics return.
//
// This is intentionally an offline helper. It neither emits a runtime cache
// artifact nor decides a model-level deployment policy.

#include "astc-gpu-d2-neural-rank.h"
#include "astc-gpu-encoder-finisher.h"
#include "astc-gpu-selector-adapter.h"

#include <cstdint>
#include <string>
#include <vector>

struct astc_gpu_d2_hybrid_options {
    // Bound CPU astcenc work after the physical proposer. Candidate zero (the
    // direct-neutral source) remains mandatory for every physical block.
    uint32_t max_finish_candidates_per_logical = 2;
    // Bound the exact decoded activation bank passed to a later global D2
    // selector. It may be smaller than the finisher budget, never larger.
    uint32_t max_ranked_candidates_per_logical = 2;
    float astcenc_quality = ASTCENC_PRE_FAST;
};

struct astc_gpu_d2_hybrid_result {
    std::vector<astc_gpu_encoder_proposal> retained_proposals;
    std::vector<astc_gpu_encoder_finished_block> finished_blocks;
    std::vector<astc_gpu_encoder_finished_block> ranked_blocks;
    std::vector<astc_gpu_d2_finished_candidate_score> activation_scores;
};

using astc_gpu_d2_selector_delta_request = astc_gpu_selector_delta_request;

// Runs the representation-specific CPU half of the hybrid path over proposals
// from either the Vulkan proposer or the deterministic CPU proposer. The same
// function is deliberately used for both sources, making proposal backend a
// performance choice rather than a semantic difference.
bool astc_gpu_d2_finish_and_rank(
    const astc_gpu_d2_candidate_bank & bank,
    const std::vector<astc_gpu_encoder_proposal> & proposals,
    const astc_gpu_d2_activation_rank_request & rank_request,
    const astc_gpu_d2_hybrid_options & options,
    astc_gpu_d2_hybrid_result & result,
    std::string & error);

// Converts exact decoded candidates into the existing global selector's
// activation-space delta contract. Deltas are relative to the direct-neutral
// decoded block, so candidate zero is always an all-zero fallback. This is a
// pure adapter: it performs no ASTC encoding and no selection itself.
bool astc_gpu_d2_make_selector_candidates(
    const astc_gpu_d2_candidate_bank & bank,
    const std::vector<astc_gpu_encoder_finished_block> & finished,
    const astc_gpu_d2_selector_delta_request & request,
    std::vector<std::vector<astc_vulkan_paired_candidate_delta>> & candidates,
    std::string & error);
