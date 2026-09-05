#pragma once

// Bounded D2 candidate-bank contract for GPU proposal -> CPU astcenc finish.
//
// Scalar-like source IDs remain globally unique, while this D2-owned record
// retains paired layout and semantic interpretation for later exact scoring.
// The shared physical proposer and finisher deliberately know none of it.

#include "astc-gpu-encoder.h"
#include "astc-vulkan-paired.h"

#include <cstdint>
#include <vector>

enum class astc_gpu_d2_candidate_family : uint8_t {
    direct_neutral = 0,
    direct_steered = 1,
    luminance_alpha = 2,
};

struct astc_gpu_d2_candidate_family_sources {
    astc_gpu_d2_candidate_family family = astc_gpu_d2_candidate_family::direct_steered;
    astc_vulkan_paired_semantic semantic = astc_vulkan_paired_semantic::direct_rgb;
    std::vector<astc_gpu_encoder_source_block> blocks;
    std::vector<astc_vulkan_paired_layout> layouts;
};

struct astc_gpu_d2_candidate_record {
    uint32_t candidate_source_block_id = 0;
    uint32_t logical_source_block_id = 0;
    uint32_t candidate_index = 0;
    astc_gpu_d2_candidate_family family = astc_gpu_d2_candidate_family::direct_neutral;
    astc_vulkan_paired_layout layout = astc_vulkan_paired_layout::rg_b;
    astc_vulkan_paired_semantic semantic = astc_vulkan_paired_semantic::direct_rgb;
};

struct astc_gpu_d2_candidate_bank {
    astc_vulkan_footprint footprint = astc_vulkan_footprint::k8x5;
    std::vector<astc_gpu_encoder_source_block> candidate_blocks;
    std::vector<astc_gpu_d2_candidate_record> records;
};

// Direct-neutral is mandatory candidate zero per physical block. Alternatives
// may use direct Alpha steering or L+A semantics, with their own layout map.
bool astc_gpu_d2_build_candidate_bank(
    astc_vulkan_footprint footprint,
    const std::vector<astc_gpu_encoder_source_block> & direct_neutral_blocks,
    const std::vector<astc_vulkan_paired_layout> & direct_neutral_layouts,
    const std::vector<astc_gpu_d2_candidate_family_sources> & alternatives,
    astc_gpu_d2_candidate_bank & bank);

bool astc_gpu_d2_candidate_bank_request(
    const astc_gpu_d2_candidate_bank & bank, uint32_t max_blocks_per_batch,
    astc_gpu_encoder_request & request);

// Temporary CPU-finisher budget: retain mandatory direct-neutral then minimum
// physical proposal-error alternatives. Exact D2 scoring is a later frontend
// stage because it must reconstruct both output rows with record semantics.
bool astc_gpu_d2_select_candidate_bank_proposals(
    const astc_gpu_d2_candidate_bank & bank,
    const std::vector<astc_gpu_encoder_proposal> & proposals,
    uint32_t max_candidates_per_logical,
    std::vector<astc_gpu_encoder_proposal> & selected);
