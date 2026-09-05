#pragma once

// D1 candidate-bank contract for the hybrid GPU-proposer / CPU-finisher path.
//
// Each logical physical position owns an obligatory scalar source candidate and
// optional alternative source families (currently gauge-L+A). Candidate source
// IDs are unique globally so the generic GPU proposer and CPU finisher need no
// D1-specific side state. This file preserves the mapping back to the original
// logical ASTC block for later neural selection.

#include "astc-gpu-encoder.h"

#include <cstdint>
#include <vector>

enum class astc_gpu_d1_candidate_family : uint8_t {
    scalar = 0,
    gauge_la = 1,
};

struct astc_gpu_d1_candidate_family_sources {
    astc_gpu_d1_candidate_family family = astc_gpu_d1_candidate_family::gauge_la;
    // Exactly one source block for each scalar physical block, in matching
    // raster order. The builder assigns unique candidate source IDs itself.
    std::vector<astc_gpu_encoder_source_block> blocks;
};

struct astc_gpu_d1_candidate_record {
    uint32_t candidate_source_block_id = 0;
    uint32_t logical_source_block_id = 0;
    uint32_t candidate_index = 0;
    astc_gpu_d1_candidate_family family = astc_gpu_d1_candidate_family::scalar;
};

struct astc_gpu_d1_candidate_bank {
    astc_vulkan_footprint footprint = astc_vulkan_footprint::k4x4;
    std::vector<astc_gpu_encoder_source_block> candidate_blocks;
    std::vector<astc_gpu_d1_candidate_record> records;
};

// Scalar is always candidate zero per logical block. Each additional family
// adds one candidate per logical block. The function does no quality decision.
bool astc_gpu_d1_build_candidate_bank(
    astc_vulkan_footprint footprint,
    const std::vector<astc_gpu_encoder_source_block> & scalar_blocks,
    const std::vector<astc_gpu_d1_candidate_family_sources> & alternatives,
    astc_gpu_d1_candidate_bank & bank);

// Creates one generic physical proposal request. The caller may split this
// request with astc_gpu_encoder_plan_batches for a bounded persistent session.
bool astc_gpu_d1_candidate_bank_request(
    const astc_gpu_d1_candidate_bank & bank, uint32_t max_blocks_per_batch,
    astc_gpu_encoder_request & request);

// First inexpensive selection policy: scalar is mandatory, then retain the
// lowest physical proposal errors until max_candidates_per_logical is reached.
// It is only a bounded CPU-finisher budget; neural activation scoring replaces
// this ranking in the later exact-selection stage.
bool astc_gpu_d1_select_candidate_bank_proposals(
    const astc_gpu_d1_candidate_bank & bank,
    const std::vector<astc_gpu_encoder_proposal> & proposals,
    uint32_t max_candidates_per_logical,
    std::vector<astc_gpu_encoder_proposal> & selected);
