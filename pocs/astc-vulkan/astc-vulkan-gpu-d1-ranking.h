#pragma once

// Offline D1 GPU-ranking transport contract.
//
// D1 stores one logical weight per ASTC texel.  The CPU always creates legal
// ASTC payloads with astcenc; this type only batches those payloads into an
// ordinary sampled ASTC atlas for a future Vulkan score pass.  It deliberately
// contains no D2 layout metadata, so the same contract applies to scalar and
// scalar-anchored L+A gauge candidates from 4x4 through 10x8.

#include "astc-vulkan-format.h"

#include <array>
#include <cstdint>
#include <vector>

enum class astc_vulkan_d1_semantic_decoder : uint8_t {
    // One decoded channel reconstructs the logical weight.
    scalar = 0,
    // The runtime reconstruction uses the L+A gauge contract.  The exact
    // affine constants live in the artifact, not in this transport.
    gauge_la = 1,
};

struct astc_vulkan_gpu_d1_ranking_candidate {
    std::array<uint8_t, 16> payload{};
};

struct astc_vulkan_gpu_d1_ranking_record {
    uint32_t atlas_block_x = 0;
    uint32_t atlas_block_y = 0;
    uint32_t source_block = 0;
    // Candidate zero is the mandatory scalar/gauge-neutral baseline of its
    // source block.  The value is local to the packed atlas batch.
    uint32_t baseline_record = 0;
    uint32_t candidate_index = 0;
};

struct astc_vulkan_gpu_d1_ranking_atlas {
    astc_vulkan_footprint footprint = astc_vulkan_footprint::k4x4;
    astc_vulkan_d1_semantic_decoder decoder = astc_vulkan_d1_semantic_decoder::scalar;
    uint32_t atlas_blocks_x = 0;
    uint32_t atlas_blocks_y = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<uint8_t> payload;
    std::vector<astc_vulkan_gpu_d1_ranking_record> records;
};

struct astc_vulkan_gpu_d1_ranking_source_batch {
    uint32_t first_source_block = 0;
    uint32_t source_block_count = 0;
    uint32_t candidate_count = 0;
};

// All D1 footprints known to the artifact format are legal here.  Individual
// Vulkan devices still capability-check the actual sampled ASTC format.
bool astc_vulkan_build_gpu_d1_ranking_atlas(
    astc_vulkan_footprint footprint,
    astc_vulkan_d1_semantic_decoder decoder,
    uint32_t atlas_blocks_x,
    const std::vector<std::vector<astc_vulkan_gpu_d1_ranking_candidate>> & candidates,
    astc_vulkan_gpu_d1_ranking_atlas & result);

bool astc_vulkan_plan_gpu_d1_ranking_batches(
    const std::vector<std::vector<astc_vulkan_gpu_d1_ranking_candidate>> & candidates,
    uint32_t max_candidate_count,
    std::vector<astc_vulkan_gpu_d1_ranking_source_batch> & batches);

bool astc_vulkan_build_gpu_d1_ranking_atlas_range(
    astc_vulkan_footprint footprint,
    astc_vulkan_d1_semantic_decoder decoder,
    uint32_t atlas_blocks_x,
    const std::vector<std::vector<astc_vulkan_gpu_d1_ranking_candidate>> & candidates,
    uint32_t first_source_block,
    uint32_t source_block_count,
    astc_vulkan_gpu_d1_ranking_atlas & result);
