#pragma once

#include "astc-vulkan-format.h"
#include "astc-vulkan-paired.h"

#include <array>
#include <cstdint>
#include <vector>

// Offline GPU candidate-ranking transport contract.
//
// CPU astcenc remains responsible for producing every legal 16-byte ASTC
// payload. This module only packs those payloads into one ordinary sampled
// ASTC image atlas so a Vulkan compute shader can use fixed-function decode
// while scoring D2 candidate deltas. It owns no Vulkan objects and is safe to
// test without a device.
//
// Suitable for: experimental paired-D2 offline ranking. It is not a runtime
// inference texture layout and it does not alter the artifact format.

struct astc_vulkan_gpu_ranking_candidate {
    std::array<uint8_t, 16> payload{};
    astc_vulkan_paired_layout layout = astc_vulkan_paired_layout::rg_b;
};

struct astc_vulkan_gpu_ranking_record {
    // Location of this candidate's single physical ASTC block in the atlas.
    uint32_t atlas_block_x = 0;
    uint32_t atlas_block_y = 0;

    // Original physical D2 block and its neutral candidate record. Candidate
    // zero is mandatory and is the delta baseline for its source block.
    uint32_t source_block = 0;
    uint32_t baseline_record = 0;
    uint32_t candidate_index = 0;
    astc_vulkan_paired_layout layout = astc_vulkan_paired_layout::rg_b;
};

struct astc_vulkan_gpu_ranking_atlas {
    astc_vulkan_footprint footprint = astc_vulkan_footprint::k8x5;
    uint32_t atlas_blocks_x = 0;
    uint32_t atlas_blocks_y = 0;
    uint32_t width = 0;
    uint32_t height = 0;

    // ASTC blocks in normal raster order, exactly 16 bytes per atlas block.
    std::vector<uint8_t> payload;
    std::vector<astc_vulkan_gpu_ranking_record> records;
};

struct astc_vulkan_gpu_ranking_source_batch {
    uint32_t first_source_block = 0;
    uint32_t source_block_count = 0;
    uint32_t candidate_count = 0;
};

// Packs a per-source-block candidate pool into a rectangular sampled ASTC
// atlas. Every inner vector must be non-empty; element zero is its mandatory
// neutral baseline. `atlas_blocks_x` is the upload-batch shaping choice, not
// a neural or ASTC encoding parameter.
bool astc_vulkan_build_gpu_ranking_atlas(
    astc_vulkan_footprint footprint,
    uint32_t atlas_blocks_x,
    const std::vector<std::vector<astc_vulkan_gpu_ranking_candidate>> & candidates,
    astc_vulkan_gpu_ranking_atlas & result);

// Partitions source-block candidate pools without splitting a pool. This is
// the worker unit for streamed CPU-producer/GPU-consumer execution: every
// source block keeps its neutral baseline and its candidates in one batch.
// Returns false if one source pool alone exceeds max_candidate_count.
bool astc_vulkan_plan_gpu_ranking_batches(
    const std::vector<std::vector<astc_vulkan_gpu_ranking_candidate>> & candidates,
    uint32_t max_candidate_count,
    std::vector<astc_vulkan_gpu_ranking_source_batch> & batches);

// Builds one atlas for a planned source-block range. `source_block` values in
// records remain global, while baseline_record is local to this atlas.
bool astc_vulkan_build_gpu_ranking_atlas_range(
    astc_vulkan_footprint footprint,
    uint32_t atlas_blocks_x,
    const std::vector<std::vector<astc_vulkan_gpu_ranking_candidate>> & candidates,
    uint32_t first_source_block,
    uint32_t source_block_count,
    astc_vulkan_gpu_ranking_atlas & result);
