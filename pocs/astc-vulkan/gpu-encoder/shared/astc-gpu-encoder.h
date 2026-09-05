#pragma once

// Offline GPU ASTC encoder: shared physical proposal contract.
//
// This layer is intentionally representation-agnostic.  It owns the physical
// source texels that a GPU backend may inspect, batching, and symbolic
// proposal output.  It does *not* construct an ASTC payload in v1.  A later
// CPU finisher uses the existing astcenc path to turn retained proposals into
// legal 16-byte blocks; astcenc remains the reference and fallback.
//
// D1 and D2 frontends may create different source texels, but no D1 gauge or
// D2 paired-row assumption is allowed in this header.

#include "astc-vulkan-format.h"

#include <array>
#include <cstdint>
#include <vector>

enum class astc_gpu_encode_mode : uint8_t {
    // GPU produces symbolic endpoint/weight/mode hints. CPU performs exact
    // legal ASTC construction and verification.
    propose,
    // Reserved for a later, deliberately small legal ASTC subset with GPU
    // quantization, BISE packing, and 128-bit payload emission.
    exact_subset,
};

struct astc_gpu_encoder_texel {
    std::array<float, 4> rgba{};
};

struct astc_gpu_encoder_source_block {
    astc_vulkan_footprint footprint = astc_vulkan_footprint::k4x4;
    uint32_t source_block_id = 0;
    // Raster order, exactly footprint_width * footprint_height elements.
    std::vector<astc_gpu_encoder_texel> texels;
};

// A compact physical proposal. Its mode fields are hints, not an ASTC
// bitstream contract. The CPU finisher may reject or refine every hint.
struct astc_gpu_encoder_proposal {
    uint32_t source_block_id = 0;
    uint32_t mode_family = 0;
    uint32_t weight_grid_x = 0;
    uint32_t weight_grid_y = 0;
    std::array<float, 4> endpoint_low{};
    std::array<float, 4> endpoint_high{};
    float approximate_error = 0.0f;
};

// Common opaque identity carried from a representation-specific candidate
// bank through the physical GPU proposer. The shared layer never interprets
// family_id; D1/D2 use it only to recover their semantic record offline.
struct astc_gpu_encoder_candidate_identity {
    uint32_t source_block_id = 0;
    uint32_t logical_source_block_id = 0;
    uint32_t candidate_index = 0;
    uint32_t family_id = 0;
};

struct astc_gpu_encoder_request {
    astc_gpu_encode_mode mode = astc_gpu_encode_mode::propose;
    astc_vulkan_footprint footprint = astc_vulkan_footprint::k4x4;
    // Batch size is a host scheduling choice. A backend must process the
    // complete batch in one dispatch/readback cycle, never callback per block.
    uint32_t max_blocks_per_batch = 256;
    std::vector<astc_gpu_encoder_source_block> blocks;
    // Optional, but when present it must be one-to-one and in the same order
    // as blocks. This keeps physical batching independent of D1/D2 semantics.
    std::vector<astc_gpu_encoder_candidate_identity> candidate_metadata;
};

struct astc_gpu_encoder_batch {
    uint32_t first_block = 0;
    uint32_t block_count = 0;
};

// Validates physical source dimensions and plans deterministic, contiguous
// batches. This is shared by Vulkan, CPU-reference, and future backends.
bool astc_gpu_encoder_plan_batches(const astc_gpu_encoder_request & request,
                                   std::vector<astc_gpu_encoder_batch> & batches);

// CPU-only reference for the *proposal* stage. It supplies deterministic
// endpoint bounds and a bounded error statistic, allowing the host contract
// and D1 source construction to be tested before a Vulkan kernel is added.
bool astc_gpu_encoder_propose_cpu_reference(
    const astc_gpu_encoder_request & request,
    std::vector<astc_gpu_encoder_proposal> & proposals);
