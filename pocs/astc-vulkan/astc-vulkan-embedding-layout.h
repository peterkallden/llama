#pragma once

#include "astc-vulkan-format.h"

#include <cstdint>
#include <string>
#include <vector>

// Token-local ASTC layout primitives.
//
// Matrix D1/D2 layouts optimize a dense matmul: one ASTC texel represents one
// (D1) or two (D2) matrix weights at the same input/reduction location. Token
// embeddings are gathered a row at a time, so their artifact layout instead
// keeps every ASTC microtile within a single logical embedding vector.
//
// E1 stores one logical embedding dimension per physical texel. E2 stores two
// dimensions per texel using standard ASTC luminance + alpha semantics. This
// header owns only logical packing, affine normalization, and metadata
// validation. It deliberately does not encode ASTC payloads or add a runtime
// get_rows provider; those require separate exact-codec and Vulkan gates.

enum class astc_vulkan_embedding_representation : uint32_t {
    kNative = 0,
    kE1Local = 1,
    kE2LocalLA = 2,
};

struct astc_vulkan_embedding_profile {
    astc_vulkan_embedding_representation representation =
        astc_vulkan_embedding_representation::kNative;
    astc_vulkan_footprint footprint = astc_vulkan_footprint::k6x6;

    // E1 has one logical value per texel. E2-LA has luminance and alpha, so
    // both decoded channels are semantic embedding values.
    uint32_t lanes_per_texel = 0;
};

// Per-token affine normalization:
//   normalized[d] = (embedding[d] - bias) / scale
//   embedding[d]  = scale * decoded[d] + bias
// Scale is always finite and positive. The scalar bias is broadcast over an
// embedding vector and is not an ASTC semantic channel.
struct astc_vulkan_embedding_affine {
    float bias = 0.0f;
    float scale = 1.0f;
};

struct astc_vulkan_embedding_dimension_location {
    uint32_t tile_index = 0;
    uint32_t texel_index = 0;
    uint32_t lane = 0;
};

// A global logical-to-physical permutation. Identity is the initial profile;
// later discovery may install a global dimension pairing/permutation when it
// improves exact ASTC replay. Per-token permutations are intentionally outside
// this v1 contract because they require a gather scatter map per lookup.
struct astc_vulkan_embedding_layout {
    astc_vulkan_embedding_profile profile;
    uint32_t dimensions = 0;
    std::vector<uint32_t> logical_to_physical;
};

// One token is described by its storage class and row index. The eventual
// runtime provider can use this descriptor to route protected tokens to native
// storage and all other tokens to an E1/E2 ASTC resource.
struct astc_vulkan_embedding_token_descriptor {
    astc_vulkan_embedding_representation representation =
        astc_vulkan_embedding_representation::kNative;
    uint32_t resource_row = 0;
};

bool astc_vulkan_embedding_make_profile(astc_vulkan_embedding_representation representation,
                                        astc_vulkan_footprint footprint,
                                        astc_vulkan_embedding_profile & profile,
                                        std::string & error);
bool astc_vulkan_embedding_validate_layout(const astc_vulkan_embedding_layout & layout,
                                           std::string & error);

uint32_t astc_vulkan_embedding_values_per_tile(const astc_vulkan_embedding_profile & profile);
uint32_t astc_vulkan_embedding_tile_count(const astc_vulkan_embedding_layout & layout);
uint64_t astc_vulkan_embedding_payload_bytes_per_token(const astc_vulkan_embedding_layout & layout);

bool astc_vulkan_embedding_locate_dimension(const astc_vulkan_embedding_layout & layout,
                                            uint32_t logical_dimension,
                                            astc_vulkan_embedding_dimension_location & location,
                                            std::string & error);

astc_vulkan_embedding_affine astc_vulkan_embedding_affine_from_row(
    const std::vector<float> & embedding);
bool astc_vulkan_embedding_normalize(const std::vector<float> & embedding,
                                     const astc_vulkan_embedding_affine & affine,
                                     std::vector<float> & normalized,
                                     std::string & error);
bool astc_vulkan_embedding_restore(const astc_vulkan_embedding_layout & layout,
                                   const std::vector<float> & decoded_physical,
                                   const astc_vulkan_embedding_affine & affine,
                                   std::vector<float> & embedding,
                                   std::string & error);
