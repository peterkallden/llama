#include "astc-vulkan-embedding-layout.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace {

uint32_t footprint_width(astc_vulkan_footprint footprint) {
    return astc_vulkan_footprint_is_valid(footprint) ?
        astc_vulkan_format(footprint).block_width : 0;
}

uint32_t footprint_height(astc_vulkan_footprint footprint) {
    return astc_vulkan_footprint_is_valid(footprint) ?
        astc_vulkan_format(footprint).block_height : 0;
}

} // namespace

bool astc_vulkan_embedding_make_profile(astc_vulkan_embedding_representation representation,
                                        astc_vulkan_footprint footprint,
                                        astc_vulkan_embedding_profile & profile,
                                        std::string & error) {
    if (footprint_width(footprint) == 0 || footprint_height(footprint) == 0) {
        error = "embedding profile requires a valid ASTC footprint";
        return false;
    }
    switch (representation) {
        case astc_vulkan_embedding_representation::kE1Local:
            profile = { representation, footprint, 1 };
            return true;
        case astc_vulkan_embedding_representation::kE2LocalLA:
            profile = { representation, footprint, 2 };
            return true;
        case astc_vulkan_embedding_representation::kNative:
            error = "native embeddings have no ASTC tile profile";
            return false;
    }
    error = "unknown embedding representation";
    return false;
}

uint32_t astc_vulkan_embedding_values_per_tile(const astc_vulkan_embedding_profile & profile) {
    const uint32_t width = footprint_width(profile.footprint);
    const uint32_t height = footprint_height(profile.footprint);
    if (profile.lanes_per_texel == 0 || width == 0 || height == 0 ||
        width > std::numeric_limits<uint32_t>::max() / height ||
        width * height > std::numeric_limits<uint32_t>::max() / profile.lanes_per_texel) {
        return 0;
    }
    return width * height * profile.lanes_per_texel;
}

bool astc_vulkan_embedding_validate_layout(const astc_vulkan_embedding_layout & layout,
                                           std::string & error) {
    if (layout.dimensions == 0) {
        error = "embedding layout requires at least one dimension";
        return false;
    }
    if (layout.profile.representation == astc_vulkan_embedding_representation::kNative ||
        astc_vulkan_embedding_values_per_tile(layout.profile) == 0) {
        error = "embedding layout requires an E1 or E2 ASTC profile";
        return false;
    }
    if (layout.logical_to_physical.size() != layout.dimensions) {
        error = "embedding dimension permutation size does not match dimensions";
        return false;
    }
    std::vector<bool> seen(layout.dimensions, false);
    for (uint32_t physical : layout.logical_to_physical) {
        if (physical >= layout.dimensions || seen[physical]) {
            error = "embedding dimension permutation is not bijective";
            return false;
        }
        seen[physical] = true;
    }
    return true;
}

uint32_t astc_vulkan_embedding_tile_count(const astc_vulkan_embedding_layout & layout) {
    const uint32_t values = astc_vulkan_embedding_values_per_tile(layout.profile);
    return values == 0 || layout.dimensions == 0 ? 0 :
        (layout.dimensions + values - 1) / values;
}

uint64_t astc_vulkan_embedding_payload_bytes_per_token(const astc_vulkan_embedding_layout & layout) {
    return static_cast<uint64_t>(astc_vulkan_embedding_tile_count(layout)) * 16;
}

bool astc_vulkan_embedding_locate_dimension(const astc_vulkan_embedding_layout & layout,
                                            uint32_t logical_dimension,
                                            astc_vulkan_embedding_dimension_location & location,
                                            std::string & error) {
    if (!astc_vulkan_embedding_validate_layout(layout, error)) return false;
    if (logical_dimension >= layout.dimensions) {
        error = "embedding logical dimension is out of range";
        return false;
    }
    const uint32_t values = astc_vulkan_embedding_values_per_tile(layout.profile);
    const uint32_t physical = layout.logical_to_physical[logical_dimension];
    const uint32_t in_tile = physical % values;
    location.tile_index = physical / values;
    location.texel_index = in_tile / layout.profile.lanes_per_texel;
    location.lane = in_tile % layout.profile.lanes_per_texel;
    return true;
}

astc_vulkan_embedding_affine astc_vulkan_embedding_affine_from_row(
    const std::vector<float> & embedding) {
    if (embedding.empty()) return {};
    double sum = 0.0;
    for (float value : embedding) sum += value;
    const float bias = static_cast<float>(sum / embedding.size());
    float scale = 0.0f;
    for (float value : embedding) scale = std::max(scale, std::abs(value - bias));
    if (!std::isfinite(scale) || scale <= std::numeric_limits<float>::min()) scale = 1.0f;
    return { bias, scale };
}

bool astc_vulkan_embedding_normalize(const std::vector<float> & embedding,
                                     const astc_vulkan_embedding_affine & affine,
                                     std::vector<float> & normalized,
                                     std::string & error) {
    if (!std::isfinite(affine.bias) || !std::isfinite(affine.scale) || affine.scale <= 0.0f) {
        error = "embedding affine parameters must be finite with positive scale";
        return false;
    }
    normalized.resize(embedding.size());
    for (size_t i = 0; i < embedding.size(); ++i) {
        if (!std::isfinite(embedding[i])) {
            error = "embedding source contains a non-finite value";
            return false;
        }
        normalized[i] = (embedding[i] - affine.bias) / affine.scale;
    }
    return true;
}

bool astc_vulkan_embedding_restore(const astc_vulkan_embedding_layout & layout,
                                   const std::vector<float> & decoded_physical,
                                   const astc_vulkan_embedding_affine & affine,
                                   std::vector<float> & embedding,
                                   std::string & error) {
    if (!astc_vulkan_embedding_validate_layout(layout, error)) return false;
    if (decoded_physical.size() < layout.dimensions) {
        error = "decoded embedding source is shorter than the logical dimension count";
        return false;
    }
    if (!std::isfinite(affine.bias) || !std::isfinite(affine.scale) || affine.scale <= 0.0f) {
        error = "embedding affine parameters must be finite with positive scale";
        return false;
    }
    embedding.resize(layout.dimensions);
    for (uint32_t logical = 0; logical < layout.dimensions; ++logical) {
        const float value = decoded_physical[layout.logical_to_physical[logical]];
        if (!std::isfinite(value)) {
            error = "decoded embedding contains a non-finite value";
            return false;
        }
        embedding[logical] = affine.scale * value + affine.bias;
    }
    return true;
}
