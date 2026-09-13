#include "astc-vulkan-embedding-layout.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#define CHECK(expression) do { \
    if (!(expression)) { \
        std::fprintf(stderr, "check failed: %s (%s:%d)\\n", #expression, __FILE__, __LINE__); \
        return 1; \
    } \
} while (false)

int main() {
    std::string error;
    astc_vulkan_embedding_profile e1;
    CHECK(astc_vulkan_embedding_make_profile(
        astc_vulkan_embedding_representation::kE1Local,
        astc_vulkan_footprint::k10x5, e1, error));
    CHECK(e1.lanes_per_texel == 1);
    CHECK(astc_vulkan_embedding_values_per_tile(e1) == 50);

    astc_vulkan_embedding_profile e2;
    CHECK(astc_vulkan_embedding_make_profile(
        astc_vulkan_embedding_representation::kE2LocalLA,
        astc_vulkan_footprint::k8x5, e2, error));
    CHECK(e2.lanes_per_texel == 2);
    CHECK(astc_vulkan_embedding_values_per_tile(e2) == 80);

    astc_vulkan_embedding_layout layout;
    layout.profile = e2;
    layout.dimensions = 1536;
    layout.logical_to_physical.resize(layout.dimensions);
    for (uint32_t logical = 0; logical < layout.dimensions; ++logical) {
        // Deterministic non-identity pair-oriented permutation. The smoke
        // exercises the same scatter contract a future pair-map uses.
        layout.logical_to_physical[logical] = (logical % 2 == 0) ?
            logical + 1 : logical - 1;
    }
    CHECK(astc_vulkan_embedding_validate_layout(layout, error));
    CHECK(astc_vulkan_embedding_tile_count(layout) == 20);
    CHECK(astc_vulkan_embedding_payload_bytes_per_token(layout) == 320);

    astc_vulkan_embedding_dimension_location location;
    CHECK(astc_vulkan_embedding_locate_dimension(layout, 0, location, error));
    CHECK(location.tile_index == 0 && location.texel_index == 0 && location.lane == 1);
    CHECK(astc_vulkan_embedding_locate_dimension(layout, 79, location, error));
    CHECK(location.tile_index == 0 && location.texel_index == 39 && location.lane == 0);
    CHECK(astc_vulkan_embedding_locate_dimension(layout, 80, location, error));
    CHECK(location.tile_index == 1 && location.texel_index == 0 && location.lane == 1);

    std::vector<float> source(layout.dimensions);
    for (uint32_t i = 0; i < layout.dimensions; ++i) {
        source[i] = std::sin(static_cast<float>(i) * 0.013f) + 0.25f;
    }
    const auto affine = astc_vulkan_embedding_affine_from_row(source);
    std::vector<float> normalized;
    CHECK(astc_vulkan_embedding_normalize(source, affine, normalized, error));

    // Simulate ASTC decode in physical ordering. Exact identity here isolates
    // layout/affine correctness; ASTC encode/decode error belongs in the later
    // E1/E2 codec discovery smoke.
    std::vector<float> decoded_physical(layout.dimensions);
    for (uint32_t logical = 0; logical < layout.dimensions; ++logical) {
        decoded_physical[layout.logical_to_physical[logical]] = normalized[logical];
    }
    std::vector<float> restored;
    CHECK(astc_vulkan_embedding_restore(layout, decoded_physical, affine, restored, error));
    for (uint32_t i = 0; i < layout.dimensions; ++i) {
        CHECK(std::abs(restored[i] - source[i]) < 1e-6f);
    }

    astc_vulkan_embedding_token_descriptor protected_token;
    protected_token.representation = astc_vulkan_embedding_representation::kNative;
    protected_token.resource_row = 17;
    astc_vulkan_embedding_token_descriptor compressed_token;
    compressed_token.representation = astc_vulkan_embedding_representation::kE2LocalLA;
    compressed_token.resource_row = 42;
    CHECK(protected_token.representation != compressed_token.representation);

    layout.logical_to_physical.back() = 0;
    CHECK(!astc_vulkan_embedding_validate_layout(layout, error));
    std::puts("ASTC Vulkan token-local E1/E2 embedding layout contract passed");
    return 0;
}

#undef CHECK
