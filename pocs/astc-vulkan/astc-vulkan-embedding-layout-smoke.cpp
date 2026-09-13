#include "astc-vulkan-embedding-layout.h"

#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

namespace {

bool print_profile(astc_vulkan_embedding_representation representation,
                   astc_vulkan_footprint footprint,
                   uint32_t dimensions) {
    astc_vulkan_embedding_profile profile;
    std::string error;
    if (!astc_vulkan_embedding_make_profile(representation, footprint, profile, error)) {
        std::fprintf(stderr, "profile error: %s\n", error.c_str());
        return false;
    }
    astc_vulkan_embedding_layout layout;
    layout.profile = profile;
    layout.dimensions = dimensions;
    layout.logical_to_physical.resize(dimensions);
    std::iota(layout.logical_to_physical.begin(), layout.logical_to_physical.end(), 0);
    if (!astc_vulkan_embedding_validate_layout(layout, error)) {
        std::fprintf(stderr, "layout error: %s\n", error.c_str());
        return false;
    }
    std::printf("embedding-smoke rep=%s footprint=%ux%u dimensions=%u values/tile=%u tiles/token=%u bytes/token=%llu\n",
                representation == astc_vulkan_embedding_representation::kE1Local ? "e1-local" : "e2-local-la",
                astc_vulkan_format(footprint).block_width, astc_vulkan_format(footprint).block_height,
                dimensions, astc_vulkan_embedding_values_per_tile(profile),
                astc_vulkan_embedding_tile_count(layout),
                static_cast<unsigned long long>(astc_vulkan_embedding_payload_bytes_per_token(layout)));
    return true;
}

} // namespace

int main() {
    constexpr uint32_t qwen_embedding_dimensions = 1536;
    return print_profile(astc_vulkan_embedding_representation::kE1Local,
                         astc_vulkan_footprint::k10x5, qwen_embedding_dimensions) &&
           print_profile(astc_vulkan_embedding_representation::kE2LocalLA,
                         astc_vulkan_footprint::k8x5, qwen_embedding_dimensions) &&
           print_profile(astc_vulkan_embedding_representation::kE2LocalLA,
                         astc_vulkan_footprint::k10x5, qwen_embedding_dimensions) ? 0 : 1;
}
