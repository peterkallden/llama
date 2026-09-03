#include "astc-vulkan-gpu-ranking.h"

#include <cassert>
#include <cstdio>

namespace {

astc_vulkan_gpu_ranking_candidate candidate(uint8_t first_byte,
                                             astc_vulkan_paired_layout layout) {
    astc_vulkan_gpu_ranking_candidate result;
    result.payload[0] = first_byte;
    result.payload[15] = static_cast<uint8_t>(first_byte + 1);
    result.layout = layout;
    return result;
}

} // namespace

int main() {
    const std::vector<std::vector<astc_vulkan_gpu_ranking_candidate>> pools{
        {candidate(10, astc_vulkan_paired_layout::rg_b),
         candidate(20, astc_vulkan_paired_layout::r_gb)},
        {candidate(30, astc_vulkan_paired_layout::rg_b)},
        {candidate(40, astc_vulkan_paired_layout::r_gb),
         candidate(50, astc_vulkan_paired_layout::rg_b),
         candidate(60, astc_vulkan_paired_layout::r_gb)},
    };
    astc_vulkan_gpu_ranking_atlas atlas;
    assert(astc_vulkan_build_gpu_ranking_atlas(astc_vulkan_footprint::k8x5, 4, pools, atlas));
    assert(atlas.width == 32 && atlas.height == 10);
    assert(atlas.atlas_blocks_x == 4 && atlas.atlas_blocks_y == 2);
    assert(atlas.payload.size() == 6 * 16 && atlas.records.size() == 6);
    assert(atlas.payload[0] == 10 && atlas.payload[16] == 20 && atlas.payload[32] == 30);
    assert(atlas.records[0].source_block == 0 && atlas.records[0].baseline_record == 0);
    assert(atlas.records[1].source_block == 0 && atlas.records[1].baseline_record == 0);
    assert(atlas.records[2].source_block == 1 && atlas.records[2].baseline_record == 2);
    assert(atlas.records[5].source_block == 2 && atlas.records[5].baseline_record == 3);
    assert(atlas.records[5].atlas_block_x == 1 && atlas.records[5].atlas_block_y == 1);
    assert(!astc_vulkan_build_gpu_ranking_atlas(astc_vulkan_footprint::k8x8, 4, pools, atlas));
    assert(!astc_vulkan_build_gpu_ranking_atlas(astc_vulkan_footprint::k8x5, 0, pools, atlas));
    std::puts("ASTC Vulkan GPU ranking atlas contract passed");
    return 0;
}
