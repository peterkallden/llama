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
    assert(atlas.records[1].layout == astc_vulkan_paired_layout::r_gb);
    assert(atlas.records[5].layout == astc_vulkan_paired_layout::r_gb);

    // D2_6x5 and D2_10x5 have the same two-logical-rows semantic contract as D2_8x5,
    // but a distinct standard ASTC image format and atlas geometry.
    assert(astc_vulkan_build_gpu_ranking_atlas(astc_vulkan_footprint::k6x5, 2, pools, atlas));
    assert(atlas.width == 12 && atlas.height == 15);
    assert(atlas.records.size() == 6 && atlas.records[1].layout == astc_vulkan_paired_layout::r_gb);
    assert(astc_vulkan_build_gpu_ranking_atlas(astc_vulkan_footprint::k10x5, 2, pools, atlas));
    assert(atlas.width == 20 && atlas.height == 15);
    assert(atlas.records.size() == 6 && atlas.records[1].layout == astc_vulkan_paired_layout::r_gb);
    assert(!astc_vulkan_build_gpu_ranking_atlas(astc_vulkan_footprint::k8x8, 4, pools, atlas));
    assert(!astc_vulkan_build_gpu_ranking_atlas(astc_vulkan_footprint::k8x5, 0, pools, atlas));

    std::vector<astc_vulkan_gpu_ranking_source_batch> batches;
    assert(astc_vulkan_plan_gpu_ranking_batches(pools, 3, batches));
    assert(batches.size() == 2);
    assert(batches[0].first_source_block == 0 && batches[0].source_block_count == 2 &&
           batches[0].candidate_count == 3);
    assert(batches[1].first_source_block == 2 && batches[1].source_block_count == 1 &&
           batches[1].candidate_count == 3);
    assert(astc_vulkan_build_gpu_ranking_atlas_range(astc_vulkan_footprint::k8x5, 4, pools,
        batches[1].first_source_block, batches[1].source_block_count, atlas));
    assert(atlas.records.size() == 3 && atlas.records[0].source_block == 2 &&
           atlas.records[0].baseline_record == 0 && atlas.records[2].candidate_index == 2);
    assert(!astc_vulkan_plan_gpu_ranking_batches(pools, 2, batches));
    std::puts("ASTC Vulkan GPU ranking atlas contract passed");
    return 0;
}
