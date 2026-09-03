#include "astc-vulkan-gpu-d1-ranking.h"

#include <cassert>

namespace {
astc_vulkan_gpu_d1_ranking_candidate candidate(uint8_t marker) {
    astc_vulkan_gpu_d1_ranking_candidate result;
    result.payload[0] = marker;
    result.payload[15] = static_cast<uint8_t>(marker + 1);
    return result;
}
}

int main() {
    const std::vector<std::vector<astc_vulkan_gpu_d1_ranking_candidate>> pools{
        {candidate(10), candidate(20)}, {candidate(30)}, {candidate(40), candidate(50)}};
    astc_vulkan_gpu_d1_ranking_atlas atlas;
    assert(astc_vulkan_build_gpu_d1_ranking_atlas(
        astc_vulkan_footprint::k4x4, astc_vulkan_d1_semantic_decoder::scalar, 3, pools, atlas));
    assert(atlas.width == 12 && atlas.height == 8 && atlas.records.size() == 5);
    assert(atlas.records[1].baseline_record == 0 && atlas.records[2].baseline_record == 2);
    assert(atlas.records[4].source_block == 2 && atlas.records[4].baseline_record == 3);
    assert(astc_vulkan_build_gpu_d1_ranking_atlas(
        astc_vulkan_footprint::k10x8, astc_vulkan_d1_semantic_decoder::gauge_la, 2, pools, atlas));
    assert(atlas.width == 20 && atlas.height == 24 &&
           atlas.decoder == astc_vulkan_d1_semantic_decoder::gauge_la);
    std::vector<astc_vulkan_gpu_d1_ranking_source_batch> batches;
    assert(astc_vulkan_plan_gpu_d1_ranking_batches(pools, 3, batches));
    assert(batches.size() == 2 && batches[0].candidate_count == 3 && batches[1].candidate_count == 2);
    assert(astc_vulkan_build_gpu_d1_ranking_atlas_range(
        astc_vulkan_footprint::k6x6, astc_vulkan_d1_semantic_decoder::gauge_la, 2,
        pools, batches[1].first_source_block, batches[1].source_block_count, atlas));
    assert(atlas.records.size() == 2 && atlas.records[0].source_block == 2);
    return 0;
}
