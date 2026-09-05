#include "astc-vulkan-gpu-ranking.h"

#include <algorithm>
#include <limits>

namespace {

bool is_paired_d2_footprint(astc_vulkan_footprint footprint) {
    return footprint == astc_vulkan_footprint::k6x5 ||
           footprint == astc_vulkan_footprint::k8x5 ||
           footprint == astc_vulkan_footprint::k10x5;
}

} // namespace

bool astc_vulkan_build_gpu_ranking_atlas(
        astc_vulkan_footprint footprint,
        uint32_t atlas_blocks_x,
        const std::vector<std::vector<astc_vulkan_gpu_ranking_candidate>> & candidates,
        astc_vulkan_gpu_ranking_atlas & result) {
    return astc_vulkan_build_gpu_ranking_atlas_range(footprint, atlas_blocks_x, candidates,
        0, static_cast<uint32_t>(candidates.size()), result);
}

bool astc_vulkan_build_gpu_ranking_atlas_range(
        astc_vulkan_footprint footprint,
        uint32_t atlas_blocks_x,
        const std::vector<std::vector<astc_vulkan_gpu_ranking_candidate>> & candidates,
        uint32_t first_source_block, uint32_t source_block_count,
        astc_vulkan_gpu_ranking_atlas & result) {
    result = {};
    if (!is_paired_d2_footprint(footprint) || atlas_blocks_x == 0 || candidates.empty() ||
        first_source_block >= candidates.size() || source_block_count == 0 ||
        source_block_count > candidates.size() - first_source_block) return false;

    uint64_t total_candidates = 0;
    for (uint32_t source = first_source_block;
         source < first_source_block + source_block_count; ++source) {
        const auto & pool = candidates[source];
        if (pool.empty()) return false;
        total_candidates += pool.size();
    }
    if (total_candidates > std::numeric_limits<uint32_t>::max()) return false;
    const uint64_t atlas_blocks_y = (total_candidates + atlas_blocks_x - 1) / atlas_blocks_x;
    const astc_vulkan_format_info format = astc_vulkan_format(footprint);
    if (atlas_blocks_y > std::numeric_limits<uint32_t>::max() ||
        atlas_blocks_x > std::numeric_limits<uint32_t>::max() / format.block_width ||
        atlas_blocks_y > std::numeric_limits<uint32_t>::max() / format.block_height) return false;

    result.footprint = footprint;
    result.atlas_blocks_x = atlas_blocks_x;
    result.atlas_blocks_y = static_cast<uint32_t>(atlas_blocks_y);
    result.width = atlas_blocks_x * format.block_width;
    result.height = result.atlas_blocks_y * format.block_height;
    result.payload.assign(static_cast<size_t>(total_candidates) * format.block_bytes, 0);
    result.records.reserve(static_cast<size_t>(total_candidates));

    uint32_t record_index = 0;
    for (uint32_t source_block = first_source_block;
         source_block < first_source_block + source_block_count; ++source_block) {
        const uint32_t baseline_record = record_index;
        const auto & pool = candidates[source_block];
        for (uint32_t candidate_index = 0; candidate_index < pool.size(); ++candidate_index, ++record_index) {
            const uint32_t atlas_x = record_index % atlas_blocks_x;
            const uint32_t atlas_y = record_index / atlas_blocks_x;
            const auto & candidate = pool[candidate_index];
            std::copy(candidate.payload.begin(), candidate.payload.end(),
                      result.payload.begin() + static_cast<size_t>(record_index) * format.block_bytes);
            result.records.push_back({atlas_x, atlas_y, source_block, baseline_record,
                                      candidate_index, candidate.layout});
        }
    }
    return true;
}

bool astc_vulkan_plan_gpu_ranking_batches(
        const std::vector<std::vector<astc_vulkan_gpu_ranking_candidate>> & candidates,
        uint32_t max_candidate_count,
        std::vector<astc_vulkan_gpu_ranking_source_batch> & batches) {
    batches.clear();
    if (candidates.empty() || max_candidate_count == 0) return false;

    uint32_t first = 0;
    uint32_t count = 0;
    for (uint32_t source = 0; source < candidates.size(); ++source) {
        const uint64_t pool_count = candidates[source].size();
        if (pool_count == 0 || pool_count > max_candidate_count) {
            batches.clear();
            return false;
        }
        if (count != 0 && static_cast<uint64_t>(count) + pool_count > max_candidate_count) {
            batches.push_back({first, source - first, count});
            first = source;
            count = 0;
        }
        count += static_cast<uint32_t>(pool_count);
    }
    if (count != 0) batches.push_back({first,
        static_cast<uint32_t>(candidates.size()) - first, count});
    return !batches.empty();
}
