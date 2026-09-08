#include "astc-gpu-d1-exact-subset.h"

namespace {

bool make_request(const astc_gpu_d1_candidate_bank & semantic_bank,
                  astc_gpu_d1_candidate_family family,
                  astc_gpu_exact_subset_kind kind,
                  uint32_t max_blocks_per_batch,
                  astc_gpu_encoder_request & request) {
    request = {};
    request.mode = astc_gpu_encode_mode::exact_subset;
    request.footprint = astc_vulkan_footprint::k6x6;
    request.exact_subset = kind;
    request.max_blocks_per_batch = max_blocks_per_batch;
    for (size_t index = 0; index < semantic_bank.records.size(); ++index) {
        const auto & record = semantic_bank.records[index];
        if (record.family != family) continue;
        request.blocks.push_back(semantic_bank.candidate_blocks[index]);
        request.candidate_metadata.push_back({record.candidate_source_block_id,
                                              record.logical_source_block_id,
                                              record.candidate_index,
                                              static_cast<uint32_t>(record.family)});
    }
    return !request.blocks.empty();
}

} // namespace

bool astc_gpu_d1_build_refined_6x6_exact_subset_candidate_bank(
    const std::vector<astc_gpu_encoder_source_block> & scalar_blocks,
    uint32_t max_blocks_per_batch,
    astc_gpu_d1_exact_subset_candidate_bank & bank) {
    bank = {};
    if (scalar_blocks.empty() || max_blocks_per_batch == 0 ||
        !astc_gpu_d1_build_candidate_bank(
            astc_vulkan_footprint::k6x6, scalar_blocks,
            {{astc_gpu_d1_candidate_family::scalar_refined, scalar_blocks},
             {astc_gpu_d1_candidate_family::scalar_mean_refined, scalar_blocks},
             {astc_gpu_d1_candidate_family::scalar_quantile_refined, scalar_blocks}},
            bank.semantic_bank)) return false;
    return make_request(bank.semantic_bank, astc_gpu_d1_candidate_family::scalar,
                        astc_gpu_exact_subset_kind::d1_luminance_binary_6x6,
                        max_blocks_per_batch, bank.binary_request) &&
           make_request(bank.semantic_bank, astc_gpu_d1_candidate_family::scalar_refined,
                        astc_gpu_exact_subset_kind::d1_luminance_binary_refined_6x6,
                        max_blocks_per_batch, bank.refined_request) &&
           make_request(bank.semantic_bank, astc_gpu_d1_candidate_family::scalar_mean_refined,
                        astc_gpu_exact_subset_kind::d1_luminance_binary_mean_refined_6x6,
                        max_blocks_per_batch, bank.mean_refined_request) &&
           make_request(bank.semantic_bank, astc_gpu_d1_candidate_family::scalar_quantile_refined,
                        astc_gpu_exact_subset_kind::d1_luminance_binary_quantile_refined_6x6,
                        max_blocks_per_batch, bank.quantile_refined_request);
}
