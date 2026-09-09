#include "astc-gpu-d2-exact-subset.h"

#include <cmath>

namespace {

bool is_luminance_alpha_source(const astc_gpu_encoder_source_block & block) {
    const auto format = astc_vulkan_format(block.footprint);
    if ((block.footprint != astc_vulkan_footprint::k6x5 &&
         block.footprint != astc_vulkan_footprint::k8x5 &&
         block.footprint != astc_vulkan_footprint::k10x5) ||
        block.texels.size() != size_t(format.block_width) * format.block_height) return false;
    for (const auto & texel : block.texels) {
        for (const float value : texel.rgba) {
            if (!std::isfinite(value) || value < 0.0f || value > 1.0f) return false;
        }
        if (std::fabs(texel.rgba[0] - texel.rgba[1]) > 1e-6f ||
            std::fabs(texel.rgba[0] - texel.rgba[2]) > 1e-6f) return false;
    }
    return true;
}

astc_gpu_exact_subset_kind one_plane_kind(astc_vulkan_footprint footprint) {
    switch (footprint) {
    case astc_vulkan_footprint::k6x5: return astc_gpu_exact_subset_kind::luminance_alpha_binary_6x5;
    case astc_vulkan_footprint::k8x5: return astc_gpu_exact_subset_kind::luminance_alpha_binary_8x5;
    case astc_vulkan_footprint::k10x5: return astc_gpu_exact_subset_kind::luminance_alpha_binary_10x5;
    default: return astc_gpu_exact_subset_kind::void_extent_unorm16;
    }
}

astc_gpu_exact_subset_kind dual_plane_kind(astc_vulkan_footprint footprint) {
    switch (footprint) {
    case astc_vulkan_footprint::k6x5: return astc_gpu_exact_subset_kind::luminance_alpha_dual_binary_6x5;
    case astc_vulkan_footprint::k8x5: return astc_gpu_exact_subset_kind::luminance_alpha_dual_binary_8x5;
    case astc_vulkan_footprint::k10x5: return astc_gpu_exact_subset_kind::luminance_alpha_dual_binary_10x5;
    default: return astc_gpu_exact_subset_kind::void_extent_unorm16;
    }
}

} // namespace

bool astc_gpu_d2_build_luminance_alpha_exact_subset_bank(
    const std::vector<astc_gpu_encoder_source_block> & source_blocks,
    uint32_t max_blocks_per_batch,
    astc_gpu_d2_exact_subset_bank & bank) {
    bank = {};
    if (source_blocks.empty() || max_blocks_per_batch == 0) return false;
    for (const auto & block : source_blocks) if (!is_luminance_alpha_source(block)) return false;

    const auto footprint = source_blocks.front().footprint;
    const auto make_request = [&](astc_gpu_exact_subset_kind mode) {
        astc_gpu_encoder_request request;
        request.footprint = footprint;
        request.mode = astc_gpu_encode_mode::exact_subset;
        request.exact_subset = mode;
        request.max_blocks_per_batch = max_blocks_per_batch;
        request.blocks = source_blocks;
        return request;
    };
    bank.one_plane = make_request(one_plane_kind(footprint));
    bank.alpha_dual_plane = make_request(dual_plane_kind(footprint));
    // The extra fitting controls are audited only for 8x5 today. The H5
    // profile remains useful at 6x5/10x5 with its base and dual-plane modes,
    // without pretending that 8x5's refinement has been ported.
    if (footprint == astc_vulkan_footprint::k8x5) {
        bank.one_plane_luminance_weights = make_request(
            astc_gpu_exact_subset_kind::luminance_alpha_binary_luminance_weights_8x5);
        bank.one_plane_alpha_weights = make_request(
            astc_gpu_exact_subset_kind::luminance_alpha_binary_alpha_weights_8x5);
        bank.one_plane_refined = make_request(
            astc_gpu_exact_subset_kind::luminance_alpha_binary_refined_8x5);
        bank.one_plane_mean_refined = make_request(
            astc_gpu_exact_subset_kind::luminance_alpha_binary_mean_refined_8x5);
        bank.one_plane_quantile_refined = make_request(
            astc_gpu_exact_subset_kind::luminance_alpha_binary_quantile_refined_8x5);
    }
    return true;
}

bool astc_gpu_d2_build_luminance_alpha_exact_subset_candidate_bank(
    const std::vector<astc_gpu_encoder_source_block> & source_blocks,
    const std::vector<astc_vulkan_paired_layout> & layouts,
    const std::vector<astc_vulkan_d2_pairing> & pairings,
    uint32_t max_blocks_per_batch,
    astc_gpu_d2_exact_subset_candidate_bank & bank) {
    bank = {};
    astc_gpu_d2_exact_subset_bank ignored_source_requests;
    if (!astc_gpu_d2_build_luminance_alpha_exact_subset_bank(
            source_blocks, max_blocks_per_batch, ignored_source_requests) ||
        layouts.size() != source_blocks.size()) return false;
    const std::vector<astc_gpu_d2_candidate_family_sources> alternatives{
        {astc_gpu_d2_candidate_family::luminance_alpha,
         astc_vulkan_paired_semantic::luminance_alpha, source_blocks, layouts},
        {astc_gpu_d2_candidate_family::luminance_alpha,
         astc_vulkan_paired_semantic::luminance_alpha, source_blocks, layouts},
        {astc_gpu_d2_candidate_family::luminance_alpha,
         astc_vulkan_paired_semantic::luminance_alpha, source_blocks, layouts},
        {astc_gpu_d2_candidate_family::luminance_alpha,
         astc_vulkan_paired_semantic::luminance_alpha, source_blocks, layouts},
    };
    if (!astc_gpu_d2_build_candidate_bank(
            astc_vulkan_footprint::k8x5, source_blocks, layouts, alternatives,
            bank.semantic_bank, pairings,
            astc_vulkan_paired_semantic::luminance_alpha)) return false;

    const auto make_request = [&](uint32_t candidate_index, astc_gpu_exact_subset_kind mode,
                                  astc_gpu_encoder_request & request) {
        request = {};
        request.mode = astc_gpu_encode_mode::exact_subset;
        request.footprint = astc_vulkan_footprint::k8x5;
        request.exact_subset = mode;
        request.max_blocks_per_batch = max_blocks_per_batch;
        for (size_t index = 0; index < bank.semantic_bank.records.size(); ++index) {
            const auto & record = bank.semantic_bank.records[index];
            if (record.candidate_index != candidate_index) continue;
            request.blocks.push_back(bank.semantic_bank.candidate_blocks[index]);
            request.candidate_metadata.push_back({record.candidate_source_block_id,
                                                  record.logical_source_block_id,
                                                  record.candidate_index,
                                                  static_cast<uint32_t>(record.family)});
        }
        return !request.blocks.empty();
    };
    return make_request(0, astc_gpu_exact_subset_kind::luminance_alpha_binary_8x5,
                        bank.physical_bank.one_plane) &&
        make_request(1, astc_gpu_exact_subset_kind::luminance_alpha_binary_luminance_weights_8x5,
                     bank.physical_bank.one_plane_luminance_weights) &&
        make_request(2, astc_gpu_exact_subset_kind::luminance_alpha_binary_alpha_weights_8x5,
                     bank.physical_bank.one_plane_alpha_weights) &&
        make_request(3, astc_gpu_exact_subset_kind::luminance_alpha_binary_refined_8x5,
                     bank.physical_bank.one_plane_refined) &&
        make_request(4, astc_gpu_exact_subset_kind::luminance_alpha_dual_binary_8x5,
                     bank.physical_bank.alpha_dual_plane);
}
