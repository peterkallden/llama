#include "astc-gpu-d2-candidates.h"

#include <algorithm>
#include <unordered_map>

namespace {

bool compatible_blocks(astc_vulkan_footprint footprint,
                       const std::vector<astc_gpu_encoder_source_block> & blocks,
                       size_t expected_count) {
    const auto format = astc_vulkan_format(footprint);
    if (blocks.size() != expected_count || format.block_width == 0 || format.block_height == 0) return false;
    for (const auto & block : blocks) {
        if (block.footprint != footprint ||
            block.texels.size() != size_t(format.block_width) * format.block_height) return false;
    }
    return true;
}

} // namespace

bool astc_gpu_d2_build_candidate_bank(
    astc_vulkan_footprint footprint,
    const std::vector<astc_gpu_encoder_source_block> & direct_neutral_blocks,
    const std::vector<astc_vulkan_paired_layout> & direct_neutral_layouts,
    const std::vector<astc_gpu_d2_candidate_family_sources> & alternatives,
    astc_gpu_d2_candidate_bank & bank) {
    bank = {};
    if (direct_neutral_blocks.empty() ||
        !compatible_blocks(footprint, direct_neutral_blocks, direct_neutral_blocks.size()) ||
        direct_neutral_layouts.size() != direct_neutral_blocks.size()) return false;
    for (const auto & alternative : alternatives) {
        if (alternative.family == astc_gpu_d2_candidate_family::direct_neutral ||
            !compatible_blocks(footprint, alternative.blocks, direct_neutral_blocks.size()) ||
            alternative.layouts.size() != direct_neutral_blocks.size()) return false;
    }
    bank.footprint = footprint;
    bank.candidate_blocks.reserve(direct_neutral_blocks.size() * (alternatives.size() + 1));
    bank.records.reserve(bank.candidate_blocks.capacity());
    uint32_t next_id = 0;
    for (size_t logical = 0; logical < direct_neutral_blocks.size(); ++logical) {
        const auto append = [&](const astc_gpu_encoder_source_block & source,
                                astc_gpu_d2_candidate_family family,
                                astc_vulkan_paired_layout layout,
                                astc_vulkan_paired_semantic semantic,
                                uint32_t candidate_index) {
            auto copy = source;
            copy.source_block_id = next_id;
            bank.candidate_blocks.push_back(std::move(copy));
            bank.records.push_back({next_id, direct_neutral_blocks[logical].source_block_id,
                                    candidate_index, family, layout, semantic});
            ++next_id;
        };
        append(direct_neutral_blocks[logical], astc_gpu_d2_candidate_family::direct_neutral,
               direct_neutral_layouts[logical], astc_vulkan_paired_semantic::direct_rgb, 0);
        for (uint32_t index = 0; index < alternatives.size(); ++index) {
            const auto & alternative = alternatives[index];
            append(alternative.blocks[logical], alternative.family, alternative.layouts[logical],
                   alternative.semantic, index + 1);
        }
    }
    return true;
}

bool astc_gpu_d2_candidate_bank_request(
    const astc_gpu_d2_candidate_bank & bank, uint32_t max_blocks_per_batch,
    astc_gpu_encoder_request & request) {
    if (bank.candidate_blocks.empty() || bank.records.size() != bank.candidate_blocks.size() ||
        max_blocks_per_batch == 0) return false;
    request = {};
    request.footprint = bank.footprint;
    request.max_blocks_per_batch = max_blocks_per_batch;
    request.blocks = bank.candidate_blocks;
    request.candidate_metadata.reserve(bank.records.size());
    for (const auto & record : bank.records) {
        request.candidate_metadata.push_back({record.candidate_source_block_id,
                                              record.logical_source_block_id,
                                              record.candidate_index,
                                              static_cast<uint32_t>(record.family)});
    }
    return true;
}

bool astc_gpu_d2_select_candidate_bank_proposals(
    const astc_gpu_d2_candidate_bank & bank,
    const std::vector<astc_gpu_encoder_proposal> & proposals,
    uint32_t max_candidates_per_logical,
    std::vector<astc_gpu_encoder_proposal> & selected) {
    selected.clear();
    if (max_candidates_per_logical == 0 || bank.records.empty()) return false;
    std::unordered_map<uint32_t, astc_gpu_d2_candidate_record> records;
    for (const auto & record : bank.records) records.emplace(record.candidate_source_block_id, record);
    std::unordered_map<uint32_t, std::vector<astc_gpu_encoder_proposal>> grouped;
    for (const auto & proposal : proposals) {
        const auto record = records.find(proposal.source_block_id);
        if (record == records.end()) return false;
        grouped[record->second.logical_source_block_id].push_back(proposal);
    }
    for (const auto & baseline : bank.records) {
        if (baseline.family != astc_gpu_d2_candidate_family::direct_neutral) continue;
        const auto group = grouped.find(baseline.logical_source_block_id);
        if (group == grouped.end()) return false;
        const auto mandatory = std::find_if(group->second.begin(), group->second.end(), [&](const auto & proposal) {
            return proposal.source_block_id == baseline.candidate_source_block_id;
        });
        if (mandatory == group->second.end()) return false;
        selected.push_back(*mandatory);
        std::vector<astc_gpu_encoder_proposal> alternatives;
        for (const auto & proposal : group->second) if (proposal.source_block_id != mandatory->source_block_id) alternatives.push_back(proposal);
        std::sort(alternatives.begin(), alternatives.end(), [](const auto & a, const auto & b) {
            if (a.approximate_error != b.approximate_error) return a.approximate_error < b.approximate_error;
            return a.source_block_id < b.source_block_id;
        });
        const uint32_t count = std::min<uint32_t>(max_candidates_per_logical - 1, alternatives.size());
        selected.insert(selected.end(), alternatives.begin(), alternatives.begin() + count);
    }
    return !selected.empty();
}
