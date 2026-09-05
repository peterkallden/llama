#include "astc-gpu-d1-candidates.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace {

bool compatible_blocks(astc_vulkan_footprint footprint,
                       const std::vector<astc_gpu_encoder_source_block> & blocks,
                       size_t expected_count) {
    if (blocks.size() != expected_count) return false;
    const auto format = astc_vulkan_format(footprint);
    if (format.block_width == 0 || format.block_height == 0) return false;
    for (const auto & block : blocks) {
        if (block.footprint != footprint ||
            block.texels.size() != size_t(format.block_width) * format.block_height) return false;
    }
    return true;
}

} // namespace

bool astc_gpu_d1_build_candidate_bank(
    astc_vulkan_footprint footprint,
    const std::vector<astc_gpu_encoder_source_block> & scalar_blocks,
    const std::vector<astc_gpu_d1_candidate_family_sources> & alternatives,
    astc_gpu_d1_candidate_bank & bank) {
    bank = {};
    if (scalar_blocks.empty() || !compatible_blocks(footprint, scalar_blocks, scalar_blocks.size())) return false;
    for (const auto & alternative : alternatives) {
        if (alternative.family == astc_gpu_d1_candidate_family::scalar ||
            !compatible_blocks(footprint, alternative.blocks, scalar_blocks.size())) return false;
    }
    bank.footprint = footprint;
    bank.candidate_blocks.reserve(scalar_blocks.size() * (alternatives.size() + 1));
    bank.records.reserve(bank.candidate_blocks.capacity());
    uint32_t next_id = 0;
    for (size_t logical_index = 0; logical_index < scalar_blocks.size(); ++logical_index) {
        const auto append = [&](const astc_gpu_encoder_source_block & source,
                                astc_gpu_d1_candidate_family family, uint32_t candidate_index) {
            auto copy = source;
            copy.source_block_id = next_id;
            bank.candidate_blocks.push_back(std::move(copy));
            bank.records.push_back({next_id, scalar_blocks[logical_index].source_block_id,
                                    candidate_index, family});
            ++next_id;
        };
        append(scalar_blocks[logical_index], astc_gpu_d1_candidate_family::scalar, 0);
        for (uint32_t family_index = 0; family_index < alternatives.size(); ++family_index) {
            append(alternatives[family_index].blocks[logical_index],
                   alternatives[family_index].family, family_index + 1);
        }
    }
    return true;
}

bool astc_gpu_d1_candidate_bank_request(
    const astc_gpu_d1_candidate_bank & bank, uint32_t max_blocks_per_batch,
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

bool astc_gpu_d1_select_candidate_bank_proposals(
    const astc_gpu_d1_candidate_bank & bank,
    const std::vector<astc_gpu_encoder_proposal> & proposals,
    uint32_t max_candidates_per_logical,
    std::vector<astc_gpu_encoder_proposal> & selected) {
    selected.clear();
    if (max_candidates_per_logical == 0 || bank.records.empty()) return false;
    std::unordered_map<uint32_t, astc_gpu_d1_candidate_record> records;
    records.reserve(bank.records.size());
    for (const auto & record : bank.records) records.emplace(record.candidate_source_block_id, record);
    std::unordered_map<uint32_t, std::vector<astc_gpu_encoder_proposal>> grouped;
    for (const auto & proposal : proposals) {
        if (!records.count(proposal.source_block_id)) return false;
        grouped[records.at(proposal.source_block_id).logical_source_block_id].push_back(proposal);
    }
    for (const auto & scalar_record : bank.records) {
        if (scalar_record.family != astc_gpu_d1_candidate_family::scalar) continue;
        const auto found = grouped.find(scalar_record.logical_source_block_id);
        if (found == grouped.end()) return false;
        const auto scalar = std::find_if(found->second.begin(), found->second.end(),
            [&](const auto & proposal) { return proposal.source_block_id == scalar_record.candidate_source_block_id; });
        if (scalar == found->second.end()) return false;
        selected.push_back(*scalar);
        std::vector<astc_gpu_encoder_proposal> alternatives;
        for (const auto & proposal : found->second) if (proposal.source_block_id != scalar->source_block_id) alternatives.push_back(proposal);
        std::sort(alternatives.begin(), alternatives.end(), [](const auto & a, const auto & b) {
            if (a.approximate_error != b.approximate_error) return a.approximate_error < b.approximate_error;
            return a.source_block_id < b.source_block_id;
        });
        const uint32_t extra = max_candidates_per_logical - 1;
        selected.insert(selected.end(), alternatives.begin(), alternatives.begin() + std::min<uint32_t>(extra, alternatives.size()));
    }
    return !selected.empty();
}
