#include "astc-gpu-d2-neural-rank.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace {

float decoded_member(const astc_gpu_d2_candidate_record & record,
                     const astc_gpu_encoder_finished_block & block,
                     uint32_t texel, unsigned int member) {
    const auto * rgba = block.decoded_rgba.data() + texel * 4;
    return astc_vulkan_paired_weight(
        {rgba[0], rgba[1], rgba[2], rgba[3]}, member, record.layout,
        astc_vulkan_paired_basis::direct, record.semantic);
}

float source_member(const astc_gpu_d2_candidate_record & record,
                    const astc_gpu_encoder_source_block & block,
                    uint32_t texel, unsigned int member) {
    const auto & rgba = block.texels[texel].rgba;
    return astc_vulkan_paired_weight(
        {rgba[0], rgba[1], rgba[2], rgba[3]}, member, record.layout,
        astc_vulkan_paired_basis::direct, astc_vulkan_paired_semantic::direct_rgb);
}

} // namespace

bool astc_gpu_d2_rank_finished_candidates_activation(
    const astc_gpu_d2_candidate_bank & bank,
    const std::vector<astc_gpu_encoder_finished_block> & finished,
    const astc_gpu_d2_activation_rank_request & request,
    uint32_t max_candidates_per_logical,
    std::vector<astc_gpu_encoder_finished_block> & selected,
    std::vector<astc_gpu_d2_finished_candidate_score> & scores) {
    selected.clear();
    scores.clear();
    if (max_candidates_per_logical == 0 || request.tensor_width == 0 ||
        request.tensor_height == 0 || request.source_blocks_x == 0 ||
        request.activations.empty() ||
        request.activations.size() % request.tensor_width != 0 ||
        bank.records.empty()) return false;
    const auto format = astc_vulkan_format(bank.footprint);
    if (format.block_width == 0 || format.block_height == 0) return false;

    std::unordered_map<uint32_t, astc_gpu_d2_candidate_record> records;
    std::unordered_map<uint32_t, const astc_gpu_encoder_source_block *> source;
    for (size_t index = 0; index < bank.records.size(); ++index) {
        const auto & record = bank.records[index];
        records.emplace(record.candidate_source_block_id, record);
        if (record.family == astc_gpu_d2_candidate_family::direct_neutral)
            source.emplace(record.logical_source_block_id, &bank.candidate_blocks[index]);
    }
    std::unordered_map<uint32_t, const astc_gpu_encoder_finished_block *> decoded;
    for (const auto & block : finished) {
        if (block.footprint != bank.footprint || !records.count(block.source_block_id) ||
            block.decoded_rgba.size() != size_t(format.block_width) * format.block_height * 4 ||
            !decoded.emplace(block.source_block_id, &block).second) return false;
    }
    const uint32_t samples = static_cast<uint32_t>(request.activations.size() / request.tensor_width);
    for (const auto & block : finished) {
        const auto & record = records.at(block.source_block_id);
        const auto original = source.find(record.logical_source_block_id);
        if (original == source.end()) return false;
        const uint32_t block_x = record.logical_source_block_id % request.source_blocks_x;
        const uint32_t block_y = record.logical_source_block_id / request.source_blocks_x;
        double error = 0.0;
        for (uint32_t sample = 0; sample < samples; ++sample) {
            for (uint32_t y = 0; y < format.block_height; ++y) {
                const uint32_t texture_row = block_y * format.block_height + y;
                for (unsigned int member = 0; member < 2; ++member) {
                    const uint32_t row = texture_row * 2u + member;
                    if (row >= request.tensor_height) continue;
                    double output_error = 0.0;
                    for (uint32_t x = 0; x < format.block_width; ++x) {
                        const uint32_t column = block_x * format.block_width + x;
                        if (column >= request.tensor_width) continue;
                        const uint32_t texel = y * format.block_width + x;
                        const float delta = source_member(
                            record, *original->second, texel, member) -
                            decoded_member(record, block, texel, member);
                        output_error += delta * request.activations[
                            size_t(sample) * request.tensor_width + column];
                    }
                    error += output_error * output_error;
                }
            }
        }
        scores.push_back({block.source_block_id, record.logical_source_block_id,
                          record.family, record.layout, record.semantic, error});
    }

    std::unordered_map<uint32_t, std::vector<astc_gpu_d2_finished_candidate_score>> grouped;
    for (const auto & score : scores) grouped[score.logical_source_block_id].push_back(score);
    for (const auto & baseline : bank.records) {
        if (baseline.family != astc_gpu_d2_candidate_family::direct_neutral) continue;
        const auto group = grouped.find(baseline.logical_source_block_id);
        if (group == grouped.end()) return false;
        const auto mandatory = std::find_if(group->second.begin(), group->second.end(),
            [&](const auto & score) {
                return score.candidate_source_block_id == baseline.candidate_source_block_id;
            });
        if (mandatory == group->second.end()) return false;
        selected.push_back(*decoded.at(mandatory->candidate_source_block_id));
        std::vector<astc_gpu_d2_finished_candidate_score> alternatives;
        for (const auto & score : group->second)
            if (score.candidate_source_block_id != mandatory->candidate_source_block_id)
                alternatives.push_back(score);
        std::sort(alternatives.begin(), alternatives.end(), [](const auto & a, const auto & b) {
            if (a.activation_error != b.activation_error) return a.activation_error < b.activation_error;
            return a.candidate_source_block_id < b.candidate_source_block_id;
        });
        const uint32_t extras = max_candidates_per_logical - 1;
        for (uint32_t index = 0; index < std::min<uint32_t>(extras, alternatives.size()); ++index)
            selected.push_back(*decoded.at(alternatives[index].candidate_source_block_id));
    }
    return !selected.empty();
}
