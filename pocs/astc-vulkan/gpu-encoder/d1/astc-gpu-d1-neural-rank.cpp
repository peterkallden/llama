#include "astc-gpu-d1-neural-rank.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace {

float decoded_weight(astc_gpu_d1_candidate_family family,
                     const astc_gpu_encoder_finished_block & block,
                     uint32_t texel) {
    const float * rgba = block.decoded_rgba.data() + texel * 4;
    if (family == astc_gpu_d1_candidate_family::gauge_la) {
        return 0.5f * (rgba[0] + rgba[3]);
    }
    return (rgba[0] + rgba[1] + rgba[2]) / 3.0f;
}

} // namespace

bool astc_gpu_d1_rank_finished_candidates_activation(
    const astc_gpu_d1_candidate_bank & bank,
    const std::vector<astc_gpu_encoder_finished_block> & finished,
    const astc_gpu_d1_activation_rank_request & request,
    uint32_t max_candidates_per_logical,
    std::vector<astc_gpu_encoder_finished_block> & selected,
    std::vector<astc_gpu_d1_finished_candidate_score> & scores) {
    selected.clear();
    scores.clear();
    if (max_candidates_per_logical == 0 || request.tensor_width == 0 || request.tensor_height == 0 ||
        request.source_blocks_x == 0 || request.activations.empty() ||
        request.activations.size() % request.tensor_width != 0) return false;
    const auto format = astc_vulkan_format(bank.footprint);
    if (format.block_width == 0 || format.block_height == 0 || bank.records.empty()) return false;
    std::unordered_map<uint32_t, astc_gpu_d1_candidate_record> records;
    std::unordered_map<uint32_t, const astc_gpu_encoder_source_block *> scalar_source;
    for (size_t index = 0; index < bank.records.size(); ++index) {
        const auto & record = bank.records[index];
        records.emplace(record.candidate_source_block_id, record);
        if (record.family == astc_gpu_d1_candidate_family::scalar) {
            scalar_source.emplace(record.logical_source_block_id, &bank.candidate_blocks[index]);
        }
    }
    std::unordered_map<uint32_t, const astc_gpu_encoder_finished_block *> finished_by_id;
    for (const auto & block : finished) {
        if (block.footprint != bank.footprint || !records.count(block.source_block_id) ||
            !finished_by_id.emplace(block.source_block_id, &block).second ||
            block.decoded_rgba.size() != size_t(format.block_width) * format.block_height * 4) return false;
    }
    const uint32_t samples = static_cast<uint32_t>(request.activations.size() / request.tensor_width);
    for (const auto & block : finished) {
        const auto & record = records.at(block.source_block_id);
        const auto source = scalar_source.find(record.logical_source_block_id);
        if (source == scalar_source.end()) return false;
        const uint32_t block_x = record.logical_source_block_id % request.source_blocks_x;
        const uint32_t block_y = record.logical_source_block_id / request.source_blocks_x;
        double error = 0.0;
        for (uint32_t sample = 0; sample < samples; ++sample) {
            for (uint32_t y = 0; y < format.block_height; ++y) {
                const uint32_t row = block_y * format.block_height + y;
                if (row >= request.tensor_height) continue;
                // Rows are output-orthogonal for the one-sided activation
                // objective. Do not allow an error on one row to cancel an
                // error on another row before squaring.
                double output_error = 0.0;
                for (uint32_t x = 0; x < format.block_width; ++x) {
                    const uint32_t column = block_x * format.block_width + x;
                    if (column >= request.tensor_width) continue;
                    const uint32_t texel = y * format.block_width + x;
                    const float original = source->second->texels[texel].rgba[0];
                    output_error += (original - decoded_weight(record.family, block, texel)) *
                        request.activations[size_t(sample) * request.tensor_width + column];
                }
                error += output_error * output_error;
            }
        }
        scores.push_back({block.source_block_id, record.logical_source_block_id, record.family, error});
    }
    std::unordered_map<uint32_t, std::vector<astc_gpu_d1_finished_candidate_score>> grouped;
    for (const auto & score : scores) grouped[score.logical_source_block_id].push_back(score);
    std::unordered_map<uint32_t, const astc_gpu_encoder_finished_block *> finished_lookup = finished_by_id;
    for (const auto & record : bank.records) {
        if (record.family != astc_gpu_d1_candidate_family::scalar) continue;
        const auto group = grouped.find(record.logical_source_block_id);
        if (group == grouped.end()) return false;
        const auto scalar = std::find_if(group->second.begin(), group->second.end(), [&](const auto & score) {
            return score.candidate_source_block_id == record.candidate_source_block_id;
        });
        if (scalar == group->second.end()) return false;
        selected.push_back(*finished_lookup.at(scalar->candidate_source_block_id));
        std::vector<astc_gpu_d1_finished_candidate_score> alternatives;
        for (const auto & score : group->second) if (score.candidate_source_block_id != scalar->candidate_source_block_id) alternatives.push_back(score);
        std::sort(alternatives.begin(), alternatives.end(), [](const auto & a, const auto & b) {
            if (a.activation_error != b.activation_error) return a.activation_error < b.activation_error;
            return a.candidate_source_block_id < b.candidate_source_block_id;
        });
        const uint32_t extras = max_candidates_per_logical - 1;
        for (uint32_t index = 0; index < std::min<uint32_t>(extras, alternatives.size()); ++index) {
            selected.push_back(*finished_lookup.at(alternatives[index].candidate_source_block_id));
        }
    }
    return !selected.empty();
}
