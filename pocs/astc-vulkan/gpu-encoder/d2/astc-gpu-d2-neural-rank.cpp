#include "astc-gpu-d2-neural-rank.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace {

bool source_weight_for_row(const astc_gpu_d2_candidate_record & record,
                           const astc_gpu_encoder_source_block & block,
                           uint32_t width, uint32_t local_row, uint32_t x,
                           float & value) {
    uint32_t pair_slot = 0;
    unsigned int member = 0;
    if (!astc_gpu_d2_record_slot_for_row(record, local_row, pair_slot, member)) return false;
    const auto & rgba = block.texels[size_t(pair_slot) * width + x].rgba;
    float first = 0.0f, second = 0.0f;
    astc_gpu_d2_record_restore_pair(record, {rgba[0], rgba[1], rgba[2], rgba[3]}, first, second);
    value = member == 0 ? first : second;
    return true;
}

bool decoded_weight_for_row(const astc_gpu_d2_candidate_record & record,
                            const astc_gpu_encoder_finished_block & block,
                            uint32_t width, uint32_t local_row, uint32_t x,
                            float & value) {
    uint32_t pair_slot = 0;
    unsigned int member = 0;
    if (!astc_gpu_d2_record_slot_for_row(record, local_row, pair_slot, member)) return false;
    const auto * rgba = block.decoded_rgba.data() + (size_t(pair_slot) * width + x) * 4;
    float first = 0.0f, second = 0.0f;
    astc_gpu_d2_record_restore_pair(record, {rgba[0], rgba[1], rgba[2], rgba[3]}, first, second);
    value = member == 0 ? first : second;
    return true;
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
    if (format.block_width == 0 || format.block_height == 0 ||
        format.block_height * 2u != astc_vulkan_d2_pair_group_rows) return false;

    std::unordered_map<uint32_t, astc_gpu_d2_candidate_record> records;
    std::unordered_map<uint32_t, const astc_gpu_encoder_source_block *> source;
    std::unordered_map<uint32_t, astc_gpu_d2_candidate_record> source_records;
    for (size_t index = 0; index < bank.records.size(); ++index) {
        const auto & record = bank.records[index];
        records.emplace(record.candidate_source_block_id, record);
        if (record.family == astc_gpu_d2_candidate_family::direct_neutral) {
            source.emplace(record.logical_source_block_id, &bank.candidate_blocks[index]);
            source_records.emplace(record.logical_source_block_id, record);
        }
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
        const auto original_record = source_records.find(record.logical_source_block_id);
        if (original == source.end() || original_record == source_records.end()) return false;
        const uint32_t block_x = record.logical_source_block_id % request.source_blocks_x;
        const uint32_t block_y = record.logical_source_block_id / request.source_blocks_x;
        double error = 0.0;
        for (uint32_t sample = 0; sample < samples; ++sample) {
            for (uint32_t y = 0; y < format.block_height; ++y) {
                for (uint32_t local_row = 0; local_row < 2u; ++local_row) {
                    const uint32_t pair_row = 2u * y + local_row;
                    const uint32_t row = block_y * format.block_height * 2u + pair_row;
                    if (row >= request.tensor_height) continue;
                    double output_error = 0.0;
                    for (uint32_t x = 0; x < format.block_width; ++x) {
                        const uint32_t column = block_x * format.block_width + x;
                        if (column >= request.tensor_width) continue;
                        float original_weight = 0.0f;
                        float decoded_weight = 0.0f;
                        if (!source_weight_for_row(original_record->second, *original->second,
                                                   format.block_width, pair_row, x, original_weight) ||
                            !decoded_weight_for_row(record, block, format.block_width,
                                                    pair_row, x, decoded_weight)) return false;
                        const float delta = original_weight - decoded_weight;
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
