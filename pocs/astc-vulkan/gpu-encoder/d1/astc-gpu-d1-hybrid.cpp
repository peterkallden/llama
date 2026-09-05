#include "astc-gpu-d1-hybrid.h"

#include <unordered_map>

namespace {

float decoded_weight(astc_gpu_d1_candidate_family family,
                     const astc_gpu_encoder_finished_block & block,
                     uint32_t texel) {
    const float * rgba = block.decoded_rgba.data() + texel * 4;
    return family == astc_gpu_d1_candidate_family::gauge_la ?
        0.5f * (rgba[0] + rgba[3]) :
        (rgba[0] + rgba[1] + rgba[2]) / 3.0f;
}

std::vector<double> make_delta(
    astc_gpu_d1_candidate_family candidate_family,
    const astc_gpu_encoder_finished_block & candidate,
    astc_gpu_d1_candidate_family baseline_family,
    const astc_gpu_encoder_finished_block & baseline,
    const std::vector<double> & activations,
    uint32_t width, uint32_t height, uint32_t source_blocks_x,
    uint32_t block_width, uint32_t block_height, uint32_t logical_block) {
    const uint32_t samples = static_cast<uint32_t>(activations.size() / width);
    std::vector<double> delta(size_t(samples) * height, 0.0);
    const uint32_t block_x = logical_block % source_blocks_x;
    const uint32_t block_y = logical_block / source_blocks_x;
    for (uint32_t sample = 0; sample < samples; ++sample) {
        for (uint32_t y = 0; y < block_height; ++y) {
            const uint32_t row = block_y * block_height + y;
            if (row >= height) continue;
            for (uint32_t x = 0; x < block_width; ++x) {
                const uint32_t column = block_x * block_width + x;
                if (column >= width) continue;
                const uint32_t texel = y * block_width + x;
                delta[size_t(sample) * height + row] +=
                    (decoded_weight(candidate_family, candidate, texel) -
                     decoded_weight(baseline_family, baseline, texel)) *
                    activations[size_t(sample) * width + column];
            }
        }
    }
    return delta;
}

} // namespace

bool astc_gpu_d1_make_selector_candidates(
    const astc_gpu_d1_candidate_bank & bank,
    const std::vector<astc_gpu_encoder_finished_block> & finished,
    const astc_gpu_d1_selector_delta_request & request,
    std::vector<std::vector<astc_vulkan_paired_candidate_delta>> & candidates,
    std::string & error) {
    candidates.clear();
    if (bank.records.empty() || finished.empty() ||
        !astc_gpu_validate_selector_delta_request(request, "D1", error)) {
        if (error.empty()) error = "invalid D1 selector delta request";
        return false;
    }
    const auto format = astc_vulkan_format(bank.footprint);
    if (format.block_width == 0 || format.block_height == 0) {
        error = "unsupported D1 selector footprint";
        return false;
    }
    std::unordered_map<uint32_t, astc_gpu_d1_candidate_record> records;
    for (const auto & record : bank.records) records.emplace(record.candidate_source_block_id, record);
    std::unordered_map<uint32_t, const astc_gpu_encoder_finished_block *> decoded;
    for (const auto & block : finished) {
        if (!records.count(block.source_block_id) || block.decoded_rgba.size() !=
                size_t(format.block_width) * format.block_height * 4 ||
            !decoded.emplace(block.source_block_id, &block).second) {
            error = "malformed D1 finished candidate";
            return false;
        }
    }
    for (const auto & baseline_record : bank.records) {
        if (baseline_record.family != astc_gpu_d1_candidate_family::scalar) continue;
        const auto baseline = decoded.find(baseline_record.candidate_source_block_id);
        if (baseline == decoded.end()) {
            error = "D1 selector bank is missing scalar baseline";
            return false;
        }
        std::vector<astc_vulkan_paired_candidate_delta> group;
        for (const auto & record : bank.records) {
            if (record.logical_source_block_id != baseline_record.logical_source_block_id) continue;
            const auto block = decoded.find(record.candidate_source_block_id);
            if (block == decoded.end()) continue;
            astc_vulkan_paired_candidate_delta delta;
            delta.payload = block->second->payload;
            if (record.candidate_source_block_id == baseline_record.candidate_source_block_id) {
                delta.calibration_delta = astc_gpu_zero_selector_delta(request, false);
                if (!request.validation_activations.empty())
                    delta.validation_delta = astc_gpu_zero_selector_delta(request, true);
            } else {
                delta.calibration_delta = make_delta(
                    record.family, *block->second, baseline_record.family, *baseline->second,
                    request.calibration_activations, request.tensor_width,
                    request.tensor_height, request.source_blocks_x,
                    format.block_width, format.block_height,
                    baseline_record.logical_source_block_id);
                if (!request.validation_activations.empty()) delta.validation_delta = make_delta(
                    record.family, *block->second, baseline_record.family, *baseline->second,
                    request.validation_activations, request.tensor_width,
                    request.tensor_height, request.source_blocks_x,
                    format.block_width, format.block_height,
                    baseline_record.logical_source_block_id);
            }
            group.push_back(std::move(delta));
        }
        if (!astc_gpu_finalize_selector_group(
                group, baseline->second->payload, request, "D1", error)) return false;
        candidates.push_back(std::move(group));
    }
    if (candidates.empty()) {
        error = "D1 selector bank has no logical blocks";
        return false;
    }
    return true;
}
