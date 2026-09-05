#include "astc-gpu-d2-hybrid.h"

#include <unordered_map>

namespace {

float paired_member(const astc_gpu_d2_candidate_record & record,
                    const astc_gpu_encoder_finished_block & block,
                    uint32_t texel, unsigned int member) {
    const auto * rgba = block.decoded_rgba.data() + texel * 4;
    return astc_vulkan_paired_weight(
        {rgba[0], rgba[1], rgba[2], rgba[3]}, member, record.layout,
        astc_vulkan_paired_basis::direct, record.semantic);
}

std::vector<double> make_delta(
    const astc_gpu_d2_candidate_record & candidate_record,
    const astc_gpu_encoder_finished_block & candidate,
    const astc_gpu_d2_candidate_record & baseline_record,
    const astc_gpu_encoder_finished_block & baseline,
    const std::vector<double> & activations,
    uint32_t width, uint32_t height, uint32_t source_blocks_x,
    uint32_t block_width, uint32_t block_height) {
    const uint32_t samples = static_cast<uint32_t>(activations.size() / width);
    std::vector<double> delta(size_t(samples) * height, 0.0);
    const uint32_t logical_block = candidate_record.logical_source_block_id;
    const uint32_t block_x = logical_block % source_blocks_x;
    const uint32_t block_y = logical_block / source_blocks_x;
    for (uint32_t sample = 0; sample < samples; ++sample) {
        for (uint32_t y = 0; y < block_height; ++y) {
            const uint32_t texture_row = block_y * block_height + y;
            for (unsigned int member = 0; member < 2; ++member) {
                const uint32_t row = texture_row * 2u + member;
                if (row >= height) continue;
                double output_delta = 0.0;
                for (uint32_t x = 0; x < block_width; ++x) {
                    const uint32_t column = block_x * block_width + x;
                    if (column >= width) continue;
                    const uint32_t texel = y * block_width + x;
                    const double weight_delta =
                        paired_member(candidate_record, candidate, texel, member) -
                        paired_member(baseline_record, baseline, texel, member);
                    output_delta += weight_delta * activations[
                        size_t(sample) * width + column];
                }
                delta[size_t(sample) * height + row] = output_delta;
            }
        }
    }
    return delta;
}

} // namespace

bool astc_gpu_d2_finish_and_rank(
    const astc_gpu_d2_candidate_bank & bank,
    const std::vector<astc_gpu_encoder_proposal> & proposals,
    const astc_gpu_d2_activation_rank_request & rank_request,
    const astc_gpu_d2_hybrid_options & options,
    astc_gpu_d2_hybrid_result & result,
    std::string & error) {
    result = {};
    if (options.max_finish_candidates_per_logical == 0 ||
        options.max_ranked_candidates_per_logical == 0 ||
        options.max_ranked_candidates_per_logical >
            options.max_finish_candidates_per_logical) {
        error = "invalid D2 hybrid candidate budgets";
        return false;
    }

    if (!astc_gpu_d2_select_candidate_bank_proposals(
            bank, proposals, options.max_finish_candidates_per_logical,
            result.retained_proposals)) {
        error = "D2 hybrid proposal retention failed";
        return false;
    }

    astc_gpu_encoder_request request;
    if (!astc_gpu_d2_candidate_bank_request(
            bank, static_cast<uint32_t>(bank.candidate_blocks.size()), request)) {
        error = "D2 hybrid request construction failed";
        return false;
    }
    if (!astc_gpu_encoder_finish(
            request, result.retained_proposals, options.astcenc_quality,
            result.finished_blocks, error)) return false;

    if (!astc_gpu_d2_rank_finished_candidates_activation(
            bank, result.finished_blocks, rank_request,
            options.max_ranked_candidates_per_logical,
            result.ranked_blocks, result.activation_scores)) {
        error = "D2 hybrid exact activation ranking failed";
        return false;
    }
    return true;
}

bool astc_gpu_d2_make_selector_candidates(
    const astc_gpu_d2_candidate_bank & bank,
    const std::vector<astc_gpu_encoder_finished_block> & finished,
    const astc_gpu_d2_selector_delta_request & request,
    std::vector<std::vector<astc_vulkan_paired_candidate_delta>> & candidates,
    std::string & error) {
    candidates.clear();
    if (bank.records.empty() || finished.empty() ||
        !astc_gpu_validate_selector_delta_request(request, "D2", error)) {
        if (error.empty()) error = "invalid D2 selector delta request";
        return false;
    }
    const auto format = astc_vulkan_format(bank.footprint);
    if (format.block_width == 0 || format.block_height == 0) {
        error = "unsupported D2 selector footprint";
        return false;
    }
    std::unordered_map<uint32_t, const astc_gpu_d2_candidate_record *> records;
    for (const auto & record : bank.records) records.emplace(record.candidate_source_block_id, &record);
    std::unordered_map<uint32_t, const astc_gpu_encoder_finished_block *> decoded;
    for (const auto & block : finished) {
        const auto record = records.find(block.source_block_id);
        if (record == records.end() || block.decoded_rgba.size() !=
                size_t(format.block_width) * format.block_height * 4 ||
            !decoded.emplace(block.source_block_id, &block).second) {
            error = "malformed D2 finished candidate";
            return false;
        }
    }
    candidates.reserve(bank.candidate_blocks.size());
    for (const auto & baseline_record : bank.records) {
        if (baseline_record.family != astc_gpu_d2_candidate_family::direct_neutral) continue;
        const auto baseline = decoded.find(baseline_record.candidate_source_block_id);
        if (baseline == decoded.end()) {
            error = "D2 selector bank is missing neutral baseline";
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
                    record, *block->second, baseline_record, *baseline->second,
                    request.calibration_activations, request.tensor_width,
                    request.tensor_height, request.source_blocks_x,
                    format.block_width, format.block_height);
                if (!request.validation_activations.empty()) delta.validation_delta = make_delta(
                    record, *block->second, baseline_record, *baseline->second,
                    request.validation_activations, request.tensor_width,
                    request.tensor_height, request.source_blocks_x,
                    format.block_width, format.block_height);
            }
            group.push_back(std::move(delta));
        }
        // Reorder defensively; candidate-bank order is not part of the
        // representation-neutral selector contract.
        if (!astc_gpu_finalize_selector_group(
                group, baseline->second->payload, request, "D2", error)) return false;
        candidates.push_back(std::move(group));
    }
    if (candidates.empty()) {
        error = "D2 selector bank has no logical blocks";
        return false;
    }
    return true;
}
