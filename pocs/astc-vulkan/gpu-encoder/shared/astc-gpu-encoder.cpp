#include "astc-gpu-encoder.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>

namespace {

bool valid_block(const astc_gpu_encoder_source_block & block,
                 astc_vulkan_footprint expected) {
    if (block.footprint != expected) return false;
    const auto format = astc_vulkan_format(expected);
    const uint32_t width = format.block_width;
    const uint32_t height = format.block_height;
    return width != 0 && height != 0 && block.texels.size() == size_t(width) * height;
}

} // namespace

bool astc_gpu_encoder_plan_batches(const astc_gpu_encoder_request & request,
                                   std::vector<astc_gpu_encoder_batch> & batches) {
    batches.clear();
    if (request.mode != astc_gpu_encode_mode::propose || request.max_blocks_per_batch == 0) return false;
    if (!request.candidate_metadata.empty() &&
        request.candidate_metadata.size() != request.blocks.size()) return false;
    std::unordered_set<uint32_t> source_ids;
    source_ids.reserve(request.blocks.size());
    for (size_t index = 0; index < request.blocks.size(); ++index) {
        if (!valid_block(request.blocks[index], request.footprint) ||
            !source_ids.insert(request.blocks[index].source_block_id).second) return false;
        if (!request.candidate_metadata.empty() &&
            request.candidate_metadata[index].source_block_id != request.blocks[index].source_block_id) return false;
    }
    for (uint32_t first = 0; first < request.blocks.size(); first += request.max_blocks_per_batch) {
        batches.push_back({first, std::min<uint32_t>(request.max_blocks_per_batch,
                                                      request.blocks.size() - first)});
    }
    return true;
}

bool astc_gpu_encoder_propose_cpu_reference(
    const astc_gpu_encoder_request & request,
    std::vector<astc_gpu_encoder_proposal> & proposals) {
    std::vector<astc_gpu_encoder_batch> batches;
    if (!astc_gpu_encoder_plan_batches(request, batches)) return false;
    proposals.clear();
    proposals.reserve(request.blocks.size());
    for (const auto & block : request.blocks) {
        astc_gpu_encoder_proposal proposal;
        proposal.source_block_id = block.source_block_id;
        const auto format = astc_vulkan_format(block.footprint);
        proposal.weight_grid_x = format.block_width;
        proposal.weight_grid_y = format.block_height;
        proposal.endpoint_low.fill(std::numeric_limits<float>::infinity());
        proposal.endpoint_high.fill(-std::numeric_limits<float>::infinity());
        std::array<double, 4> mean{};
        for (const auto & texel : block.texels) {
            for (uint32_t channel = 0; channel < 4; ++channel) {
                const float value = texel.rgba[channel];
                if (!std::isfinite(value)) return false;
                proposal.endpoint_low[channel] = std::min(proposal.endpoint_low[channel], value);
                proposal.endpoint_high[channel] = std::max(proposal.endpoint_high[channel], value);
                mean[channel] += value;
            }
        }
        for (double & value : mean) value /= static_cast<double>(block.texels.size());
        double squared_error = 0.0;
        for (const auto & texel : block.texels) for (uint32_t channel = 0; channel < 4; ++channel) {
            const double delta = texel.rgba[channel] - mean[channel];
            squared_error += delta * delta;
        }
        proposal.approximate_error = static_cast<float>(squared_error / block.texels.size());
        proposals.push_back(proposal);
    }
    return true;
}
