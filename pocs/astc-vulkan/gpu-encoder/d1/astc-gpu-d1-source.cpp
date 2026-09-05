#include "astc-gpu-d1-source.h"

#include <cmath>

bool astc_gpu_d1_build_scalar_source_blocks(
    astc_vulkan_footprint footprint,
    const std::vector<float> & normalized_weights,
    uint32_t rows,
    uint32_t columns,
    std::vector<astc_gpu_encoder_source_block> & blocks) {
    blocks.clear();
    const auto format = astc_vulkan_format(footprint);
    const uint32_t block_width = format.block_width;
    const uint32_t block_height = format.block_height;
    if (block_width == 0 || block_height == 0 || rows == 0 || columns == 0 ||
        normalized_weights.size() != size_t(rows) * columns) return false;
    const uint32_t blocks_x = (columns + block_width - 1) / block_width;
    const uint32_t blocks_y = (rows + block_height - 1) / block_height;
    blocks.reserve(size_t(blocks_x) * blocks_y);
    for (uint32_t by = 0; by < blocks_y; ++by) for (uint32_t bx = 0; bx < blocks_x; ++bx) {
        astc_gpu_encoder_source_block block;
        block.footprint = footprint;
        block.source_block_id = static_cast<uint32_t>(blocks.size());
        block.texels.resize(size_t(block_width) * block_height);
        // Edge padding is deterministic scalar zero; edge loss/selection is a
        // later semantic concern and is deliberately outside this physical API.
        for (uint32_t y = 0; y < block_height; ++y) for (uint32_t x = 0; x < block_width; ++x) {
            const uint32_t row = by * block_height + y;
            const uint32_t column = bx * block_width + x;
            float weight = 0.0f;
            if (row < rows && column < columns) {
                weight = normalized_weights[size_t(row) * columns + column];
                if (!std::isfinite(weight)) return false;
            }
            block.texels[size_t(y) * block_width + x].rgba = {weight, weight, weight, 1.0f};
        }
        blocks.push_back(std::move(block));
    }
    return true;
}

bool astc_gpu_d1_build_gauge_la_source_blocks(
    astc_vulkan_footprint footprint,
    const std::vector<float> & normalized_weights,
    const std::vector<float> & gauge_delta,
    uint32_t rows,
    uint32_t columns,
    std::vector<astc_gpu_encoder_source_block> & blocks) {
    blocks.clear();
    const auto format = astc_vulkan_format(footprint);
    const uint32_t block_width = format.block_width;
    const uint32_t block_height = format.block_height;
    if (block_width == 0 || block_height == 0 || rows == 0 || columns == 0 ||
        normalized_weights.size() != size_t(rows) * columns ||
        gauge_delta.size() != normalized_weights.size()) return false;
    const uint32_t blocks_x = (columns + block_width - 1) / block_width;
    const uint32_t blocks_y = (rows + block_height - 1) / block_height;
    blocks.reserve(size_t(blocks_x) * blocks_y);
    for (uint32_t by = 0; by < blocks_y; ++by) for (uint32_t bx = 0; bx < blocks_x; ++bx) {
        astc_gpu_encoder_source_block block;
        block.footprint = footprint;
        block.source_block_id = static_cast<uint32_t>(blocks.size());
        block.texels.resize(size_t(block_width) * block_height);
        for (uint32_t y = 0; y < block_height; ++y) for (uint32_t x = 0; x < block_width; ++x) {
            const uint32_t row = by * block_height + y;
            const uint32_t column = bx * block_width + x;
            float q = 0.0f;
            float delta = 0.0f;
            if (row < rows && column < columns) {
                const size_t index = size_t(row) * columns + column;
                q = normalized_weights[index];
                delta = gauge_delta[index];
                if (!std::isfinite(q) || !std::isfinite(delta)) return false;
            }
            const float luminance = q + delta;
            const float alpha = q - delta;
            block.texels[size_t(y) * block_width + x].rgba = {
                luminance, luminance, luminance, alpha};
        }
        blocks.push_back(std::move(block));
    }
    return true;
}
