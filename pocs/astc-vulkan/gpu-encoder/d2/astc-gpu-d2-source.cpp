#include "astc-gpu-d2-source.h"

#include <cmath>
#include <cstring>

namespace {

bool valid_unorm(float value) {
    return std::isfinite(value) && value >= 0.0f && value <= 1.0f;
}

bool geometry(astc_vulkan_footprint footprint, uint32_t logical_rows,
              uint32_t logical_columns, uint32_t & blocks_x,
              uint32_t & blocks_y, uint32_t & texture_rows) {
    const auto format = astc_vulkan_format(footprint);
    if (format.block_width == 0 || format.block_height == 0 ||
        logical_rows == 0 || logical_columns == 0) return false;
    texture_rows = (logical_rows + 1u) / 2u;
    blocks_x = (logical_columns + format.block_width - 1u) / format.block_width;
    blocks_y = (texture_rows + format.block_height - 1u) / format.block_height;
    return true;
}

} // namespace

bool astc_gpu_d2_expand_pair_map(
        astc_vulkan_footprint footprint, uint32_t logical_rows,
        uint32_t logical_columns, const std::vector<uint8_t> & pair_map,
        std::vector<astc_vulkan_d2_pairing> & pairings) {
    pairings.clear();
    uint32_t blocks_x = 0, blocks_y = 0, texture_rows = 0;
    const auto format = astc_vulkan_format(footprint);
    if (!geometry(footprint, logical_rows, logical_columns, blocks_x, blocks_y, texture_rows) ||
        format.block_height * 2u != astc_vulkan_d2_pair_group_rows) return false;
    if (pair_map.empty()) return true;
    const size_t group_count = (logical_rows + astc_vulkan_d2_pair_group_rows - 1u) /
        astc_vulkan_d2_pair_group_rows;
    if (pair_map.size() != group_count * astc_vulkan_d2_pair_group_rows) return false;
    std::vector<astc_vulkan_d2_pairing> by_group(group_count);
    for (size_t group = 0; group < group_count; ++group) {
        std::memcpy(by_group[group].row_order.data(),
                    pair_map.data() + group * astc_vulkan_d2_pair_group_rows,
                    astc_vulkan_d2_pair_group_rows);
        if (!astc_vulkan_d2_pairing_is_valid(by_group[group])) return false;
    }
    pairings.reserve(size_t(blocks_x) * blocks_y);
    for (uint32_t by = 0; by < blocks_y; ++by) {
        const size_t group = std::min<size_t>(by, by_group.size() - 1u);
        for (uint32_t bx = 0; bx < blocks_x; ++bx) pairings.push_back(by_group[group]);
    }
    return true;
}

bool astc_gpu_d2_make_uniform_layout_map(
    astc_vulkan_footprint footprint, uint32_t logical_rows,
    uint32_t logical_columns, astc_vulkan_paired_layout layout,
    std::vector<astc_vulkan_paired_layout> & layouts) {
    uint32_t blocks_x = 0, blocks_y = 0, texture_rows = 0;
    if (!geometry(footprint, logical_rows, logical_columns, blocks_x, blocks_y, texture_rows)) return false;
    layouts.assign(size_t(blocks_x) * blocks_y, layout);
    return true;
}

bool astc_gpu_d2_build_paired_source_blocks(
    astc_vulkan_footprint footprint,
    const std::vector<float> & normalized_weights,
    uint32_t logical_rows, uint32_t logical_columns,
    const std::vector<astc_vulkan_paired_layout> & layouts,
    const std::vector<float> & steering_texels,
    astc_vulkan_paired_semantic semantic,
    std::vector<astc_gpu_encoder_source_block> & blocks,
    const std::vector<astc_vulkan_d2_pairing> & pairings,
    astc_vulkan_d2_givens_transform transform) {
    blocks.clear();
    uint32_t blocks_x = 0, blocks_y = 0, texture_rows = 0;
    const auto format = astc_vulkan_format(footprint);
    if (!geometry(footprint, logical_rows, logical_columns, blocks_x, blocks_y, texture_rows) ||
        normalized_weights.size() != size_t(logical_rows) * logical_columns ||
        layouts.size() != size_t(blocks_x) * blocks_y ||
        (!pairings.empty() && pairings.size() != size_t(blocks_x) * blocks_y) ||
        (!steering_texels.empty() && steering_texels.size() != size_t(texture_rows) * logical_columns)) return false;
    if (!pairings.empty() && format.block_height * 2u != astc_vulkan_d2_pair_group_rows) return false;
    for (const float weight : normalized_weights) if (!valid_unorm(weight)) return false;
    for (const float steering : steering_texels) if (!valid_unorm(steering)) return false;
    if (!std::isfinite(transform.radians)) return false;
    blocks.reserve(size_t(blocks_x) * blocks_y);
    for (uint32_t by = 0; by < blocks_y; ++by) for (uint32_t bx = 0; bx < blocks_x; ++bx) {
        astc_gpu_encoder_source_block block;
        block.footprint = footprint;
        block.source_block_id = static_cast<uint32_t>(blocks.size());
        block.texels.resize(size_t(format.block_width) * format.block_height);
        const auto layout = layouts[size_t(by) * blocks_x + bx];
        const astc_vulkan_d2_pairing * pairing = nullptr;
        if (!pairings.empty()) {
            pairing = &pairings[size_t(by) * blocks_x + bx];
            if (!astc_vulkan_d2_pairing_is_valid(*pairing)) return false;
        }
        for (uint32_t y = 0; y < format.block_height; ++y) for (uint32_t x = 0; x < format.block_width; ++x) {
            const uint32_t texture_row = by * format.block_height + y;
            const uint32_t column = bx * format.block_width + x;
            float q0 = 0.5f, q1 = 0.5f, steering = 0.5f;
            if (texture_row < texture_rows && column < logical_columns) {
                const uint32_t group_row = by * format.block_height * 2u;
                const uint32_t row0 = pairing == nullptr ? texture_row * 2u :
                    group_row + pairing->row_order[2u * y];
                const uint32_t row1 = pairing == nullptr ? row0 + 1u :
                    group_row + pairing->row_order[2u * y + 1u];
                if (row0 < logical_rows) q0 = normalized_weights[size_t(row0) * logical_columns + column];
                if (row1 < logical_rows) q1 = normalized_weights[size_t(row1) * logical_columns + column];
                if (!steering_texels.empty()) steering = steering_texels[size_t(texture_row) * logical_columns + column];
            }
            if (std::fabs(transform.radians) > 1e-7f) {
                float u = 0.0f, v = 0.0f;
                astc_vulkan_d2_pair_forward(q0 - 0.5f, q1 - 0.5f, transform, u, v);
                constexpr float inverse_sqrt2 = 0.7071067811865475244f;
                q0 = 0.5f + inverse_sqrt2 * u;
                q1 = 0.5f + inverse_sqrt2 * v;
            }
            const auto texel = astc_vulkan_make_paired_texel(q0, q1, steering, layout,
                astc_vulkan_paired_basis::direct, semantic);
            block.texels[size_t(y) * format.block_width + x].rgba = {texel.r, texel.g, texel.b, texel.a};
        }
        blocks.push_back(std::move(block));
    }
    return true;
}
