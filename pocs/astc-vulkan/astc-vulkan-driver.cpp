#include "astc-vulkan-driver.h"

#include <algorithm>
bool astc_vulkan_pack_atlas(const astc_vulkan_atlas_config & config,
                            const std::vector<astc_vulkan_tensor_record> & tensors,
                            std::vector<astc_vulkan_atlas_placement> & placements,
                            std::string & error) {
    if (config.max_width == 0 || config.max_height == 0) {
        error = "ASTC Vulkan atlas dimensions must be non-zero";
        return false;
    }
    placements.clear();
    struct cursor { uint32_t page = 0; uint32_t x = 0; uint32_t y = 0; uint32_t row_height = 0; };
    cursor cursors[3]{};
    for (const astc_vulkan_tensor_record & tensor : tensors) {
        if (tensor.name.empty() || tensor.width == 0 || tensor.height == 0 ||
            astc_vulkan_format(tensor.footprint).block_width == 0) {
            error = "invalid ASTC Vulkan atlas tensor";
            return false;
        }
        const astc_vulkan_format_info info = astc_vulkan_format(tensor.footprint);
        if (tensor.width > config.max_width || tensor.height > config.max_height) {
            error = "ASTC Vulkan tensor exceeds atlas dimensions";
            return false;
        }
        cursor & state = cursors[static_cast<size_t>(tensor.footprint)];
        const uint32_t aligned_width = ((tensor.width + info.block_width - 1) / info.block_width) * info.block_width;
        const uint32_t aligned_height = ((tensor.height + info.block_height - 1) / info.block_height) * info.block_height;
        if (aligned_width > config.max_width || aligned_height > config.max_height) {
            error = "ASTC Vulkan tensor block extent exceeds atlas dimensions";
            return false;
        }
        if (static_cast<uint64_t>(state.x) + aligned_width > config.max_width) {
            state.x = 0;
            state.y += state.row_height;
            state.row_height = 0;
        }
        if (state.y + tensor.height > config.max_height) {
            ++state.page;
            state.x = state.y = state.row_height = 0;
        }
        if (static_cast<uint64_t>(state.x) + aligned_width > config.max_width ||
            static_cast<uint64_t>(state.y) + aligned_height > config.max_height) {
            error = "ASTC Vulkan tensor cannot be placed in atlas";
            return false;
        }
        placements.push_back({tensor.name, tensor.footprint, state.page, state.x, state.y,
                              tensor.width, tensor.height});
        state.x += aligned_width;
        state.row_height = std::max(state.row_height, aligned_height);
    }
    error.clear();
    return true;
}
