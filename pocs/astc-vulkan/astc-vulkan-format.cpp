#include "astc-vulkan-format.h"

#include <limits>

astc_vulkan_format_info astc_vulkan_format(astc_vulkan_footprint footprint) {
    switch (footprint) {
        case astc_vulkan_footprint::k4x4: return {4, 4, 16};
        case astc_vulkan_footprint::k5x5: return {5, 5, 16};
        case astc_vulkan_footprint::k6x6: return {6, 6, 16};
        case astc_vulkan_footprint::k8x6: return {8, 6, 16};
        case astc_vulkan_footprint::k8x8: return {8, 8, 16};
    }
    return {0, 0, 0};
}

bool astc_vulkan_footprint_is_valid(astc_vulkan_footprint footprint) {
    return astc_vulkan_format(footprint).block_width != 0;
}

bool astc_vulkan_footprint_is_experimental(astc_vulkan_footprint footprint) {
    return footprint == astc_vulkan_footprint::k8x6 ||
           footprint == astc_vulkan_footprint::k8x8;
}

uint64_t astc_vulkan_block_count(astc_vulkan_footprint footprint,
                                 uint32_t width, uint32_t height) {
    const astc_vulkan_format_info info = astc_vulkan_format(footprint);
    if (info.block_width == 0 || width == 0 || height == 0) return 0;
    const uint64_t blocks_x = (static_cast<uint64_t>(width) + info.block_width - 1) /
                              info.block_width;
    const uint64_t blocks_y = (static_cast<uint64_t>(height) + info.block_height - 1) /
                              info.block_height;
    return blocks_x * blocks_y;
}

uint64_t astc_vulkan_image_bytes(astc_vulkan_footprint footprint,
                                 uint32_t width, uint32_t height) {
    const uint64_t blocks = astc_vulkan_block_count(footprint, width, height);
    const astc_vulkan_format_info info = astc_vulkan_format(footprint);
    if (info.block_bytes == 0 || blocks > std::numeric_limits<uint64_t>::max() / info.block_bytes) {
        return 0;
    }
    return blocks * info.block_bytes;
}
