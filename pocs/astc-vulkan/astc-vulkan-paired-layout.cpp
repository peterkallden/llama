#include "astc-vulkan-paired-layout.h"

#include <limits>

uint32_t astc_vulkan_paired_storage_height(uint32_t logical_height) {
    return logical_height / 2 + logical_height % 2;
}

uint64_t astc_vulkan_paired_block_count(astc_vulkan_footprint footprint,
                                         uint32_t logical_width,
                                         uint32_t logical_height) {
    return astc_vulkan_block_count(footprint, logical_width,
                                   astc_vulkan_paired_storage_height(logical_height));
}

uint64_t astc_vulkan_paired_layout_word_count(astc_vulkan_footprint footprint,
                                               uint32_t logical_width,
                                               uint32_t logical_height) {
    const uint64_t blocks = astc_vulkan_paired_block_count(footprint, logical_width, logical_height);
    return blocks == 0 ? 0 : (blocks + 31) / 32;
}

uint64_t astc_vulkan_paired_layout_bytes(astc_vulkan_footprint footprint,
                                         uint32_t logical_width,
                                         uint32_t logical_height) {
    const uint64_t words = astc_vulkan_paired_layout_word_count(footprint, logical_width, logical_height);
    return words > std::numeric_limits<uint64_t>::max() / sizeof(uint32_t) ? 0 :
           words * sizeof(uint32_t);
}

bool astc_vulkan_paired_layout_get(const std::vector<uint32_t> & words,
                                   uint64_t block_index,
                                   astc_vulkan_paired_layout & layout) {
    const uint64_t word = block_index / 32;
    if (word >= words.size()) return false;
    layout = ((words[static_cast<size_t>(word)] >> (block_index % 32)) & 1u) == 0 ?
        astc_vulkan_paired_layout::rg_b : astc_vulkan_paired_layout::r_gb;
    return true;
}

bool astc_vulkan_paired_layout_set(std::vector<uint32_t> & words,
                                   uint64_t block_index,
                                   astc_vulkan_paired_layout layout) {
    const uint64_t word = block_index / 32;
    if (word >= words.size()) return false;
    const uint32_t mask = 1u << (block_index % 32);
    uint32_t & value = words[static_cast<size_t>(word)];
    if (layout == astc_vulkan_paired_layout::rg_b) value &= ~mask;
    else if (layout == astc_vulkan_paired_layout::r_gb) value |= mask;
    else return false;
    return true;
}
