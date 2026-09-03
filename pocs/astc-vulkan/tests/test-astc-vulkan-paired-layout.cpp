#include "astc-vulkan-driver.h"
#include "astc-vulkan-paired-layout.h"

#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

int main() {
    constexpr auto footprint = astc_vulkan_footprint::k8x5;
    constexpr uint32_t width = 16;
    constexpr uint32_t logical_height = 21;
    static_assert(sizeof(uint32_t) == 4, "layout words require uint32_t");
    assert(astc_vulkan_paired_storage_height(logical_height) == 11);
    assert(astc_vulkan_paired_block_count(footprint, width, logical_height) == 6);
    assert(astc_vulkan_paired_layout_word_count(footprint, width, logical_height) == 1);
    assert(astc_vulkan_paired_layout_bytes(footprint, width, logical_height) == 4);

    std::vector<uint32_t> words(1, 0);
    assert(astc_vulkan_paired_layout_set(words, 1, astc_vulkan_paired_layout::r_gb));
    assert(astc_vulkan_paired_layout_set(words, 5, astc_vulkan_paired_layout::r_gb));
    astc_vulkan_paired_layout layout = astc_vulkan_paired_layout::rg_b;
    assert(astc_vulkan_paired_layout_get(words, 0, layout) && layout == astc_vulkan_paired_layout::rg_b);
    assert(astc_vulkan_paired_layout_get(words, 1, layout) && layout == astc_vulkan_paired_layout::r_gb);
    assert(astc_vulkan_paired_layout_get(words, 5, layout) && layout == astc_vulkan_paired_layout::r_gb);
    assert(!astc_vulkan_paired_layout_get(words, 32, layout));

    astc_vulkan_tensor_record record;
    record.name = "blk.0.ffn_down.weight";
    record.width = width;
    record.height = logical_height;
    record.footprint = footprint;
    record.byte_size = astc_vulkan_image_bytes(footprint, width, 11);
    record.representation = astc_vulkan_representation::kPairedD2;
    record.layout_byte_size = sizeof(uint32_t);
    record.layout_hash64 = astc_vulkan_payload_hash64(
        reinterpret_cast<const uint8_t *>(words.data()), sizeof(uint32_t));
    std::vector<uint8_t> payload(static_cast<size_t>(record.byte_size), 0);
    std::string error;
    assert(astc_vulkan_validate_payload(record, payload.data(), payload.size(), error));
    assert(astc_vulkan_validate_layout_map(record,
        reinterpret_cast<const uint8_t *>(words.data()), sizeof(uint32_t), error));

    astc_vulkan_manifest manifest;
    manifest.tensors.push_back(record);
    assert(astc_vulkan_validate_manifest(manifest, error));
    assert(astc_vulkan_validate_layout_blob(manifest, sizeof(uint32_t), error));
    assert(!astc_vulkan_validate_layout_blob(manifest, sizeof(uint32_t) - 1, error));
    std::vector<astc_vulkan_atlas_placement> placements;
    assert(!astc_vulkan_pack_atlas({4096, 4096}, manifest.tensors, placements, error));
    assert(error.find("metadata-aware") != std::string::npos);
    const std::string path = "astc-vulkan-paired-layout.manifest";
    assert(astc_vulkan_write_manifest(path, manifest, error));
    astc_vulkan_manifest reread;
    assert(astc_vulkan_read_manifest(path, reread, error));
    assert(reread.version == 3 && reread.tensors[0].representation == record.representation);
    assert(reread.tensors[0].height == logical_height && reread.tensors[0].layout_hash64 == record.layout_hash64);
    std::remove(path.c_str());

    record.layout_byte_size = 0;
    manifest.tensors[0] = record;
    assert(!astc_vulkan_validate_manifest(manifest, error));
    std::puts("ASTC Vulkan paired layout metadata contract passed");
    return 0;
}
