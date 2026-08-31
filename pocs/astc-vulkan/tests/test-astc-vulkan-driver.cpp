#include "astc-vulkan-driver.h"

#include <cassert>
#include <cstdio>
#include <string>

int main() {
    assert(astc_vulkan_block_count(astc_vulkan_footprint::k4x4, 1536, 32) == 384 * 8);
    assert(astc_vulkan_image_bytes(astc_vulkan_footprint::k6x6, 1536, 128) == 90112);

    astc_vulkan_manifest expected;
    expected.model_fingerprint = "smollm2-test";
    expected.tensors = {
        {"blk.0.ffn_down.weight", 1536, 32, astc_vulkan_footprint::k6x6, 0,
         astc_vulkan_image_bytes(astc_vulkan_footprint::k6x6, 1536, 32)},
        {"blk.0.attn_q.weight", 576, 32, astc_vulkan_footprint::k4x4,
         astc_vulkan_image_bytes(astc_vulkan_footprint::k6x6, 1536, 32),
         astc_vulkan_image_bytes(astc_vulkan_footprint::k4x4, 576, 32)},
    };
    std::string error;
    const std::string path = "astc-vulkan-driver-test.manifest";
    assert(astc_vulkan_write_manifest(path, expected, error));
    astc_vulkan_manifest actual;
    assert(astc_vulkan_read_manifest(path, actual, error));
    assert(actual.version == expected.version);
    assert(actual.model_fingerprint == expected.model_fingerprint);
    assert(actual.tensors.size() == expected.tensors.size());
    assert(actual.tensors[0].name == expected.tensors[0].name);
    assert(actual.tensors[1].byte_offset == expected.tensors[1].byte_offset);

    astc_vulkan_manifest invalid = expected;
    invalid.tensors[1].byte_offset = 1;
    assert(!astc_vulkan_validate_manifest(invalid, error));
    std::vector<astc_vulkan_atlas_placement> placements;
    assert(astc_vulkan_pack_atlas({4096, 4096}, expected.tensors, placements, error));
    assert(placements.size() == expected.tensors.size());
    assert(placements[0].page == 0 && placements[0].x == 0 && placements[0].y == 0);
    assert(placements[1].page == 0 && placements[1].footprint == astc_vulkan_footprint::k4x4);
    assert(!astc_vulkan_pack_atlas({1024, 1024}, expected.tensors, placements, error));
    std::vector<astc_vulkan_tensor_record> edge = {
        {"edge", 4093, 1, astc_vulkan_footprint::k6x6, 0,
         astc_vulkan_image_bytes(astc_vulkan_footprint::k6x6, 4093, 1)},
    };
    assert(!astc_vulkan_pack_atlas({4096, 4096}, edge, placements, error));
    std::remove(path.c_str());
    std::puts("ASTC Vulkan driver metadata contract passed");
    return 0;
}
