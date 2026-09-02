#include "astc-vulkan-driver.h"

#include <cassert>
#include <array>
#include <cstdio>
#include <cmath>
#include <string>

int main() {
    assert(astc_vulkan_block_count(astc_vulkan_footprint::k4x4, 1536, 32) == 384 * 8);
    assert(astc_vulkan_image_bytes(astc_vulkan_footprint::k6x6, 1536, 128) == 90112);

    astc_vulkan_manifest expected;
    expected.model_fingerprint = "smollm2-test";
    const std::array<uint8_t, 4> payload = {1, 2, 3, 4};
    expected.tensors = {
        {"blk.0.ffn_down.weight", 1536, 32, astc_vulkan_footprint::k6x6, 0,
         astc_vulkan_image_bytes(astc_vulkan_footprint::k6x6, 1536, 32),
         astc_vulkan_representation::kGaugeLumaAlpha, 1.0f, 0.25f, -0.5f,
         astc_vulkan_payload_hash64(payload.data(), payload.size())},
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
    assert(actual.tensors[0].representation == astc_vulkan_representation::kGaugeLumaAlpha);
    assert(actual.tensors[0].scale_a == 0.25f);
    assert(actual.tensors[0].payload_hash64 == expected.tensors[0].payload_hash64);
    assert(astc_vulkan_validate_payload(actual.tensors[0], payload.data(), payload.size(), error));
    auto bad_payload = payload;
    bad_payload[0] ^= 1;
    assert(!astc_vulkan_validate_payload(actual.tensors[0], bad_payload.data(), bad_payload.size(), error));
    assert(astc_vulkan_find_tensor(actual, "blk.0.attn_q.weight") != nullptr);
    assert(astc_vulkan_find_tensor(actual, "missing") == nullptr);

    astc_vulkan_manifest invalid = expected;
    invalid.tensors[1].byte_offset = 1;
    assert(!astc_vulkan_validate_manifest(invalid, error));
    astc_vulkan_manifest legacy = expected;
    legacy.version = 1;
    const std::string legacy_path = "astc-vulkan-driver-test-v1.manifest";
    assert(astc_vulkan_write_manifest(legacy_path, legacy, error));
    astc_vulkan_manifest legacy_read;
    assert(astc_vulkan_read_manifest(legacy_path, legacy_read, error));
    assert(legacy_read.version == 1);
    assert(legacy_read.tensors[0].representation == astc_vulkan_representation::kScalar);
    assert(legacy_read.tensors[0].scale_l == 1.0f);
    assert(legacy_read.tensors[0].scale_a == 0.0f);
    assert(legacy_read.tensors[0].offset == 0.0f);
    assert(legacy_read.tensors[0].payload_hash64 == 0);
    std::remove(legacy_path.c_str());
    invalid = expected;
    invalid.tensors[0].scale_l = NAN;
    assert(!astc_vulkan_validate_manifest(invalid, error));
    invalid = expected;
    invalid.tensors[0].representation = static_cast<astc_vulkan_representation>(255);
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
