#include "astc-vulkan-driver.h"
#include "astc-vulkan-paired-layout.h"
#include "astc-vulkan-tensor-catalog.h"

#include <cassert>
#include <array>
#include <cstdio>
#include <cmath>
#include <string>

int main() {
    static_assert(static_cast<uint8_t>(astc_vulkan_footprint::k8x8) == 4,
                  "serialized 8x8 footprint ID changed");
    static_assert(static_cast<uint8_t>(astc_vulkan_footprint::k10x6) == 5,
                  "new serialized footprint must be appended");
    static_assert(static_cast<uint8_t>(astc_vulkan_footprint::k10x8) == 6,
                  "10x8 serialized footprint ID changed");
    static_assert(static_cast<uint8_t>(astc_vulkan_footprint::k8x5) == 7,
                  "new serialized footprint must be appended");
    static_assert(static_cast<uint8_t>(astc_vulkan_footprint::k10x5) == 8,
                  "new serialized footprint must be appended");
    static_assert(static_cast<uint8_t>(astc_vulkan_footprint::k6x5) == 9,
                  "new serialized footprint must be appended");
    assert(!astc_vulkan_footprint_is_experimental(astc_vulkan_footprint::k6x6));
    assert(astc_vulkan_footprint_is_experimental(astc_vulkan_footprint::k8x6));
    assert(astc_vulkan_footprint_is_experimental(astc_vulkan_footprint::k10x6));
    assert(astc_vulkan_footprint_is_experimental(astc_vulkan_footprint::k8x8));
    assert(astc_vulkan_footprint_is_experimental(astc_vulkan_footprint::k10x8));
    assert(astc_vulkan_footprint_is_experimental(astc_vulkan_footprint::k8x5));
    assert(astc_vulkan_footprint_is_experimental(astc_vulkan_footprint::k10x5));
    assert(astc_vulkan_footprint_is_experimental(astc_vulkan_footprint::k6x5));
    assert(astc_vulkan_footprint_is_valid(astc_vulkan_footprint::k8x6));
    assert(astc_vulkan_footprint_is_valid(astc_vulkan_footprint::k10x6));
    assert(astc_vulkan_footprint_is_valid(astc_vulkan_footprint::k8x8));
    assert(astc_vulkan_footprint_is_valid(astc_vulkan_footprint::k10x8));
    assert(astc_vulkan_footprint_is_valid(astc_vulkan_footprint::k8x5));
    assert(astc_vulkan_footprint_is_valid(astc_vulkan_footprint::k10x5));
    assert(astc_vulkan_footprint_is_valid(astc_vulkan_footprint::k6x5));
    assert(astc_vulkan_image_bytes(astc_vulkan_footprint::k6x5, 6, 5) == 16);
    assert(astc_vulkan_block_count(astc_vulkan_footprint::k4x4, 1536, 32) == 384 * 8);
    assert(astc_vulkan_image_bytes(astc_vulkan_footprint::k6x6, 1536, 128) == 90112);
    assert(astc_vulkan_image_bytes(astc_vulkan_footprint::k10x8, 1536, 128) ==
           16u * 154u * 16u);

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
    assert(astc_vulkan_validate_payload_blob(actual, actual.tensors[1].byte_offset +
                                             actual.tensors[1].byte_size, error));
    assert(!astc_vulkan_validate_payload_blob(actual, actual.tensors[1].byte_offset,
                                              error));
    auto bad_payload = payload;
    bad_payload[0] ^= 1;
    assert(!astc_vulkan_validate_payload(actual.tensors[0], bad_payload.data(), bad_payload.size(), error));
    assert(astc_vulkan_find_tensor(actual, "blk.0.attn_q.weight") != nullptr);
    assert(astc_vulkan_find_tensor(actual, "missing") == nullptr);

    const auto down_cap = astc_vulkan_tensor_role_capability_for_name("blk.0.ffn_down.weight");
    assert(down_cap.matrix_candidate && down_cap.native_binding_ready && down_cap.source_builder_ready);
    for (const char * name : {
            "blk.0.ffn_up.weight", "blk.0.ffn_gate.weight",
            "blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight",
            "blk.0.attn_output.weight", "output.weight"}) {
        const auto capability = astc_vulkan_tensor_role_capability_for_name(name);
        assert(capability.matrix_candidate && capability.native_binding_ready);
        assert(!capability.source_builder_ready);
    }
    const auto norm_cap = astc_vulkan_tensor_role_capability_for_name("blk.0.attn_norm.weight");
    assert(!norm_cap.matrix_candidate && !norm_cap.native_binding_ready);

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
    astc_vulkan_manifest affine_v2 = expected;
    affine_v2.version = 2;
    const std::string affine_v2_path = "astc-vulkan-driver-test-v2.manifest";
    assert(astc_vulkan_write_manifest(affine_v2_path, affine_v2, error));
    astc_vulkan_manifest affine_v2_read;
    assert(astc_vulkan_read_manifest(affine_v2_path, affine_v2_read, error));
    assert(affine_v2_read.version == 2);
    assert(affine_v2_read.tensors[0].representation == astc_vulkan_representation::kGaugeLumaAlpha);
    assert(affine_v2_read.tensors[0].layout_byte_size == 0);
    std::remove(affine_v2_path.c_str());
    invalid = expected;
    invalid.tensors[0].scale_l = NAN;
    assert(!astc_vulkan_validate_manifest(invalid, error));
    invalid = expected;
    invalid.tensors[0].representation = static_cast<astc_vulkan_representation>(255);
    assert(!astc_vulkan_validate_manifest(invalid, error));
    invalid = expected;
    invalid.tensors[1].name = invalid.tensors[0].name;
    assert(!astc_vulkan_validate_manifest(invalid, error));

    // v4 stores multiple immutable, evidence-bearing artifacts for one tensor.
    // In particular, neutral and validation-selected D2-LA may legitimately
    // share a logical tensor name while owning distinct payload ranges.
    astc_vulkan_manifest v4;
    v4.version = 4;
    v4.model_fingerprint = "pythia-d2-la-test";
    const uint64_t d2_bytes = astc_vulkan_image_bytes(astc_vulkan_footprint::k8x5, 8, 5);
    const uint64_t d2_layout_bytes = astc_vulkan_paired_layout_bytes(
        astc_vulkan_footprint::k8x5, 8, 10);
    astc_vulkan_artifact_record neutral_artifact;
    neutral_artifact.id = "blk.0.ffn_down.weight/d2-la/neutral";
    neutral_artifact.storage = {"blk.0.ffn_down.weight", 8, 10,
        astc_vulkan_footprint::k8x5, 0, d2_bytes,
        astc_vulkan_representation::kPairedD2, 1.0f, 0.0f, 0.0f, 0,
        0, d2_layout_bytes, 0};
    neutral_artifact.paired_semantic = astc_vulkan_paired_semantic::luminance_alpha;
    neutral_artifact.encoder_profile = "d2-la-neutral";
    neutral_artifact.evidence = {true, true, 0.1f, 0.2f, 0.3f, 90.0f,
                                 "calibration-hash", "replay-hash"};
    astc_vulkan_artifact_record selected_artifact = neutral_artifact;
    selected_artifact.id = "blk.0.ffn_down.weight/d2-la/selected";
    selected_artifact.storage.byte_offset = d2_bytes;
    selected_artifact.variant = astc_vulkan_artifact_variant::validation_selected;
    selected_artifact.encoder_profile = "d2-la-selected";
    v4.artifacts = {neutral_artifact, selected_artifact};
    const std::string v4_path = "astc-vulkan-driver-test-v4.manifest";
    assert(astc_vulkan_validate_manifest(v4, error));
    assert(astc_vulkan_write_manifest(v4_path, v4, error));
    astc_vulkan_manifest v4_read;
    assert(astc_vulkan_read_manifest(v4_path, v4_read, error));
    assert(v4_read.version == 4 && v4_read.tensors.empty() && v4_read.artifacts.size() == 2);
    const auto * selected = astc_vulkan_find_artifact(v4_read, selected_artifact.id);
    assert(selected != nullptr);
    assert(selected->paired_semantic == astc_vulkan_paired_semantic::luminance_alpha);
    assert(selected->variant == astc_vulkan_artifact_variant::validation_selected);
    assert(selected->evidence.loss_delta == 0.3f);
    assert(!astc_vulkan_validate_payload_blob(v4_read, d2_bytes, error));
    assert(astc_vulkan_validate_payload_blob(v4_read, d2_bytes * 2, error));
    astc_vulkan_manifest duplicate_artifact = v4;
    duplicate_artifact.artifacts[1].id = duplicate_artifact.artifacts[0].id;
    assert(!astc_vulkan_validate_manifest(duplicate_artifact, error));
    std::remove(v4_path.c_str());
    std::vector<astc_vulkan_atlas_placement> placements;
    assert(astc_vulkan_pack_atlas({4096, 4096}, expected.tensors, placements, error));
    assert(placements.size() == expected.tensors.size());
    assert(placements[0].page == 0 && placements[0].x == 0 && placements[0].y == 0);
    assert(placements[1].page == 0 && placements[1].footprint == astc_vulkan_footprint::k4x4);
    assert(!astc_vulkan_pack_atlas({1024, 1024}, expected.tensors, placements, error));
    const std::vector<astc_vulkan_tensor_record> experimental = {
        {"experimental-8x6", 64, 48, astc_vulkan_footprint::k8x6, 0,
         astc_vulkan_image_bytes(astc_vulkan_footprint::k8x6, 64, 48)},
        {"experimental-10x6", 80, 48, astc_vulkan_footprint::k10x6, 0,
         astc_vulkan_image_bytes(astc_vulkan_footprint::k10x6, 80, 48)},
        {"experimental-8x8", 64, 64, astc_vulkan_footprint::k8x8, 0,
         astc_vulkan_image_bytes(astc_vulkan_footprint::k8x8, 64, 64)},
        {"experimental-10x8", 80, 64, astc_vulkan_footprint::k10x8, 0,
         astc_vulkan_image_bytes(astc_vulkan_footprint::k10x8, 80, 64)},
    };
    assert(astc_vulkan_pack_atlas({4096, 4096}, experimental, placements, error));
    assert(placements.size() == experimental.size());
    assert(placements[0].footprint == astc_vulkan_footprint::k8x6);
    assert(placements[1].footprint == astc_vulkan_footprint::k10x6);
    assert(placements[2].footprint == astc_vulkan_footprint::k8x8);
    assert(placements[3].footprint == astc_vulkan_footprint::k10x8);
    std::vector<astc_vulkan_tensor_record> edge = {
        {"edge", 4093, 1, astc_vulkan_footprint::k6x6, 0,
         astc_vulkan_image_bytes(astc_vulkan_footprint::k6x6, 4093, 1)},
    };
    assert(!astc_vulkan_pack_atlas({4096, 4096}, edge, placements, error));
    std::remove(path.c_str());
    std::puts("ASTC Vulkan driver metadata contract passed");
    return 0;
}
