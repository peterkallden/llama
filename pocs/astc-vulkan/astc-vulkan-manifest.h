#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "astc-vulkan-artifact.h"
#include "astc-vulkan-format.h"

// The representation is an offline encoding choice. Runtime code must treat
// the payload as ordinary standard ASTC and use the recorded affine decode.
enum class astc_vulkan_representation : uint8_t {
    kScalar = 0,
    kGaugeLumaAlpha = 1,
    kCDelta = 2,
    // Two adjacent logical output rows per ASTC texel. A separate packed
    // layout map selects RG/B or R/GB per physical ASTC block.
    kPairedD2 = 3,
};

struct astc_vulkan_tensor_record {
    std::string name;
    uint32_t width = 0;
    uint32_t height = 0;
    astc_vulkan_footprint footprint = astc_vulkan_footprint::k6x6;
    uint64_t byte_offset = 0;
    uint64_t byte_size = 0;

    astc_vulkan_representation representation = astc_vulkan_representation::kScalar;
    float scale_l = 1.0f;
    float scale_a = 0.0f;
    float offset = 0.0f;
    // Zero means that the producer did not provide a payload checksum.
    uint64_t payload_hash64 = 0;

    // v3 paired-D2 metadata. `height` always means logical tensor height;
    // paired-D2 stores ceil(height / 2) texture rows. The first concrete
    // profiles are D2_6x5 (2.133 b/w), D2_8x5 (1.60 b/w), and D2_10x5
    // (1.28 b/w). The layout map
    // is a separate packed uint32 blob, addressed independently from ASTC
    // bytes.
    uint64_t layout_byte_offset = 0;
    uint64_t layout_byte_size = 0;
    uint64_t layout_hash64 = 0;
};

// v4 multi-artifact cache entry. Storage ranges remain independently
// verifiable so a single cache can contain neutral/selected, normalized and
// semantic variants for the same source tensor. `storage` keeps the existing
// per-payload contract intact; the artifact fields explain how it was made and
// which runtime semantic must be used.
struct astc_vulkan_artifact_record {
    std::string id;
    astc_vulkan_tensor_record storage;
    astc_vulkan_artifact_variant variant = astc_vulkan_artifact_variant::neutral;
    astc_vulkan_normalization normalization = astc_vulkan_normalization::none;
    astc_vulkan_paired_semantic paired_semantic = astc_vulkan_paired_semantic::direct_rgb;
    std::string encoder_profile;
    astc_vulkan_artifact_evidence evidence{};
    uint64_t row_scale_byte_offset = 0;
    uint64_t row_scale_byte_size = 0;
    uint64_t row_scale_hash64 = 0;
    // v5 D2 row-pair permutation. Each 10-logical-row group occupies ten
    // bytes: physical pair/member slot -> original local logical row.  A
    // direct map is deliberately used instead of a 10-bit matching index so
    // its meaning never depends on an enumerator order.
    uint64_t pair_map_byte_offset = 0;
    uint64_t pair_map_byte_size = 0;
    uint64_t pair_map_hash64 = 0;
};

struct astc_vulkan_manifest {
    uint32_t version = 3;
    std::string model_fingerprint;
    // v1-v3 legacy single-artifact tensor table. v4 writers use artifacts;
    // keeping this table preserves read compatibility and low-level tools.
    std::vector<astc_vulkan_tensor_record> tensors;
    std::vector<astc_vulkan_artifact_record> artifacts;
};

// Plain data shared by manifest loading, CPU reference code, and the shader
// push-constant contract. Keep the field order stable and 16-byte aligned.
struct astc_vulkan_reconstruction {
    float scale_l = 1.0f;
    float scale_a = 0.0f;
    float offset = 0.0f;
    float reserved = 0.0f;
};

const astc_vulkan_tensor_record * astc_vulkan_find_tensor(
    const astc_vulkan_manifest & manifest, const std::string & name);
const astc_vulkan_artifact_record * astc_vulkan_find_artifact(
    const astc_vulkan_manifest & manifest, const std::string & id);

// FNV-1a is deliberately small and dependency-free. It detects stale or
// mismatched sidecar payloads; it is not intended as a cryptographic hash.
uint64_t astc_vulkan_payload_hash64(const uint8_t * data, size_t size);

bool astc_vulkan_validate_payload(const astc_vulkan_tensor_record & tensor,
                                  const uint8_t * data, size_t size,
                                  std::string & error);
bool astc_vulkan_validate_payload_blob(const astc_vulkan_manifest & manifest,
                                       uint64_t blob_size, std::string & error);
bool astc_vulkan_validate_layout_blob(const astc_vulkan_manifest & manifest,
                                      uint64_t blob_size, std::string & error);
bool astc_vulkan_validate_layout_map(const astc_vulkan_tensor_record & tensor,
                                     const uint8_t * data, size_t size,
                                     std::string & error);
bool astc_vulkan_validate_pair_map(const astc_vulkan_artifact_record & artifact,
                                   const uint8_t * data, size_t size,
                                   std::string & error);

bool astc_vulkan_validate_manifest(const astc_vulkan_manifest & manifest,
                                   std::string & error);
bool astc_vulkan_write_manifest(const std::string & path,
                                const astc_vulkan_manifest & manifest,
                                std::string & error);
bool astc_vulkan_read_manifest(const std::string & path,
                               astc_vulkan_manifest & manifest,
                               std::string & error);
