#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "astc-vulkan-format.h"

// The representation is an offline encoding choice. Runtime code must treat
// the payload as ordinary standard ASTC and use the recorded affine decode.
enum class astc_vulkan_representation : uint8_t {
    kScalar = 0,
    kGaugeLumaAlpha = 1,
    kCDelta = 2,
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
};

struct astc_vulkan_manifest {
    uint32_t version = 2;
    std::string model_fingerprint;
    std::vector<astc_vulkan_tensor_record> tensors;
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

// FNV-1a is deliberately small and dependency-free. It detects stale or
// mismatched sidecar payloads; it is not intended as a cryptographic hash.
uint64_t astc_vulkan_payload_hash64(const uint8_t * data, size_t size);

bool astc_vulkan_validate_payload(const astc_vulkan_tensor_record & tensor,
                                  const uint8_t * data, size_t size,
                                  std::string & error);
bool astc_vulkan_validate_payload_blob(const astc_vulkan_manifest & manifest,
                                       uint64_t blob_size, std::string & error);

bool astc_vulkan_validate_manifest(const astc_vulkan_manifest & manifest,
                                   std::string & error);
bool astc_vulkan_write_manifest(const std::string & path,
                                const astc_vulkan_manifest & manifest,
                                std::string & error);
bool astc_vulkan_read_manifest(const std::string & path,
                               astc_vulkan_manifest & manifest,
                               std::string & error);
