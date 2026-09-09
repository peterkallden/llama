#pragma once

#include "astc-vulkan-manifest.h"

#include <cstdint>
#include <string>
#include <vector>

// The first compiled-model step is intentionally an index, not a new ggml
// type. It gives a future bootstrap container stable logical tensor names and
// physical storage descriptors while the existing GGUF/sidecar runtime stays
// unchanged.
enum class astc_vulkan_compiled_storage_kind : uint8_t {
    native = 0,
    astc_d1 = 1,
    astc_d2 = 2,
};

struct astc_vulkan_compiled_tensor_record {
    // `logical_name` is the exact GGUF/llama tensor name and is the only
    // authoritative lookup key.
    std::string logical_name;
    std::string semantic_role;
    std::string canonical_path;

    // Human-readable physical contract. It is descriptive; runtime dispatch
    // still uses the typed fields and the existing ASTC manifest.
    std::string storage_class;
    std::string artifact_id;
    std::string native_type;
    astc_vulkan_compiled_storage_kind storage_kind =
        astc_vulkan_compiled_storage_kind::native;

    uint32_t width = 0;
    uint32_t height = 0;
    uint64_t payload_offset = 0;
    uint64_t payload_size = 0;
    uint64_t layout_offset = 0;
    uint64_t layout_size = 0;
    uint64_t row_scale_offset = 0;
    uint64_t row_scale_size = 0;
    uint64_t pair_map_offset = 0;
    uint64_t pair_map_size = 0;
};

struct astc_vulkan_compiled_catalog {
    uint32_t version = 1;
    std::string logical_model_id;
    std::string source_model_fingerprint;
    std::string tokenizer_fingerprint;
    std::vector<astc_vulkan_compiled_tensor_record> tensors;
};

bool astc_vulkan_compiled_catalog_from_manifest(
    const astc_vulkan_manifest & manifest,
    astc_vulkan_compiled_catalog & result,
    std::string & error);

bool astc_vulkan_validate_compiled_catalog(
    const astc_vulkan_compiled_catalog & catalog,
    std::string & error);

bool astc_vulkan_write_compiled_catalog(
    const std::string & path,
    const astc_vulkan_compiled_catalog & catalog,
    std::string & error);

bool astc_vulkan_read_compiled_catalog(
    const std::string & path,
    astc_vulkan_compiled_catalog & catalog,
    std::string & error);
