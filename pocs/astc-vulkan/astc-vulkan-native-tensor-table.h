#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Logical storage map for an ASTCCM v2 container.  This is deliberately
// separate from the sidecar catalog: the latter remains a manifest index for
// the ASTC overlay, while this table accounts for *every* GGUF tensor and
// identifies the native residual bytes that make the container standalone.
enum class astc_vulkan_native_tensor_storage : uint8_t {
    native = 0,
    astc = 1,
};

struct astc_vulkan_native_tensor_record {
    std::string name;
    astc_vulkan_native_tensor_storage storage = astc_vulkan_native_tensor_storage::native;
    uint32_t ggml_type = 0;
    uint32_t n_dims = 0;
    uint64_t ne[4] = {1, 1, 1, 1};
    uint64_t native_offset = 0;
    uint64_t native_size = 0;
    uint64_t native_hash64 = 0;
};

struct astc_vulkan_native_tensor_table {
    static constexpr uint32_t kVersion = 1;
    uint32_t version = kVersion;
    std::vector<astc_vulkan_native_tensor_record> tensors;
};

bool astc_vulkan_validate_native_tensor_table(
    const astc_vulkan_native_tensor_table & table, std::string & error);
bool astc_vulkan_encode_native_tensor_table(
    const astc_vulkan_native_tensor_table & table, std::vector<uint8_t> & bytes,
    std::string & error);
bool astc_vulkan_decode_native_tensor_table(
    const std::vector<uint8_t> & bytes, astc_vulkan_native_tensor_table & table,
    std::string & error);
