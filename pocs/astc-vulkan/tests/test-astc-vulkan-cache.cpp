#include "astc-vulkan-cache.h"
#include "astc-vulkan-paired-layout.h"

#include <array>
#include <cassert>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

void write_bytes(const std::string & path, const std::vector<uint8_t> & bytes) {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    assert(file.good());
}

} // namespace

int main() {
    namespace fs = std::filesystem;
    const fs::path root("astc-vulkan-cache-test-dir");
    const fs::path model("astc-vulkan-cache-test.gguf");
    const fs::path manifest_path("astc-vulkan-cache-test.manifest");
    const fs::path payload_path("astc-vulkan-cache-test.payload");
    const fs::path layout_path("astc-vulkan-cache-test.layout");
    std::error_code ignored;
    fs::remove_all(root, ignored);
    fs::remove(model, ignored);
    fs::remove(manifest_path, ignored);
    fs::remove(payload_path, ignored);
    fs::remove(layout_path, ignored);

    write_bytes(model.string(), {0x47, 0x47, 0x55, 0x46, 1, 2, 3, 4});
    const std::vector<uint8_t> payload(16, 0x5a);
    const std::vector<uint8_t> layout(4, 0);
    write_bytes(payload_path.string(), payload);
    write_bytes(layout_path.string(), layout);

    astc_vulkan_manifest manifest;
    manifest.model_fingerprint = "fixture";
    astc_vulkan_tensor_record record;
    record.name = "blk.0.ffn_down.weight";
    record.width = 8;
    record.height = 10;
    record.footprint = astc_vulkan_footprint::k8x5;
    record.byte_size = payload.size();
    record.representation = astc_vulkan_representation::kPairedD2;
    record.payload_hash64 = astc_vulkan_payload_hash64(payload.data(), payload.size());
    record.layout_byte_size = astc_vulkan_paired_layout_bytes(record.footprint, record.width, record.height);
    record.layout_hash64 = astc_vulkan_payload_hash64(layout.data(), layout.size());
    assert(record.layout_byte_size == layout.size());
    manifest.tensors.push_back(record);
    std::string error;
    assert(astc_vulkan_write_manifest(manifest_path.string(), manifest, error));

    astc_vulkan_cache_paths paths;
    assert(astc_vulkan_cache_create(model.string(), manifest_path.string(), payload_path.string(),
                                    layout_path.string(), {}, root.string(), paths, error));
    astc_vulkan_cache_validation validation;
    assert(astc_vulkan_cache_validate(model.string(), root.string(), validation, error));
    assert(validation.has_paired_d2 && validation.manifest.tensors.size() == 1);
    assert(astc_vulkan_cache_validate(model.string(), paths.manifest, validation, error));

    write_bytes(model.string(), {0x47, 0x47, 0x55, 0x46, 9, 9, 9, 9});
    assert(!astc_vulkan_cache_validate(model.string(), root.string(), validation, error));

    fs::remove_all(root, ignored);
    fs::remove(model, ignored);
    fs::remove(manifest_path, ignored);
    fs::remove(payload_path, ignored);
    fs::remove(layout_path, ignored);
    std::puts("ASTC Vulkan cache D1/D2 contract passed");
    return 0;
}
