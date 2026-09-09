#include "astc-vulkan-catalog-loader.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <vector>

#define REQUIRE(condition) do { if (!(condition)) return 1; } while (false)

namespace {

void write_bytes(const std::filesystem::path & path, const std::vector<uint8_t> & bytes) {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!file.good()) std::abort();
}

} // namespace

int main() {
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "astc-vulkan-catalog-loader-test";
    const fs::path model = root / "model.gguf";
    const fs::path manifest = root / "manifest.astcv.input";
    const fs::path payload = root / "payload.astcpack.input";
    std::error_code ignored;
    fs::remove_all(root, ignored);
    fs::create_directories(root);
    write_bytes(model, {0x47, 0x47, 0x55, 0x46, 1, 2, 3, 4});
    write_bytes(payload, std::vector<uint8_t>(16, 0x5a));

    astc_vulkan_manifest source;
    // Keep this as a legacy single-tensor manifest: the loader must support
    // the same sidecar format used by runtime-created caches.
    source.version = 3;
    source.model_fingerprint = "catalog-loader-fixture";
    astc_vulkan_tensor_record tensor;
    tensor.name = "blk.3.ffn_down.weight";
    tensor.width = 4;
    tensor.height = 4;
    tensor.footprint = astc_vulkan_footprint::k4x4;
    tensor.byte_size = 16;
    tensor.semantic_role = "ffn.down";
    tensor.canonical_path = "layers/3/ffn/down/weight";
    // Use the actual fixture bytes for the compact checksum.
    const std::vector<uint8_t> payload_bytes(16, 0x5a);
    tensor.payload_hash64 = astc_vulkan_payload_hash64(payload_bytes.data(), payload_bytes.size());
    source.tensors.push_back(tensor);
    std::string error;
    REQUIRE(astc_vulkan_write_manifest(manifest.string(), source, error));

    astc_vulkan_cache_paths paths;
    const fs::path cache_root = root / "cache";
    REQUIRE(astc_vulkan_cache_create(model.string(), manifest.string(), payload.string(), {}, {},
                                     cache_root.string(), paths, error));
    REQUIRE(fs::is_regular_file(paths.catalog));
    REQUIRE(fs::is_regular_file(paths.catalog_sha256));
    astc_vulkan_compiled_catalog catalog;
    REQUIRE(astc_vulkan_compiled_catalog_from_manifest(source, catalog, error));
    const fs::path catalog_path = fs::path(paths.root) / "catalog.astcc";
    REQUIRE(astc_vulkan_write_compiled_catalog(catalog_path.string(), catalog, error));

    astc_vulkan_catalog_binding binding;
    REQUIRE(astc_vulkan_load_catalog_for_cache(
        model.string(), paths.root, "auto", binding, error));
    REQUIRE(binding.catalog.tensors.size() == 1);
    REQUIRE(binding.catalog.tensors.front().logical_name == tensor.name);

    catalog.tensors.front().logical_name = "blk.99.ffn_down.weight";
    REQUIRE(astc_vulkan_write_compiled_catalog(catalog_path.string(), catalog, error));
    REQUIRE(!astc_vulkan_load_catalog_for_cache(
        model.string(), paths.root, "auto", binding, error));
    fs::remove_all(root, ignored);
    return 0;
}

#undef REQUIRE
