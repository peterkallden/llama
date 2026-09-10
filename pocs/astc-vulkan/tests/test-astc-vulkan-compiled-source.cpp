#include "astc-vulkan-cache.h"
#include "astc-vulkan-compiled-model.h"
#include "astc-vulkan-compiled-source.h"
#include "astc-vulkan-hash.h"
#include "astc-vulkan-manifest.h"
#include "astc-vulkan-model-cache.h"
#include "astc-vulkan-provenance.h"

#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>

namespace {

std::vector<uint8_t> read_bytes(const std::filesystem::path & path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return {};
    file.seekg(0, std::ios::end);
    const auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> result(static_cast<size_t>(size));
    if (!result.empty()) file.read(reinterpret_cast<char *>(result.data()), size);
    return file.good() ? result : std::vector<uint8_t>{};
}

} // namespace

int main() {
    astc_vulkan_manifest manifest;
    manifest.version = 3;
    manifest.model_fingerprint = "compiled-source-test";
    astc_vulkan_tensor_record tensor;
    tensor.name = "blk.0.ffn_down.weight";
    tensor.width = 4;
    tensor.height = 4;
    tensor.footprint = astc_vulkan_footprint::k4x4;
    tensor.byte_size = 16;
    tensor.payload_hash64 = astc_vulkan_payload_hash64(
        reinterpret_cast<const uint8_t *>("0123456789abcdef"), 16);
    manifest.tensors.push_back(tensor);

    const auto nonce = static_cast<unsigned long long>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    const auto root = std::filesystem::temp_directory_path();
    const auto manifest_path = root / ("astc-vulkan-compiled-source-" + std::to_string(nonce) + ".astcv");
    const auto container_path = root / ("astc-vulkan-compiled-source-" + std::to_string(nonce) + ".astccm");
    std::string error;
    assert(astc_vulkan_write_manifest(manifest_path.string(), manifest, error));

    astc_vulkan_compiled_model model;
    model.gguf = {0x47, 0x47, 0x55, 0x46, 0x00, 0x01};
    model.source_model_fingerprint = astc_vulkan_sha256_hex(model.gguf.data(), model.gguf.size());
    model.manifest = read_bytes(manifest_path);
    model.payload.assign({'0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f'});
    assert(astc_vulkan_compiled_model_write(container_path.string(), model, error));

    std::string materialized_model_path;
    std::shared_ptr<const astc_vulkan_cache_source> retained_source;
    bool used_memory_file = false;
    {
        astc_vulkan_compiled_source source;
        assert(source.open(container_path.string(), error));
        assert(source.ready());
        materialized_model_path = source.model_path();
        used_memory_file = source.materialization_mode() ==
            astc_vulkan_compiled_source_mode::memory_file;
        assert(read_bytes(source.model_path()) == model.gguf);

        astc_vulkan_cache_validation memory_validation;
        assert(astc_vulkan_cache_validate_source(
            source.model_path(), source.cache_source(), source.source_fingerprint(),
            memory_validation, error));
        assert(memory_validation.source);
        assert(memory_validation.manifest.tensors.size() == 1);
        astc_vulkan_model_cache_catalog catalog;
        astc_vulkan_model_cache_plan_options options;
        assert(astc_vulkan_model_cache_load_catalog_from_source(
            source.model_path(), source.cache_source(), source.source_fingerprint(),
            options, catalog, error));
        assert(catalog.plan.entries.size() == 1);
        retained_source = source.cache_source();

    }
    astc_vulkan_cache_blob retained_payload;
    assert(retained_source && retained_source->blob(
        astc_vulkan_cache_blob_kind::payload, retained_payload, error));
    assert(retained_payload.size == model.payload.size());
    assert(retained_payload.data[0] == model.payload[0]);

#if defined(__linux__)
    // The descriptor is owned by the adapter and must be closed when its
    // lifetime ends; the private cache workspace must be removed as well.
    if (used_memory_file) {
        assert(materialized_model_path.rfind("/proc/self/fd/", 0) == 0);
        assert(!std::filesystem::exists(materialized_model_path));
    }
#endif
    std::error_code ignored;
    std::filesystem::remove(manifest_path, ignored);
    std::filesystem::remove(container_path, ignored);
    return 0;
}
