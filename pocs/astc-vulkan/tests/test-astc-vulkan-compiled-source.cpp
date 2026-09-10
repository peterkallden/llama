#include "astc-vulkan-cache.h"
#include "astc-vulkan-compiled-model.h"
#include "astc-vulkan-compiled-source.h"
#include "astc-vulkan-hash.h"
#include "astc-vulkan-manifest.h"
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

    {
        astc_vulkan_compiled_source source;
        assert(source.open(container_path.string(), error));
        assert(source.ready());
        assert(read_bytes(source.model_path()) == model.gguf);

        astc_vulkan_cache_validation validation;
        assert(astc_vulkan_cache_validate(source.model_path(), source.cache_path(), validation, error));
        assert(validation.manifest.tensors.size() == 1);
    }

    std::error_code ignored;
    std::filesystem::remove(manifest_path, ignored);
    std::filesystem::remove(container_path, ignored);
    return 0;
}
