#include "astc-vulkan-compiled-model.h"
#include "astc-vulkan-manifest.h"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <vector>

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
    manifest.model_fingerprint = "compiled-model-test";
    astc_vulkan_tensor_record tensor;
    tensor.name = "blk.0.ffn_down.weight";
    tensor.width = 4;
    tensor.height = 4;
    tensor.footprint = astc_vulkan_footprint::k4x4;
    tensor.byte_size = 16;
    tensor.payload_hash64 = astc_vulkan_payload_hash64(
        reinterpret_cast<const uint8_t *>("0123456789abcdef"), 16);
    manifest.tensors.push_back(tensor);

    const auto root = std::filesystem::temp_directory_path();
    const auto manifest_path = root / "astc-vulkan-compiled-model-test.astcv";
    const auto output_path = root / "astc-vulkan-compiled-model-test.astccn";
    const auto extracted_path = root / "astc-vulkan-compiled-model-test.gguf";
    std::string error;
    assert(astc_vulkan_write_manifest(manifest_path.string(), manifest, error));

    astc_vulkan_compiled_model model;
    model.source_model_fingerprint = "sha256-test";
    model.gguf = {0x47, 0x47, 0x55, 0x46, 0x00, 0x01};
    model.manifest = read_bytes(manifest_path);
    model.payload.assign({'0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f'});
    model.layout = {1, 2, 3};
    model.row_scales = {4, 5};
    model.pair_map = {6};
    model.provenance = {'p','o','c'};
    model.catalog = {'A','S','T','C','C','0','0','1'};
    assert(astc_vulkan_compiled_model_validate(model, error));
    assert(astc_vulkan_compiled_model_write(output_path.string(), model, error));

    astc_vulkan_compiled_model loaded;
    assert(astc_vulkan_compiled_model_read(output_path.string(), loaded, error));
    assert(loaded.source_model_fingerprint == model.source_model_fingerprint);
    assert(loaded.gguf == model.gguf);
    assert(loaded.manifest == model.manifest);
    assert(loaded.payload == model.payload);
    assert(loaded.layout == model.layout);
    assert(loaded.row_scales == model.row_scales);
    assert(loaded.pair_map == model.pair_map);
    assert(loaded.provenance == model.provenance);
    assert(loaded.catalog == model.catalog);
    assert(astc_vulkan_compiled_model_extract_gguf(loaded, extracted_path.string(), error));
    assert(read_bytes(extracted_path) == model.gguf);

    std::error_code ignored;
    std::filesystem::remove(manifest_path, ignored);
    std::filesystem::remove(output_path, ignored);
    std::filesystem::remove(extracted_path, ignored);
    return 0;
}
