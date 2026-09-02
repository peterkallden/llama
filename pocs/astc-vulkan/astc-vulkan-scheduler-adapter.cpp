#include "astc-vulkan-scheduler-adapter.h"

#include <fstream>

namespace {
std::vector<uint8_t> read_bytes(const std::string & path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return {};
    const std::streamsize size = file.tellg();
    if (size <= 0) return {};
    std::vector<uint8_t> result(static_cast<size_t>(size));
    file.seekg(0);
    file.read(reinterpret_cast<char *>(result.data()), size);
    return file ? result : std::vector<uint8_t>();
}
}

bool astc_vulkan_scheduler_adapter::prepare(
        const std::string & manifest_path, const std::string & payload_blob_path,
        const std::string & tensor_name, astc_vulkan_footprint footprint,
        std::string & error) {
    astc_vulkan_manifest manifest;
    const std::vector<uint8_t> blob = read_bytes(payload_blob_path);
    if (blob.empty() || !astc_vulkan_read_manifest(manifest_path, manifest, error) ||
        !astc_vulkan_validate_payload_blob(manifest, blob.size(), error)) return false;
    const astc_vulkan_tensor_record * record = astc_vulkan_find_tensor(manifest, tensor_name);
    if (record == nullptr || record->footprint != footprint ||
        record->byte_offset > blob.size() || record->byte_size > blob.size() - record->byte_offset) {
        error = "ASTC scheduler adapter tensor artifact is invalid";
        return false;
    }
    payload_.assign(blob.begin() + static_cast<size_t>(record->byte_offset),
                    blob.begin() + static_cast<size_t>(record->byte_offset + record->byte_size));
    tensor_name_ = tensor_name;
    if (!sidecar_.set_manifest(manifest, error) || !sidecar_.init(footprint, error)) return false;
    if (!sidecar_.bind_tensor(tensor_name, record->width, record->height, payload_, binding_, error)) {
        reset();
        return false;
    }
    error.clear();
    return ready();
}

bool astc_vulkan_scheduler_adapter::run(const std::vector<uint32_t> & spirv,
                                        const std::vector<float> & activations,
                                        std::vector<float> & output, std::string & error) {
    if (!ready()) {
        error = "ASTC scheduler adapter is not ready; use normal fallback";
        return false;
    }
    return sidecar_.run(spirv, activations, output, error);
}
