#include "astc-vulkan-scheduler-adapter.h"

#include "astc-vulkan-cache.h"

#include <filesystem>
#include <fstream>
#include <limits>

namespace {
bool read_range(const std::string & path, uint64_t offset, uint64_t size,
                std::vector<uint8_t> & result) {
    if (size == 0 || size > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
        offset > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max()) ||
        size > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) return false;
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;
    file.seekg(static_cast<std::streamoff>(offset));
    if (!file) return false;
    result.resize(static_cast<size_t>(size));
    file.read(reinterpret_cast<char *>(result.data()), static_cast<std::streamsize>(size));
    return file.good() || file.gcount() == static_cast<std::streamsize>(size);
}

bool get_file_size(const std::string & path, uint64_t & size) {
    std::error_code ec;
    size = std::filesystem::file_size(path, ec);
    return !ec;
}
}

bool astc_vulkan_scheduler_adapter::prepare(
        const std::string & manifest_path, const std::string & payload_blob_path,
        const std::string & tensor_name, astc_vulkan_footprint footprint,
        std::string & error, bool allow_experimental) {
    // Preparation is transactional: a failed reload must not leave a previous
    // tensor executable through ready() or run().
    reset();
    const auto fallback = [&](const std::string & reason, bool result) {
        binding_.status = astc_vulkan_binding_status::kFallback;
        binding_.fallback_reason = reason;
        error = reason;
        return result;
    };
    astc_vulkan_manifest manifest;
    uint64_t blob_size = 0;
    if (!get_file_size(payload_blob_path, blob_size) || blob_size == 0 ||
        !astc_vulkan_read_manifest(manifest_path, manifest, error) ||
        !astc_vulkan_validate_payload_blob(manifest, blob_size, error)) {
        return fallback(error.empty() ? "ASTC scheduler artifact is invalid" : error, false);
    }
    const astc_vulkan_tensor_record * record = astc_vulkan_find_tensor(manifest, tensor_name);
    if (record == nullptr || record->footprint != footprint ||
        record->byte_offset > blob_size || record->byte_size > blob_size - record->byte_offset) {
        return fallback("ASTC scheduler adapter tensor artifact is invalid", false);
    }
    // The production scheduler boundary is intentionally narrower than the
    // research sidecar: only standard 4x4/5x5/6x6 scalar/gauge payloads may
    // be selected automatically. Larger footprints and c+delta remain
    // explicit offline experiments and fall back to normal llama quantization.
    if (astc_vulkan_footprint_is_experimental(footprint) ||
        (record->representation != astc_vulkan_representation::kScalar &&
         record->representation != astc_vulkan_representation::kGaugeLumaAlpha)) {
        binding_.record = *record;
        return fallback("ASTC production adapter accepts only standard 4x4/5x5/6x6 scalar/gauge", true);
    }
    if (!read_range(payload_blob_path, record->byte_offset, record->byte_size, payload_)) {
        return fallback("ASTC scheduler adapter cannot stream tensor payload", false);
    }
    tensor_name_ = tensor_name;
    if (!sidecar_.set_manifest(manifest, error) ||
        !sidecar_.init(footprint, error, allow_experimental)) {
        return fallback(error.empty() ? "ASTC Vulkan device is unavailable" : error, false);
    }
    if (!sidecar_.bind_tensor(tensor_name, record->width, record->height, payload_, binding_, error)) {
        reset();
        return fallback(error.empty() ? "ASTC production adapter binding failed" : error, false);
    }
    if (binding_.status != astc_vulkan_binding_status::kReady) {
        if (binding_.fallback_reason.empty()) {
            binding_.fallback_reason = "ASTC production adapter binding requires normal fallback";
        }
        error = binding_.fallback_reason;
        return false;
    }
    error.clear();
    return true;
}

bool astc_vulkan_scheduler_adapter::prepare_from_cache(
        const std::string & model_path, const std::string & cache_path,
        const std::string & tensor_name, astc_vulkan_footprint footprint,
        std::string & error, bool allow_experimental) {
    reset();
    astc_vulkan_cache_validation cache;
    if (!astc_vulkan_cache_validate(model_path, cache_path, cache, error)) {
        binding_.status = astc_vulkan_binding_status::kFallback;
        binding_.fallback_reason = error.empty() ? "ASTC cache is unavailable" : error;
        error = binding_.fallback_reason;
        return false;
    }
    return prepare(cache.paths.manifest, cache.paths.payload, tensor_name, footprint,
                   error, allow_experimental);
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
