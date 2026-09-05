#include "astc-vulkan-cache.h"

#include "astc-vulkan-hash.h"
#include "astc-vulkan-provenance.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <vector>

namespace {

namespace fs = std::filesystem;

constexpr const char * kManifestFile = "manifest.astcv";
constexpr const char * kPayloadFile = "payload.astcpack";
constexpr const char * kLayoutFile = "layout-map.bin";
constexpr const char * kRowScalesFile = "row-scales.bin";
constexpr const char * kProvenanceFile = "provenance.txt";
constexpr const char * kSourceHashFile = "source.gguf.sha256";
constexpr const char * kManifestHashFile = "manifest.sha256";
constexpr const char * kPayloadHashFile = "payload.sha256";
constexpr const char * kLayoutHashFile = "layout-map.sha256";
constexpr const char * kRowScalesHashFile = "row-scales.sha256";

bool read_hash(const std::string & path, std::string & hash) {
    std::ifstream file(path, std::ios::binary);
    if (!file || !std::getline(file, hash) || hash.size() != 64) return false;
    for (const unsigned char value : hash) {
        if (!((value >= '0' && value <= '9') || (value >= 'a' && value <= 'f'))) return false;
    }
    return true;
}

bool write_hash(const std::string & path, const std::string & hash) {
    std::ofstream file(path, std::ios::binary);
    file << hash << '\n';
    return file.good();
}

bool has_paired_d2(const astc_vulkan_manifest & manifest) {
    if (manifest.version == 4) {
        for (const auto & artifact : manifest.artifacts) {
            if (artifact.storage.representation == astc_vulkan_representation::kPairedD2) return true;
        }
        return false;
    }
    for (const auto & tensor : manifest.tensors) {
        if (tensor.representation == astc_vulkan_representation::kPairedD2) return true;
    }
    return false;
}

bool has_row_scales(const astc_vulkan_manifest & manifest) {
    if (manifest.version != 4) return false;
    for (const auto & artifact : manifest.artifacts) {
        if (artifact.normalization == astc_vulkan_normalization::per_row_absmax) return true;
    }
    return false;
}

bool file_size(const std::string & path, uint64_t & size) {
    std::error_code ec;
    size = fs::file_size(path, ec);
    return !ec;
}

bool range_hash64(const std::string & path, uint64_t offset, uint64_t size,
                  uint64_t & hash, std::vector<uint8_t> & buffer) {
    std::ifstream file(path, std::ios::binary);
    if (!file || offset > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max()) ||
        size > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) return false;
    file.seekg(static_cast<std::streamoff>(offset));
    if (!file) return false;
    if (buffer.empty()) buffer.resize(1u << 20);
    uint64_t remaining = size;
    hash = 1469598103934665603ULL;
    while (remaining != 0) {
        const std::streamsize count = static_cast<std::streamsize>(std::min<uint64_t>(remaining, buffer.size()));
        file.read(reinterpret_cast<char *>(buffer.data()), count);
        if (file.gcount() != count) return false;
        hash = astc_vulkan_fnv1a64_update(hash, buffer.data(), static_cast<size_t>(count));
        remaining -= static_cast<uint64_t>(count);
    }
    return true;
}

bool validate_tensor_payloads(const astc_vulkan_manifest & manifest,
                              uint64_t payload_size, const std::string & payload_path,
                              uint64_t layout_size, const std::string & layout_path,
                              uint64_t row_scale_size, const std::string & row_scale_path,
                              std::string & error) {
    if (!astc_vulkan_validate_payload_blob(manifest, payload_size, error)) return false;
    const bool paired = has_paired_d2(manifest);
    if (paired && !astc_vulkan_validate_layout_blob(manifest, layout_size, error)) return false;
    std::vector<uint8_t> buffer;
    const auto validate_storage = [&](const astc_vulkan_tensor_record & tensor) {
        uint64_t payload_hash = 0;
        if (!range_hash64(payload_path, tensor.byte_offset, tensor.byte_size, payload_hash, buffer) ||
            (tensor.payload_hash64 != 0 && payload_hash != tensor.payload_hash64)) {
            error = "ASTC cache tensor payload checksum mismatch";
            return false;
        }
        if (tensor.representation != astc_vulkan_representation::kPairedD2) return true;
        uint64_t layout_hash = 0;
        if (!range_hash64(layout_path, tensor.layout_byte_offset, tensor.layout_byte_size, layout_hash, buffer) ||
            (tensor.layout_hash64 != 0 && layout_hash != tensor.layout_hash64)) {
            error = "ASTC cache paired layout checksum mismatch";
            return false;
        }
        return true;
    };
    if (manifest.version == 4) {
        for (const auto & artifact : manifest.artifacts) {
            if (!validate_storage(artifact.storage)) return false;
            if (artifact.normalization != astc_vulkan_normalization::per_row_absmax) continue;
            uint64_t scale_hash = 0;
            if (artifact.row_scale_byte_offset > row_scale_size ||
                artifact.row_scale_byte_size > row_scale_size - artifact.row_scale_byte_offset ||
                !range_hash64(row_scale_path, artifact.row_scale_byte_offset,
                              artifact.row_scale_byte_size, scale_hash, buffer) ||
                (artifact.row_scale_hash64 != 0 && scale_hash != artifact.row_scale_hash64)) {
                error = "ASTC cache artifact row-scale checksum mismatch";
                return false;
            }
        }
    } else {
        for (const auto & tensor : manifest.tensors) {
            if (!validate_storage(tensor)) return false;
        }
    }
    error.clear();
    return true;
}

bool verify_hash(const std::string & path, const std::string & hash_path,
                 const char * label, std::string & error) {
    std::string expected, actual;
    if (!read_hash(hash_path, expected)) {
        error = std::string("missing or invalid ") + label + " SHA-256 record";
        return false;
    }
    if (!astc_vulkan_sha256_file_hex(path, actual, error)) return false;
    if (actual != expected) {
        error = std::string(label) + " SHA-256 mismatch";
        return false;
    }
    return true;
}

bool copy_file(const std::string & input, const std::string & output, std::string & error) {
    std::error_code ec;
    if (!fs::copy_file(input, output, fs::copy_options::none, ec)) {
        error = "cannot copy ASTC cache file: " + input;
        return false;
    }
    return true;
}

} // namespace

bool astc_vulkan_cache_resolve(const std::string & model_path,
                               const std::string & requested_cache_path,
                               astc_vulkan_cache_paths & paths,
                               std::string & error) {
    if (model_path.empty()) {
        error = "ASTC cache requires a source GGUF path";
        return false;
    }
    fs::path root;
    if (requested_cache_path.empty() || requested_cache_path == "auto") {
        root = fs::path(model_path).string() + ".astc-vulkan";
    } else {
        const fs::path requested(requested_cache_path);
        root = requested.filename() == kManifestFile ? requested.parent_path() : requested;
    }
    if (root.empty()) {
        error = "ASTC cache path is empty";
        return false;
    }
    paths.root = root.string();
    paths.manifest = (root / kManifestFile).string();
    paths.payload = (root / kPayloadFile).string();
    paths.layout = (root / kLayoutFile).string();
    paths.row_scales = (root / kRowScalesFile).string();
    paths.provenance = (root / kProvenanceFile).string();
    paths.source_sha256 = (root / kSourceHashFile).string();
    paths.manifest_sha256 = (root / kManifestHashFile).string();
    paths.payload_sha256 = (root / kPayloadHashFile).string();
    paths.layout_sha256 = (root / kLayoutHashFile).string();
    paths.row_scales_sha256 = (root / kRowScalesHashFile).string();
    error.clear();
    return true;
}

bool astc_vulkan_cache_validate(const std::string & model_path,
                                const std::string & requested_cache_path,
                                astc_vulkan_cache_validation & result,
                                std::string & error) {
    result = {};
    if (!astc_vulkan_cache_resolve(model_path, requested_cache_path, result.paths, error)) return false;
    std::error_code ec;
    if (!fs::is_regular_file(model_path, ec) || !fs::is_directory(result.paths.root, ec)) {
        error = "ASTC cache or source GGUF is unavailable";
        return false;
    }
    if (!verify_hash(model_path, result.paths.source_sha256, "source GGUF", error) ||
        !verify_hash(result.paths.manifest, result.paths.manifest_sha256, "manifest", error) ||
        !verify_hash(result.paths.payload, result.paths.payload_sha256, "payload", error) ||
        !astc_vulkan_read_manifest(result.paths.manifest, result.manifest, error)) return false;

    result.has_paired_d2 = has_paired_d2(result.manifest);
    result.has_row_scales = has_row_scales(result.manifest);
    uint64_t payload_size = 0;
    uint64_t layout_size = 0;
    uint64_t row_scale_size = 0;
    if (!file_size(result.paths.payload, payload_size)) {
        error = "cannot determine ASTC cache payload size";
        return false;
    }
    if (result.has_paired_d2) {
        if (!verify_hash(result.paths.layout, result.paths.layout_sha256, "layout map", error)) return false;
        if (!file_size(result.paths.layout, layout_size)) {
            error = "cannot determine ASTC cache layout size";
            return false;
        }
    }
    if (result.has_row_scales) {
        if (!verify_hash(result.paths.row_scales, result.paths.row_scales_sha256, "row-scales", error) ||
            !file_size(result.paths.row_scales, row_scale_size)) {
            if (error.empty()) error = "cannot determine ASTC cache row-scale size";
            return false;
        }
    }
    if (payload_size == 0 || (result.has_paired_d2 && layout_size == 0) ||
        (result.has_row_scales && row_scale_size == 0) ||
        !validate_tensor_payloads(result.manifest, payload_size, result.paths.payload,
                                  layout_size, result.paths.layout,
                                  row_scale_size, result.paths.row_scales, error)) return false;
    error.clear();
    return true;
}

bool astc_vulkan_cache_create(const std::string & model_path,
                              const std::string & manifest_input,
                              const std::string & payload_input,
                              const std::string & layout_input,
                              const std::string & provenance_input,
                              const std::string & requested_cache_path,
                              astc_vulkan_cache_paths & paths,
                              std::string & error) {
    return astc_vulkan_cache_create_with_row_scales(
        model_path, manifest_input, payload_input, layout_input, {}, provenance_input,
        requested_cache_path, paths, error);
}

bool astc_vulkan_cache_create_with_row_scales(const std::string & model_path,
                                              const std::string & manifest_input,
                                              const std::string & payload_input,
                                              const std::string & layout_input,
                                              const std::string & row_scales_input,
                                              const std::string & provenance_input,
                                              const std::string & requested_cache_path,
                                              astc_vulkan_cache_paths & paths,
                                              std::string & error) {
    if (!astc_vulkan_cache_resolve(model_path, requested_cache_path, paths, error)) return false;
    std::error_code ec;
    if (!fs::is_regular_file(model_path, ec) || !fs::is_regular_file(manifest_input, ec) ||
        !fs::is_regular_file(payload_input, ec)) {
        error = "ASTC cache creation requires a GGUF, manifest and payload file";
        return false;
    }
    if (fs::exists(paths.root, ec)) {
        error = "ASTC cache already exists; refuse to overwrite it";
        return false;
    }
    astc_vulkan_manifest manifest;
    if (!astc_vulkan_read_manifest(manifest_input, manifest, error)) return false;
    const bool paired = has_paired_d2(manifest);
    const bool scaled = has_row_scales(manifest);
    if (paired && (!fs::is_regular_file(layout_input, ec))) {
        error = "paired-D2 cache creation requires a layout map";
        return false;
    }
    if (scaled && (!fs::is_regular_file(row_scales_input, ec))) {
        error = "row-scaled artifact cache creation requires a row-scale blob";
        return false;
    }
    uint64_t payload_size = 0;
    uint64_t layout_size = 0;
    uint64_t row_scale_size = 0;
    if (!file_size(payload_input, payload_size) || payload_size == 0 ||
        (paired && (!file_size(layout_input, layout_size) || layout_size == 0)) ||
        (scaled && (!file_size(row_scales_input, row_scale_size) || row_scale_size == 0)) ||
        !validate_tensor_payloads(manifest, payload_size, payload_input, layout_size, layout_input,
                                  row_scale_size, row_scales_input, error)) return false;

    std::string source_hash, manifest_hash, payload_hash, layout_hash, row_scale_hash;
    if (!astc_vulkan_sha256_file_hex(model_path, source_hash, error) ||
        !astc_vulkan_sha256_file_hex(manifest_input, manifest_hash, error) ||
        !astc_vulkan_sha256_file_hex(payload_input, payload_hash, error) ||
        (paired && !astc_vulkan_sha256_file_hex(layout_input, layout_hash, error)) ||
        (scaled && !astc_vulkan_sha256_file_hex(row_scales_input, row_scale_hash, error))) return false;

    const fs::path partial = paths.root + ".partial-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    const auto fail = [&](const std::string & reason) {
        std::error_code remove_error;
        fs::remove_all(partial, remove_error);
        error = reason;
        return false;
    };
    if (!fs::create_directories(partial, ec) || ec) return fail("cannot create ASTC cache staging directory");
    const astc_vulkan_cache_paths staging = [&]() {
        astc_vulkan_cache_paths result;
        std::string ignored;
        astc_vulkan_cache_resolve(model_path, partial.string(), result, ignored);
        return result;
    }();
    if (!copy_file(manifest_input, staging.manifest, error) ||
        !copy_file(payload_input, staging.payload, error) ||
        (paired && !copy_file(layout_input, staging.layout, error)) ||
        (scaled && !copy_file(row_scales_input, staging.row_scales, error)) ||
        (!provenance_input.empty() && !copy_file(provenance_input, staging.provenance, error)) ||
        !write_hash(staging.source_sha256, source_hash) ||
        !write_hash(staging.manifest_sha256, manifest_hash) ||
        !write_hash(staging.payload_sha256, payload_hash) ||
        (paired && !write_hash(staging.layout_sha256, layout_hash)) ||
        (scaled && !write_hash(staging.row_scales_sha256, row_scale_hash))) {
        return fail(error.empty() ? "cannot write ASTC cache staging files" : error);
    }
    astc_vulkan_cache_validation validation;
    if (!astc_vulkan_cache_validate(model_path, partial.string(), validation, error)) return fail(error);
    fs::rename(partial, paths.root, ec);
    if (ec) return fail("cannot atomically publish ASTC cache");
    error.clear();
    return true;
}
