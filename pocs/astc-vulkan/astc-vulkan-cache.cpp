#include "astc-vulkan-cache.h"

#include "astc-vulkan-provenance.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <vector>

namespace {

namespace fs = std::filesystem;

constexpr const char * kManifestFile = "manifest.astcv";
constexpr const char * kPayloadFile = "payload.astcpack";
constexpr const char * kLayoutFile = "layout-map.bin";
constexpr const char * kProvenanceFile = "provenance.txt";
constexpr const char * kSourceHashFile = "source.gguf.sha256";
constexpr const char * kManifestHashFile = "manifest.sha256";
constexpr const char * kPayloadHashFile = "payload.sha256";
constexpr const char * kLayoutHashFile = "layout-map.sha256";

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
    for (const auto & tensor : manifest.tensors) {
        if (tensor.representation == astc_vulkan_representation::kPairedD2) return true;
    }
    return false;
}

bool validate_tensor_payloads(const astc_vulkan_manifest & manifest,
                              const std::vector<uint8_t> & payload,
                              const std::vector<uint8_t> & layout,
                              std::string & error) {
    if (!astc_vulkan_validate_payload_blob(manifest, payload.size(), error)) return false;
    const bool paired = has_paired_d2(manifest);
    if (paired && !astc_vulkan_validate_layout_blob(manifest, layout.size(), error)) return false;
    for (const auto & tensor : manifest.tensors) {
        const auto payload_offset = static_cast<size_t>(tensor.byte_offset);
        if (!astc_vulkan_validate_payload(tensor, payload.data() + payload_offset,
                                          static_cast<size_t>(tensor.byte_size), error)) return false;
        if (tensor.representation != astc_vulkan_representation::kPairedD2) continue;
        const auto layout_offset = static_cast<size_t>(tensor.layout_byte_offset);
        if (!astc_vulkan_validate_layout_map(tensor, layout.data() + layout_offset,
                                             static_cast<size_t>(tensor.layout_byte_size), error)) return false;
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
    paths.provenance = (root / kProvenanceFile).string();
    paths.source_sha256 = (root / kSourceHashFile).string();
    paths.manifest_sha256 = (root / kManifestHashFile).string();
    paths.payload_sha256 = (root / kPayloadHashFile).string();
    paths.layout_sha256 = (root / kLayoutHashFile).string();
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

    const std::vector<uint8_t> payload = read_bytes(result.paths.payload);
    result.has_paired_d2 = has_paired_d2(result.manifest);
    std::vector<uint8_t> layout;
    if (result.has_paired_d2) {
        if (!verify_hash(result.paths.layout, result.paths.layout_sha256, "layout map", error)) return false;
        layout = read_bytes(result.paths.layout);
    }
    if (payload.empty() || (result.has_paired_d2 && layout.empty()) ||
        !validate_tensor_payloads(result.manifest, payload, layout, error)) return false;
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
    if (paired && (!fs::is_regular_file(layout_input, ec))) {
        error = "paired-D2 cache creation requires a layout map";
        return false;
    }
    const std::vector<uint8_t> payload = read_bytes(payload_input);
    const std::vector<uint8_t> layout = paired ? read_bytes(layout_input) : std::vector<uint8_t>();
    if (payload.empty() || (paired && layout.empty()) ||
        !validate_tensor_payloads(manifest, payload, layout, error)) return false;

    std::string source_hash, manifest_hash, payload_hash, layout_hash;
    if (!astc_vulkan_sha256_file_hex(model_path, source_hash, error) ||
        !astc_vulkan_sha256_file_hex(manifest_input, manifest_hash, error) ||
        !astc_vulkan_sha256_file_hex(payload_input, payload_hash, error) ||
        (paired && !astc_vulkan_sha256_file_hex(layout_input, layout_hash, error))) return false;

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
        (!provenance_input.empty() && !copy_file(provenance_input, staging.provenance, error)) ||
        !write_hash(staging.source_sha256, source_hash) ||
        !write_hash(staging.manifest_sha256, manifest_hash) ||
        !write_hash(staging.payload_sha256, payload_hash) ||
        (paired && !write_hash(staging.layout_sha256, layout_hash))) {
        return fail(error.empty() ? "cannot write ASTC cache staging files" : error);
    }
    astc_vulkan_cache_validation validation;
    if (!astc_vulkan_cache_validate(model_path, partial.string(), validation, error)) return fail(error);
    fs::rename(partial, paths.root, ec);
    if (ec) return fail("cannot atomically publish ASTC cache");
    error.clear();
    return true;
}
