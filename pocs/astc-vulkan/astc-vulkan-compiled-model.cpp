#include "astc-vulkan-compiled-model.h"

#include "astc-vulkan-cache.h"
#include "astc-vulkan-compiled-catalog.h"
#include "astc-vulkan-hash.h"
#include "astc-vulkan-manifest.h"

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>

namespace {

constexpr std::array<char, 8> kMagic = {'A', 'S', 'T', 'C', 'M', '0', '0', '1'};
constexpr uint32_t kMaxSectionCount = 8;
constexpr uint64_t kMaxSectionBytes = 16ull * 1024ull * 1024ull * 1024ull;

struct section_header {
    uint64_t size = 0;
    uint64_t hash = 0;
};

bool write_u32(std::ofstream & file, uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8) {
        file.put(static_cast<char>((value >> shift) & 0xffu));
    }
    return file.good();
}

bool write_u64(std::ofstream & file, uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8) {
        file.put(static_cast<char>((value >> shift) & 0xffu));
    }
    return file.good();
}

bool read_u32(std::ifstream & file, uint32_t & value) {
    value = 0;
    for (unsigned shift = 0; shift < 32; shift += 8) {
        char byte = 0;
        if (!file.get(byte)) return false;
        value |= static_cast<uint32_t>(static_cast<unsigned char>(byte)) << shift;
    }
    return true;
}

bool read_u64(std::ifstream & file, uint64_t & value) {
    value = 0;
    for (unsigned shift = 0; shift < 64; shift += 8) {
        char byte = 0;
        if (!file.get(byte)) return false;
        value |= static_cast<uint64_t>(static_cast<unsigned char>(byte)) << shift;
    }
    return true;
}

bool read_file(const std::string & path, std::vector<uint8_t> & bytes, std::string & error,
               bool required = true) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        if (!required) {
            bytes.clear();
            return true;
        }
        error = "cannot open compiled-model section: " + path;
        return false;
    }
    file.seekg(0, std::ios::end);
    const std::streamoff end = file.tellg();
    if (end < 0 || static_cast<uint64_t>(end) > kMaxSectionBytes) {
        error = "compiled-model section is too large: " + path;
        return false;
    }
    file.seekg(0, std::ios::beg);
    bytes.resize(static_cast<size_t>(end));
    if (!bytes.empty()) file.read(reinterpret_cast<char *>(bytes.data()), end);
    if (!file.good()) {
        error = "cannot read compiled-model section: " + path;
        return false;
    }
    return true;
}

uint64_t hash_bytes(const std::vector<uint8_t> & bytes) {
    return astc_vulkan_payload_hash64(bytes.data(), bytes.size());
}

bool write_section(std::ofstream & file, const std::vector<uint8_t> & bytes) {
    return bytes.empty() ||
        (file.write(reinterpret_cast<const char *>(bytes.data()),
                    static_cast<std::streamsize>(bytes.size())), file.good());
}

bool read_section(std::ifstream & file, const section_header & header,
                  std::vector<uint8_t> & bytes, std::string & error) {
    if (header.size > kMaxSectionBytes || header.size >
            static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        error = "compiled-model section exceeds safety limit";
        return false;
    }
    bytes.resize(static_cast<size_t>(header.size));
    if (!bytes.empty() && !file.read(reinterpret_cast<char *>(bytes.data()),
                                     static_cast<std::streamsize>(bytes.size()))) {
        error = "truncated compiled-model section";
        return false;
    }
    if (hash_bytes(bytes) != header.hash) {
        error = "compiled-model section checksum mismatch";
        return false;
    }
    return true;
}

std::vector<const std::vector<uint8_t> *> sections(const astc_vulkan_compiled_model & model) {
    return {&model.gguf, &model.manifest, &model.payload, &model.layout,
            &model.row_scales, &model.pair_map, &model.provenance, &model.catalog};
}

std::vector<std::vector<uint8_t> *> mutable_sections(astc_vulkan_compiled_model & model) {
    return {&model.gguf, &model.manifest, &model.payload, &model.layout,
            &model.row_scales, &model.pair_map, &model.provenance, &model.catalog};
}

} // namespace

bool astc_vulkan_compiled_model_validate(
    const astc_vulkan_compiled_model & model,
    std::string & error) {
    if (model.version != astc_vulkan_compiled_model::kCurrentVersion ||
        model.gguf.empty() || model.manifest.empty() || model.payload.empty() ||
        model.source_model_fingerprint.empty()) {
        error = "compiled-model is missing required GGUF, manifest, payload or fingerprint";
        return false;
    }
    for (const auto * section : sections(model)) {
        if (section->size() > kMaxSectionBytes) {
            error = "compiled-model section exceeds safety limit";
            return false;
        }
    }

    // Parse the embedded manifest as a final structural gate.  Payload and
    // per-blob checksums were already verified when the cache was packed; this
    // check ensures the container cannot contain arbitrary required bytes.
    const auto nonce = static_cast<uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count()) ^
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&model));
    const std::filesystem::path temp = std::filesystem::temp_directory_path() /
        ("astc-vulkan-compiled-model-manifest-" + std::to_string(nonce) + ".astcv");
    {
        std::ofstream file(temp, std::ios::binary | std::ios::trunc);
        if (!file || !write_section(file, model.manifest)) {
            error = "cannot stage embedded compiled-model manifest";
            return false;
        }
    }
    astc_vulkan_manifest manifest;
    const bool ok = astc_vulkan_read_manifest(temp.string(), manifest, error);
    std::error_code ignored;
    std::filesystem::remove(temp, ignored);
    if (!ok || !astc_vulkan_validate_manifest(manifest, error) ||
        !astc_vulkan_validate_payload_blob(manifest, model.payload.size(), error) ||
        !astc_vulkan_validate_layout_blob(manifest, model.layout.size(), error)) {
        return false;
    }
    for (const auto & artifact : manifest.artifacts) {
        if (artifact.row_scale_byte_offset > model.row_scales.size() ||
            artifact.row_scale_byte_size > model.row_scales.size() - artifact.row_scale_byte_offset ||
            artifact.pair_map_byte_offset > model.pair_map.size() ||
            artifact.pair_map_byte_size > model.pair_map.size() - artifact.pair_map_byte_offset) {
            error = "compiled-model metadata range exceeds its section";
            return false;
        }
    }
    error.clear();
    return true;
}

bool astc_vulkan_compiled_model_write(
    const std::string & output_path,
    const astc_vulkan_compiled_model & model,
    std::string & error) {
    if (!astc_vulkan_compiled_model_validate(model, error)) return false;
    if (model.source_model_fingerprint.size() > (1u << 20)) {
        error = "compiled-model fingerprint is too long";
        return false;
    }

    const std::filesystem::path output = output_path;
    if (std::filesystem::exists(output)) {
        error = "compiled-model output already exists: " + output_path;
        return false;
    }
    const std::filesystem::path staging = output.string() + ".tmp";
    std::error_code ignored;
    std::filesystem::remove(staging, ignored);
    std::ofstream file(staging, std::ios::binary | std::ios::trunc);
    if (!file) {
        error = "cannot open compiled-model output: " + staging.string();
        return false;
    }
    file.write(kMagic.data(), kMagic.size());
    if (!file.good() || !write_u32(file, model.version) || !write_u32(file, kMaxSectionCount) ||
        !write_u32(file, static_cast<uint32_t>(model.source_model_fingerprint.size())) ||
        !write_u32(file, 0) ||
        (!model.source_model_fingerprint.empty() &&
         (file.write(model.source_model_fingerprint.data(),
                     static_cast<std::streamsize>(model.source_model_fingerprint.size())),
          !file.good()))) {
        error = "cannot write compiled-model header";
        file.close();
        std::filesystem::remove(staging, ignored);
        return false;
    }
    for (const auto * section : sections(model)) {
        if (!write_u64(file, section->size()) || !write_u64(file, hash_bytes(*section))) {
            error = "cannot write compiled-model section header";
            file.close();
            std::filesystem::remove(staging, ignored);
            return false;
        }
    }
    for (const auto * section : sections(model)) {
        if (!write_section(file, *section)) {
            error = "cannot write compiled-model section";
            file.close();
            std::filesystem::remove(staging, ignored);
            return false;
        }
    }
    const bool write_ok = file.good();
    file.close();
    if (!write_ok) {
        error = "cannot flush compiled-model output: " + output_path;
        std::filesystem::remove(staging, ignored);
        return false;
    }
    std::error_code publish_error;
    std::filesystem::rename(staging, output, publish_error);
    if (publish_error) {
        error = "cannot publish compiled-model output: " + output_path;
        std::filesystem::remove(staging, ignored);
        return false;
    }
    error.clear();
    return true;
}

bool astc_vulkan_compiled_model_read(
    const std::string & input_path,
    astc_vulkan_compiled_model & model,
    std::string & error) {
    model = {};
    std::ifstream file(input_path, std::ios::binary);
    if (!file) {
        error = "cannot open compiled-model input: " + input_path;
        return false;
    }
    std::array<char, 8> magic{};
    uint32_t version = 0, count = 0, fingerprint_size = 0, reserved = 0;
    if (!file.read(magic.data(), magic.size()) || magic != kMagic ||
        !read_u32(file, version) || !read_u32(file, count) ||
        !read_u32(file, fingerprint_size) || !read_u32(file, reserved) ||
        version != astc_vulkan_compiled_model::kCurrentVersion || count != kMaxSectionCount ||
        fingerprint_size > (1u << 20)) {
        error = "invalid compiled-model header";
        return false;
    }
    model.version = version;
    model.source_model_fingerprint.resize(fingerprint_size);
    if (fingerprint_size &&
        !file.read(model.source_model_fingerprint.data(), fingerprint_size)) {
        error = "truncated compiled-model fingerprint";
        return false;
    }
    std::array<section_header, kMaxSectionCount> headers{};
    uint64_t total = 0;
    for (auto & header : headers) {
        if (!read_u64(file, header.size) || !read_u64(file, header.hash) ||
            header.size > kMaxSectionBytes ||
            header.size > std::numeric_limits<uint64_t>::max() - total) {
            error = "invalid compiled-model section table";
            return false;
        }
        total += header.size;
    }
    auto target = mutable_sections(model);
    for (size_t index = 0; index < target.size(); ++index) {
        if (!read_section(file, headers[index], *target[index], error)) return false;
    }
    if (file.peek() != std::ifstream::traits_type::eof()) {
        error = "compiled-model has trailing bytes";
        return false;
    }
    return astc_vulkan_compiled_model_validate(model, error);
}

bool astc_vulkan_compiled_model_extract_gguf(
    const astc_vulkan_compiled_model & model,
    const std::string & output_path,
    std::string & error) {
    if (!astc_vulkan_compiled_model_validate(model, error)) return false;
    std::ofstream file(output_path, std::ios::binary | std::ios::trunc);
    if (!file || !write_section(file, model.gguf)) {
        error = "cannot write extracted GGUF: " + output_path;
        return false;
    }
    error.clear();
    return true;
}

bool astc_vulkan_compiled_model_pack(
    const std::string & model_path,
    const std::string & requested_cache_path,
    const std::string & output_path,
    std::string & error) {
    astc_vulkan_cache_validation validation;
    if (!astc_vulkan_cache_validate(model_path, requested_cache_path, validation, error)) {
        return false;
    }
    astc_vulkan_compiled_model model;
    if (!read_file(model_path, model.gguf, error)) return false;
    if (!read_file(validation.paths.manifest, model.manifest, error) ||
        !read_file(validation.paths.payload, model.payload, error) ||
        !read_file(validation.paths.layout, model.layout, error, false) ||
        !read_file(validation.paths.row_scales, model.row_scales, error, false) ||
        !read_file(validation.paths.pair_map, model.pair_map, error, false) ||
        !read_file(validation.paths.provenance, model.provenance, error, false) ||
        !read_file(validation.paths.catalog, model.catalog, error, false)) {
        return false;
    }
    // Keep the source fingerprint as text; it is provenance rather than a
    // separate payload section.
    std::vector<uint8_t> fingerprint_bytes;
    if (!read_file(validation.paths.source_sha256, fingerprint_bytes, error)) return false;
    model.source_model_fingerprint.assign(
        reinterpret_cast<const char *>(fingerprint_bytes.data()), fingerprint_bytes.size());
    while (!model.source_model_fingerprint.empty() &&
           (model.source_model_fingerprint.back() == '\n' ||
            model.source_model_fingerprint.back() == '\r')) {
        model.source_model_fingerprint.pop_back();
    }
    return astc_vulkan_compiled_model_write(output_path, model, error);
}
