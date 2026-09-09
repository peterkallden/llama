#include "astc-vulkan-compiled-catalog.h"

#include "astc-vulkan-format.h"
#include "astc-vulkan-tensor-catalog.h"

#include <array>
#include <fstream>
#include <limits>
#include <unordered_set>
#include <utility>

namespace {

constexpr std::array<char, 8> kMagic = {'A', 'S', 'T', 'C', 'C', '0', '0', '1'};
constexpr uint32_t kVersion = 1;
constexpr uint32_t kMaxStringBytes = 1u << 20;
constexpr uint32_t kMaxTensorRecords = 1u << 20;

// ASTCC001 has a fixed little-endian wire representation. Do not serialize
// host structs directly: catalogs are cache artifacts and must be portable
// across CPU architectures and reproducible byte-for-byte.
bool write_u8(std::ofstream & file, uint8_t value) {
    file.put(static_cast<char>(value));
    return file.good();
}

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

bool read_u8(std::ifstream & file, uint8_t & value) {
    char byte = 0;
    if (!file.get(byte)) return false;
    value = static_cast<uint8_t>(static_cast<unsigned char>(byte));
    return true;
}

bool read_u32(std::ifstream & file, uint32_t & value) {
    value = 0;
    for (unsigned shift = 0; shift < 32; shift += 8) {
        uint8_t byte = 0;
        if (!read_u8(file, byte)) return false;
        value |= static_cast<uint32_t>(byte) << shift;
    }
    return true;
}

bool read_u64(std::ifstream & file, uint64_t & value) {
    value = 0;
    for (unsigned shift = 0; shift < 64; shift += 8) {
        uint8_t byte = 0;
        if (!read_u8(file, byte)) return false;
        value |= static_cast<uint64_t>(byte) << shift;
    }
    return true;
}

bool write_string(std::ofstream & file, const std::string & value) {
    if (value.size() > kMaxStringBytes) return false;
    const uint32_t size = static_cast<uint32_t>(value.size());
    return write_u32(file, size) &&
        (size == 0 || (file.write(value.data(), size), file.good()));
}

bool read_string(std::ifstream & file, std::string & value) {
    uint32_t size = 0;
    if (!read_u32(file, size) || size > kMaxStringBytes) return false;
    value.resize(size);
    return size == 0 || (file.read(value.data(), size), file.good());
}

bool valid_kind(astc_vulkan_compiled_storage_kind kind) {
    return kind == astc_vulkan_compiled_storage_kind::native ||
           kind == astc_vulkan_compiled_storage_kind::astc_d1 ||
           kind == astc_vulkan_compiled_storage_kind::astc_d2;
}

std::string footprint_name(astc_vulkan_footprint footprint) {
    const astc_vulkan_format_info format = astc_vulkan_format(footprint);
    return std::to_string(format.block_width) + "x" + std::to_string(format.block_height);
}

std::string storage_class(const astc_vulkan_artifact_record & artifact) {
    const std::string prefix = artifact.storage.representation == astc_vulkan_representation::kPairedD2 ?
        "astc.d2" : "astc.d1";
    const bool luminance_alpha =
        artifact.storage.representation == astc_vulkan_representation::kGaugeLumaAlpha ||
        artifact.paired_semantic == astc_vulkan_paired_semantic::luminance_alpha;
    const std::string semantic = luminance_alpha ?
        ".la" : ".direct";
    return prefix + semantic + "." + footprint_name(artifact.storage.footprint);
}

bool add_artifact(const astc_vulkan_artifact_record & artifact,
                  astc_vulkan_compiled_catalog & result,
                  std::string & error) {
    astc_vulkan_compiled_tensor_record record;
    record.logical_name = artifact.storage.name;
    record.semantic_role = artifact.storage.semantic_role.empty() ?
        astc_vulkan_tensor_semantic_role(record.logical_name) : artifact.storage.semantic_role;
    record.canonical_path = artifact.storage.canonical_path.empty() ?
        astc_vulkan_tensor_canonical_path(record.logical_name) : artifact.storage.canonical_path;
    record.storage_class = storage_class(artifact);
    record.artifact_id = artifact.id;
    record.storage_kind = artifact.storage.representation == astc_vulkan_representation::kPairedD2 ?
        astc_vulkan_compiled_storage_kind::astc_d2 : astc_vulkan_compiled_storage_kind::astc_d1;
    record.width = artifact.storage.width;
    record.height = artifact.storage.height;
    record.payload_offset = artifact.storage.byte_offset;
    record.payload_size = artifact.storage.byte_size;
    record.layout_offset = artifact.storage.layout_byte_offset;
    record.layout_size = artifact.storage.layout_byte_size;
    record.row_scale_offset = artifact.row_scale_byte_offset;
    record.row_scale_size = artifact.row_scale_byte_size;
    record.pair_map_offset = artifact.pair_map_byte_offset;
    record.pair_map_size = artifact.pair_map_byte_size;
    if (record.logical_name.empty() || record.width == 0 || record.height == 0 ||
        record.payload_size == 0) {
        error = "manifest artifact cannot be represented in compiled ASTC catalog";
        return false;
    }
    result.tensors.push_back(std::move(record));
    error.clear();
    return true;
}

} // namespace

bool astc_vulkan_compiled_catalog_from_manifest(
    const astc_vulkan_manifest & manifest,
    astc_vulkan_compiled_catalog & result,
    std::string & error) {
    if (!astc_vulkan_validate_manifest(manifest, error)) return false;
    result = {};
    result.version = kVersion;
    result.logical_model_id = manifest.model_fingerprint;
    result.source_model_fingerprint = manifest.model_fingerprint;
    if (!manifest.artifacts.empty()) {
        result.tensors.reserve(manifest.artifacts.size());
        for (const auto & artifact : manifest.artifacts) {
            if (!add_artifact(artifact, result, error)) return false;
        }
    } else {
        result.tensors.reserve(manifest.tensors.size());
        for (const auto & tensor : manifest.tensors) {
            astc_vulkan_artifact_record artifact;
            artifact.id = tensor.name;
            artifact.storage = tensor;
            artifact.paired_semantic = astc_vulkan_paired_semantic::direct_rgb;
            if (!add_artifact(artifact, result, error)) return false;
        }
    }
    return astc_vulkan_validate_compiled_catalog(result, error);
}

bool astc_vulkan_validate_compiled_catalog(
    const astc_vulkan_compiled_catalog & catalog,
    std::string & error) {
    if (catalog.version != kVersion || catalog.tensors.size() > kMaxTensorRecords) {
        error = "unsupported or oversized compiled ASTC catalog";
        return false;
    }
    std::unordered_set<std::string> artifact_ids;
    artifact_ids.reserve(catalog.tensors.size());
    for (const auto & tensor : catalog.tensors) {
        if (tensor.logical_name.empty() || tensor.logical_name.size() > kMaxStringBytes ||
            tensor.semantic_role.size() > kMaxStringBytes ||
            tensor.canonical_path.size() > kMaxStringBytes ||
            tensor.storage_class.size() > kMaxStringBytes ||
            tensor.artifact_id.size() > kMaxStringBytes ||
            tensor.native_type.size() > kMaxStringBytes || tensor.width == 0 || tensor.height == 0 ||
            !valid_kind(tensor.storage_kind) ||
            tensor.payload_size > std::numeric_limits<uint64_t>::max() - tensor.payload_offset ||
            tensor.layout_size > std::numeric_limits<uint64_t>::max() - tensor.layout_offset ||
            tensor.row_scale_size > std::numeric_limits<uint64_t>::max() - tensor.row_scale_offset ||
            tensor.pair_map_size > std::numeric_limits<uint64_t>::max() - tensor.pair_map_offset ||
            tensor.artifact_id.empty()) {
            error = "invalid compiled ASTC catalog tensor record";
            return false;
        }
        if (!artifact_ids.insert(tensor.artifact_id).second) {
            error = "duplicate compiled ASTC catalog artifact id";
            return false;
        }
        if (tensor.storage_kind == astc_vulkan_compiled_storage_kind::native &&
            !tensor.storage_class.empty()) {
            error = "native compiled catalog tensor has an ASTC storage class";
            return false;
        }
        if (tensor.storage_kind != astc_vulkan_compiled_storage_kind::native &&
            (tensor.storage_class.empty() || tensor.payload_size == 0)) {
            error = "ASTC compiled catalog tensor has no storage class";
            return false;
        }
    }
    error.clear();
    return true;
}

bool astc_vulkan_compiled_catalog_matches_manifest(
    const astc_vulkan_compiled_catalog & catalog,
    const astc_vulkan_manifest & manifest,
    std::string & error) {
    astc_vulkan_compiled_catalog expected;
    if (!astc_vulkan_compiled_catalog_from_manifest(manifest, expected, error)) return false;
    if (catalog.version != expected.version ||
        catalog.logical_model_id != expected.logical_model_id ||
        catalog.source_model_fingerprint != expected.source_model_fingerprint ||
        catalog.tensors.size() != expected.tensors.size()) {
        error = "compiled ASTC catalog does not match manifest header or record count";
        return false;
    }
    for (size_t index = 0; index < expected.tensors.size(); ++index) {
        const auto & actual = catalog.tensors[index];
        const auto & want = expected.tensors[index];
        if (actual.logical_name != want.logical_name ||
            actual.semantic_role != want.semantic_role ||
            actual.canonical_path != want.canonical_path ||
            actual.storage_class != want.storage_class ||
            actual.artifact_id != want.artifact_id ||
            actual.native_type != want.native_type ||
            actual.storage_kind != want.storage_kind ||
            actual.width != want.width || actual.height != want.height ||
            actual.payload_offset != want.payload_offset ||
            actual.payload_size != want.payload_size ||
            actual.layout_offset != want.layout_offset ||
            actual.layout_size != want.layout_size ||
            actual.row_scale_offset != want.row_scale_offset ||
            actual.row_scale_size != want.row_scale_size ||
            actual.pair_map_offset != want.pair_map_offset ||
            actual.pair_map_size != want.pair_map_size) {
            error = "compiled ASTC catalog record does not match manifest record " +
                std::to_string(index);
            return false;
        }
    }
    error.clear();
    return true;
}

bool astc_vulkan_write_compiled_catalog(
    const std::string & path,
    const astc_vulkan_compiled_catalog & catalog,
    std::string & error) {
    if (!astc_vulkan_validate_compiled_catalog(catalog, error)) return false;
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) { error = "cannot open compiled ASTC catalog for writing"; return false; }
    file.write(kMagic.data(), kMagic.size());
    if (!file.good() || !write_u32(file, catalog.version) ||
        !write_string(file, catalog.logical_model_id) ||
        !write_string(file, catalog.source_model_fingerprint) ||
        !write_string(file, catalog.tokenizer_fingerprint) ||
        !write_u32(file, static_cast<uint32_t>(catalog.tensors.size()))) {
        error = "cannot write compiled ASTC catalog header";
        return false;
    }
    for (const auto & tensor : catalog.tensors) {
        if (!write_string(file, tensor.logical_name) ||
            !write_string(file, tensor.semantic_role) ||
            !write_string(file, tensor.canonical_path) ||
            !write_string(file, tensor.storage_class) ||
            !write_string(file, tensor.artifact_id) ||
            !write_string(file, tensor.native_type) ||
            !write_u8(file, static_cast<uint8_t>(tensor.storage_kind)) ||
            !write_u32(file, tensor.width) || !write_u32(file, tensor.height) ||
            !write_u64(file, tensor.payload_offset) || !write_u64(file, tensor.payload_size) ||
            !write_u64(file, tensor.layout_offset) || !write_u64(file, tensor.layout_size) ||
            !write_u64(file, tensor.row_scale_offset) || !write_u64(file, tensor.row_scale_size) ||
            !write_u64(file, tensor.pair_map_offset) || !write_u64(file, tensor.pair_map_size)) {
            error = "cannot write compiled ASTC catalog tensor record";
            return false;
        }
    }
    error.clear();
    return true;
}

bool astc_vulkan_read_compiled_catalog(
    const std::string & path,
    astc_vulkan_compiled_catalog & catalog,
    std::string & error) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { error = "cannot open compiled ASTC catalog for reading"; return false; }
    std::array<char, 8> magic{};
    uint32_t count = 0;
    if (!file.read(magic.data(), magic.size()) || magic != kMagic ||
        !read_u32(file, catalog.version) ||
        !read_string(file, catalog.logical_model_id) ||
        !read_string(file, catalog.source_model_fingerprint) ||
        !read_string(file, catalog.tokenizer_fingerprint) ||
        !read_u32(file, count) || count > kMaxTensorRecords) {
        error = "invalid compiled ASTC catalog header";
        return false;
    }
    catalog.tensors.clear();
    catalog.tensors.reserve(count);
    for (uint32_t index = 0; index < count; ++index) {
        astc_vulkan_compiled_tensor_record tensor;
        uint8_t kind = 0;
        if (!read_string(file, tensor.logical_name) ||
            !read_string(file, tensor.semantic_role) ||
            !read_string(file, tensor.canonical_path) ||
            !read_string(file, tensor.storage_class) ||
            !read_string(file, tensor.artifact_id) ||
            !read_string(file, tensor.native_type) ||
            !read_u8(file, kind) || !read_u32(file, tensor.width) ||
            !read_u32(file, tensor.height) ||
            !read_u64(file, tensor.payload_offset) || !read_u64(file, tensor.payload_size) ||
            !read_u64(file, tensor.layout_offset) || !read_u64(file, tensor.layout_size) ||
            !read_u64(file, tensor.row_scale_offset) || !read_u64(file, tensor.row_scale_size) ||
            !read_u64(file, tensor.pair_map_offset) || !read_u64(file, tensor.pair_map_size)) {
            error = "truncated compiled ASTC catalog tensor record";
            return false;
        }
        tensor.storage_kind = static_cast<astc_vulkan_compiled_storage_kind>(kind);
        catalog.tensors.push_back(std::move(tensor));
    }
    if (file.peek() != std::ifstream::traits_type::eof()) {
        error = "compiled ASTC catalog has trailing bytes";
        return false;
    }
    return astc_vulkan_validate_compiled_catalog(catalog, error);
}
