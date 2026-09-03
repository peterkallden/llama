#include "astc-vulkan-manifest.h"
#include "astc-vulkan-hash.h"
#include "astc-vulkan-paired-layout.h"

#include "astc-vulkan-format.h"

#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <unordered_set>

namespace {

constexpr std::array<char, 8> kMagic = {'K', 'A', 'S', 'T', 'C', 'V', 'M', '1'};
constexpr uint32_t kLegacyManifestVersion = 1;
constexpr uint32_t kAffineManifestVersion = 2;
constexpr uint32_t kCurrentManifestVersion = 3;
constexpr uint32_t kMaxStringBytes = 1u << 20;
constexpr uint32_t kMaxTensorRecords = 1u << 20;

template<typename T>
bool write_scalar(std::ofstream & file, T value) {
    file.write(reinterpret_cast<const char *>(&value), sizeof(value));
    return file.good();
}

template<typename T>
bool read_scalar(std::ifstream & file, T & value) {
    file.read(reinterpret_cast<char *>(&value), sizeof(value));
    return file.good();
}

bool write_string(std::ofstream & file, const std::string & value) {
    if (value.size() > kMaxStringBytes) return false;
    const uint32_t size = static_cast<uint32_t>(value.size());
    return write_scalar(file, size) &&
           (size == 0 || (file.write(value.data(), size), file.good()));
}

bool read_string(std::ifstream & file, std::string & value) {
    uint32_t size = 0;
    if (!read_scalar(file, size) || size > kMaxStringBytes) return false;
    value.resize(size);
    return size == 0 || (file.read(value.data(), size), file.good());
}

bool valid_representation(astc_vulkan_representation representation) {
    return representation == astc_vulkan_representation::kScalar ||
           representation == astc_vulkan_representation::kGaugeLumaAlpha ||
           representation == astc_vulkan_representation::kCDelta ||
           representation == astc_vulkan_representation::kPairedD2;
}

bool is_paired_d2(const astc_vulkan_tensor_record & tensor) {
    return tensor.representation == astc_vulkan_representation::kPairedD2;
}

uint32_t storage_height(const astc_vulkan_tensor_record & tensor) {
    return is_paired_d2(tensor) ? astc_vulkan_paired_storage_height(tensor.height) : tensor.height;
}

uint64_t expected_payload_bytes(const astc_vulkan_tensor_record & tensor) {
    return astc_vulkan_image_bytes(tensor.footprint, tensor.width, storage_height(tensor));
}

uint64_t expected_layout_bytes(const astc_vulkan_tensor_record & tensor) {
    return is_paired_d2(tensor) ? astc_vulkan_paired_layout_bytes(
        tensor.footprint, tensor.width, tensor.height) : 0;
}

} // namespace

uint64_t astc_vulkan_payload_hash64(const uint8_t * data, size_t size) {
    return astc_vulkan_fnv1a64(data, size);
}

bool astc_vulkan_validate_payload(const astc_vulkan_tensor_record & tensor,
                                  const uint8_t * data, size_t size,
                                  std::string & error) {
    const uint64_t expected = expected_payload_bytes(tensor);
    if (expected == 0 || tensor.byte_size != expected || size != tensor.byte_size) {
        error = "ASTC Vulkan tensor payload size does not match its record";
        return false;
    }
    if (tensor.payload_hash64 != 0 &&
        astc_vulkan_payload_hash64(data, size) != tensor.payload_hash64) {
        error = "ASTC Vulkan tensor payload checksum mismatch";
        return false;
    }
    error.clear();
    return true;
}

bool astc_vulkan_validate_layout_map(const astc_vulkan_tensor_record & tensor,
                                     const uint8_t * data, size_t size,
                                     std::string & error) {
    const uint64_t expected = expected_layout_bytes(tensor);
    if (expected == 0 || tensor.layout_byte_size != expected || size != expected || data == nullptr) {
        error = "ASTC Vulkan paired layout map size does not match its record";
        return false;
    }
    if (tensor.layout_hash64 != 0 && astc_vulkan_payload_hash64(data, size) != tensor.layout_hash64) {
        error = "ASTC Vulkan paired layout map checksum mismatch";
        return false;
    }
    error.clear();
    return true;
}

bool astc_vulkan_validate_payload_blob(const astc_vulkan_manifest & manifest,
                                       uint64_t blob_size, std::string & error) {
    if (!astc_vulkan_validate_manifest(manifest, error)) return false;
    for (const astc_vulkan_tensor_record & tensor : manifest.tensors) {
        if (tensor.byte_offset > blob_size ||
            tensor.byte_size > blob_size - tensor.byte_offset) {
            error = "ASTC Vulkan tensor range exceeds payload blob";
            return false;
        }
    }
    error.clear();
    return true;
}

bool astc_vulkan_validate_layout_blob(const astc_vulkan_manifest & manifest,
                                      uint64_t blob_size, std::string & error) {
    if (!astc_vulkan_validate_manifest(manifest, error)) return false;
    for (const astc_vulkan_tensor_record & tensor : manifest.tensors) {
        if (!is_paired_d2(tensor)) continue;
        if (tensor.layout_byte_offset > blob_size ||
            tensor.layout_byte_size > blob_size - tensor.layout_byte_offset) {
            error = "ASTC Vulkan paired layout range exceeds layout blob";
            return false;
        }
    }
    error.clear();
    return true;
}

bool astc_vulkan_validate_manifest(const astc_vulkan_manifest & manifest,
                                   std::string & error) {
    if (manifest.version != kLegacyManifestVersion &&
        manifest.version != kAffineManifestVersion &&
        manifest.version != kCurrentManifestVersion) {
        error = "unsupported ASTC Vulkan manifest version";
        return false;
    }
    if (manifest.tensors.size() > kMaxTensorRecords) {
        error = "too many ASTC Vulkan tensor records";
        return false;
    }
    uint64_t previous_end = 0;
    std::unordered_set<std::string> names;
    names.reserve(manifest.tensors.size());
    for (const astc_vulkan_tensor_record & tensor : manifest.tensors) {
        if (tensor.name.empty() || tensor.name.size() > kMaxStringBytes ||
            tensor.width == 0 || tensor.height == 0 ||
            astc_vulkan_format(tensor.footprint).block_width == 0 ||
            !valid_representation(tensor.representation) ||
            !std::isfinite(tensor.scale_l) || !std::isfinite(tensor.scale_a) ||
            !std::isfinite(tensor.offset)) {
            error = "invalid ASTC Vulkan tensor record";
            return false;
        }
        if (!names.insert(tensor.name).second) {
            error = "duplicate ASTC Vulkan tensor name";
            return false;
        }
        if (is_paired_d2(tensor) &&
            (manifest.version < kCurrentManifestVersion || tensor.footprint != astc_vulkan_footprint::k8x5 ||
             tensor.layout_byte_size != expected_layout_bytes(tensor) ||
             tensor.layout_byte_size == 0 ||
             tensor.layout_byte_size > std::numeric_limits<uint64_t>::max() - tensor.layout_byte_offset)) {
            error = "invalid ASTC Vulkan paired-D2 layout metadata";
            return false;
        }
        if (!is_paired_d2(tensor) &&
            (tensor.layout_byte_offset != 0 || tensor.layout_byte_size != 0 || tensor.layout_hash64 != 0)) {
            error = "non-paired ASTC Vulkan tensor has layout metadata";
            return false;
        }
        const uint64_t expected_size = expected_payload_bytes(tensor);
        if (expected_size == 0 || tensor.byte_size != expected_size ||
            tensor.byte_offset < previous_end ||
            tensor.byte_size > std::numeric_limits<uint64_t>::max() - tensor.byte_offset) {
            error = "invalid ASTC Vulkan tensor byte range";
            return false;
        }
        previous_end = tensor.byte_offset + tensor.byte_size;
    }
    error.clear();
    return true;
}

const astc_vulkan_tensor_record * astc_vulkan_find_tensor(
    const astc_vulkan_manifest & manifest, const std::string & name) {
    for (const astc_vulkan_tensor_record & tensor : manifest.tensors) {
        if (tensor.name == name) return &tensor;
    }
    return nullptr;
}

bool astc_vulkan_write_manifest(const std::string & path,
                                const astc_vulkan_manifest & manifest,
                                std::string & error) {
    if (!astc_vulkan_validate_manifest(manifest, error)) return false;
    std::ofstream file(path, std::ios::binary);
    if (!file) { error = "cannot open ASTC Vulkan manifest for writing"; return false; }
    file.write(kMagic.data(), kMagic.size());
    const bool ok = file.good() && write_scalar(file, manifest.version) &&
                    write_string(file, manifest.model_fingerprint) &&
                    write_scalar(file, static_cast<uint32_t>(manifest.tensors.size()));
    if (!ok) { error = "cannot write ASTC Vulkan manifest header"; return false; }
    for (const astc_vulkan_tensor_record & tensor : manifest.tensors) {
        if (!write_string(file, tensor.name) || !write_scalar(file, tensor.width) ||
            !write_scalar(file, tensor.height) || !write_scalar(file,
                static_cast<uint8_t>(tensor.footprint)) ||
            !write_scalar(file, tensor.byte_offset) || !write_scalar(file, tensor.byte_size)) {
            error = "cannot write ASTC Vulkan tensor record";
            return false;
        }
        if (manifest.version >= kAffineManifestVersion &&
            (!write_scalar(file, static_cast<uint8_t>(tensor.representation)) ||
             !write_scalar(file, tensor.scale_l) || !write_scalar(file, tensor.scale_a) ||
             !write_scalar(file, tensor.offset) || !write_scalar(file, tensor.payload_hash64))) {
            error = "cannot write ASTC Vulkan tensor metadata";
            return false;
        }
        if (manifest.version >= kCurrentManifestVersion &&
            (!write_scalar(file, tensor.layout_byte_offset) ||
             !write_scalar(file, tensor.layout_byte_size) ||
             !write_scalar(file, tensor.layout_hash64))) {
            error = "cannot write ASTC Vulkan paired layout metadata";
            return false;
        }
    }
    error.clear();
    return true;
}

bool astc_vulkan_read_manifest(const std::string & path,
                               astc_vulkan_manifest & manifest,
                               std::string & error) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { error = "cannot open ASTC Vulkan manifest for reading"; return false; }
    std::array<char, 8> magic{};
    file.read(magic.data(), magic.size());
    uint32_t version = 0;
    uint32_t count = 0;
    if (!file.good() || magic != kMagic || !read_scalar(file, version) ||
        !read_string(file, manifest.model_fingerprint) || !read_scalar(file, count) ||
        count > kMaxTensorRecords) {
        error = "invalid ASTC Vulkan manifest header";
        return false;
    }
    manifest.version = version;
    manifest.tensors.clear();
    manifest.tensors.reserve(count);
    for (uint32_t index = 0; index < count; ++index) {
        astc_vulkan_tensor_record tensor;
        uint8_t footprint = 0;
        uint8_t representation = 0;
        if (!read_string(file, tensor.name) || !read_scalar(file, tensor.width) ||
            !read_scalar(file, tensor.height) || !read_scalar(file, footprint) ||
            !read_scalar(file, tensor.byte_offset) || !read_scalar(file, tensor.byte_size)) {
            error = "truncated ASTC Vulkan tensor record";
            return false;
        }
        tensor.footprint = static_cast<astc_vulkan_footprint>(footprint);
        if (version >= kAffineManifestVersion &&
            (!read_scalar(file, representation) ||
             !read_scalar(file, tensor.scale_l) || !read_scalar(file, tensor.scale_a) ||
             !read_scalar(file, tensor.offset) || !read_scalar(file, tensor.payload_hash64))) {
            error = "truncated ASTC Vulkan tensor metadata";
            return false;
        }
        if (version >= kAffineManifestVersion) {
            tensor.representation = static_cast<astc_vulkan_representation>(representation);
        }
        if (version >= kCurrentManifestVersion &&
            (!read_scalar(file, tensor.layout_byte_offset) ||
             !read_scalar(file, tensor.layout_byte_size) ||
             !read_scalar(file, tensor.layout_hash64))) {
            error = "truncated ASTC Vulkan paired layout metadata";
            return false;
        }
        manifest.tensors.push_back(std::move(tensor));
    }
    return astc_vulkan_validate_manifest(manifest, error);
}
