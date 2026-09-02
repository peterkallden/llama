#include "astc-vulkan-manifest.h"

#include "astc-vulkan-format.h"

#include <array>
#include <cmath>
#include <fstream>
#include <limits>

namespace {

constexpr std::array<char, 8> kMagic = {'K', 'A', 'S', 'T', 'C', 'V', 'M', '1'};
constexpr uint32_t kLegacyManifestVersion = 1;
constexpr uint32_t kCurrentManifestVersion = 2;
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
           representation == astc_vulkan_representation::kCDelta;
}

} // namespace

uint64_t astc_vulkan_payload_hash64(const uint8_t * data, size_t size) {
    uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0; i < size; ++i) {
        hash ^= data[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

bool astc_vulkan_validate_payload(const astc_vulkan_tensor_record & tensor,
                                  const uint8_t * data, size_t size,
                                  std::string & error) {
    const uint64_t expected = astc_vulkan_image_bytes(tensor.footprint,
                                                       tensor.width, tensor.height);
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

bool astc_vulkan_validate_manifest(const astc_vulkan_manifest & manifest,
                                   std::string & error) {
    if (manifest.version != kLegacyManifestVersion &&
        manifest.version != kCurrentManifestVersion) {
        error = "unsupported ASTC Vulkan manifest version";
        return false;
    }
    if (manifest.tensors.size() > kMaxTensorRecords) {
        error = "too many ASTC Vulkan tensor records";
        return false;
    }
    uint64_t previous_end = 0;
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
        const uint64_t expected_size = astc_vulkan_image_bytes(
            tensor.footprint, tensor.width, tensor.height);
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
        if (manifest.version >= kCurrentManifestVersion &&
            (!write_scalar(file, static_cast<uint8_t>(tensor.representation)) ||
             !write_scalar(file, tensor.scale_l) || !write_scalar(file, tensor.scale_a) ||
             !write_scalar(file, tensor.offset) || !write_scalar(file, tensor.payload_hash64))) {
            error = "cannot write ASTC Vulkan tensor metadata";
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
        if (version >= kCurrentManifestVersion &&
            (!read_scalar(file, representation) ||
             !read_scalar(file, tensor.scale_l) || !read_scalar(file, tensor.scale_a) ||
             !read_scalar(file, tensor.offset) || !read_scalar(file, tensor.payload_hash64))) {
            error = "truncated ASTC Vulkan tensor metadata";
            return false;
        }
        if (version >= kCurrentManifestVersion) {
            tensor.representation = static_cast<astc_vulkan_representation>(representation);
        }
        manifest.tensors.push_back(std::move(tensor));
    }
    return astc_vulkan_validate_manifest(manifest, error);
}
