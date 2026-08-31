#include "astc-vulkan-driver.h"

#include <array>
#include <algorithm>
#include <fstream>
#include <limits>

namespace {

constexpr std::array<char, 8> kMagic = {'K', 'A', 'S', 'T', 'C', 'V', 'M', '1'};
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

bool valid_footprint(astc_vulkan_footprint footprint) {
    return footprint == astc_vulkan_footprint::k4x4 ||
           footprint == astc_vulkan_footprint::k5x5 ||
           footprint == astc_vulkan_footprint::k6x6;
}

} // namespace

astc_vulkan_format_info astc_vulkan_format(astc_vulkan_footprint footprint) {
    switch (footprint) {
        case astc_vulkan_footprint::k4x4: return {4, 4, 16};
        case astc_vulkan_footprint::k5x5: return {5, 5, 16};
        case astc_vulkan_footprint::k6x6: return {6, 6, 16};
    }
    return {0, 0, 0};
}

uint64_t astc_vulkan_block_count(astc_vulkan_footprint footprint,
                                 uint32_t width, uint32_t height) {
    const astc_vulkan_format_info info = astc_vulkan_format(footprint);
    if (info.block_width == 0 || width == 0 || height == 0) return 0;
    const uint64_t blocks_x = (static_cast<uint64_t>(width) + info.block_width - 1) /
                              info.block_width;
    const uint64_t blocks_y = (static_cast<uint64_t>(height) + info.block_height - 1) /
                              info.block_height;
    return blocks_x * blocks_y;
}

uint64_t astc_vulkan_image_bytes(astc_vulkan_footprint footprint,
                                 uint32_t width, uint32_t height) {
    const uint64_t blocks = astc_vulkan_block_count(footprint, width, height);
    const astc_vulkan_format_info info = astc_vulkan_format(footprint);
    if (blocks > std::numeric_limits<uint64_t>::max() / info.block_bytes) return 0;
    return blocks * info.block_bytes;
}

bool astc_vulkan_validate_manifest(const astc_vulkan_manifest & manifest,
                                   std::string & error) {
    if (manifest.version != 1) {
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
            tensor.width == 0 || tensor.height == 0 || !valid_footprint(tensor.footprint)) {
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
        if (!read_string(file, tensor.name) || !read_scalar(file, tensor.width) ||
            !read_scalar(file, tensor.height) || !read_scalar(file, footprint) ||
            !read_scalar(file, tensor.byte_offset) || !read_scalar(file, tensor.byte_size)) {
            error = "truncated ASTC Vulkan tensor record";
            return false;
        }
        tensor.footprint = static_cast<astc_vulkan_footprint>(footprint);
        manifest.tensors.push_back(std::move(tensor));
    }
    return astc_vulkan_validate_manifest(manifest, error);
}

bool astc_vulkan_pack_atlas(const astc_vulkan_atlas_config & config,
                            const std::vector<astc_vulkan_tensor_record> & tensors,
                            std::vector<astc_vulkan_atlas_placement> & placements,
                            std::string & error) {
    if (config.max_width == 0 || config.max_height == 0) {
        error = "ASTC Vulkan atlas dimensions must be non-zero";
        return false;
    }
    placements.clear();
    struct cursor { uint32_t page = 0; uint32_t x = 0; uint32_t y = 0; uint32_t row_height = 0; };
    cursor cursors[3]{};
    for (const astc_vulkan_tensor_record & tensor : tensors) {
        if (tensor.name.empty() || tensor.width == 0 || tensor.height == 0 ||
            !valid_footprint(tensor.footprint)) {
            error = "invalid ASTC Vulkan atlas tensor";
            return false;
        }
        const astc_vulkan_format_info info = astc_vulkan_format(tensor.footprint);
        if (tensor.width > config.max_width || tensor.height > config.max_height) {
            error = "ASTC Vulkan tensor exceeds atlas dimensions";
            return false;
        }
        cursor & state = cursors[static_cast<size_t>(tensor.footprint)];
        const uint32_t aligned_width = ((tensor.width + info.block_width - 1) / info.block_width) * info.block_width;
        const uint32_t aligned_height = ((tensor.height + info.block_height - 1) / info.block_height) * info.block_height;
        if (aligned_width > config.max_width || aligned_height > config.max_height) {
            error = "ASTC Vulkan tensor block extent exceeds atlas dimensions";
            return false;
        }
        if (static_cast<uint64_t>(state.x) + aligned_width > config.max_width) {
            state.x = 0;
            state.y += state.row_height;
            state.row_height = 0;
        }
        if (state.y + tensor.height > config.max_height) {
            ++state.page;
            state.x = state.y = state.row_height = 0;
        }
        if (static_cast<uint64_t>(state.x) + aligned_width > config.max_width ||
            static_cast<uint64_t>(state.y) + aligned_height > config.max_height) {
            error = "ASTC Vulkan tensor cannot be placed in atlas";
            return false;
        }
        placements.push_back({tensor.name, tensor.footprint, state.page, state.x, state.y,
                              tensor.width, tensor.height});
        state.x += aligned_width;
        state.row_height = std::max(state.row_height, aligned_height);
    }
    error.clear();
    return true;
}
