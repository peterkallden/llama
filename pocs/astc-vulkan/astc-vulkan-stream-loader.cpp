#include "astc-vulkan-stream-loader.h"

#include <algorithm>
#include <fstream>
#include <limits>

namespace {

bool checked_mul(uint64_t a, uint64_t b, uint64_t & result) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) return false;
    result = a * b;
    return true;
}

} // namespace

bool astc_vulkan_make_stream_geometry(
        astc_vulkan_footprint footprint, uint32_t width, uint32_t logical_height,
        bool paired, astc_vulkan_stream_geometry & geometry, std::string & error) {
    geometry = {};
    const astc_vulkan_format_info format = astc_vulkan_format(footprint);
    if (format.block_width == 0 || width == 0 || logical_height == 0) {
        error = "invalid ASTC stream geometry";
        return false;
    }
    const uint32_t physical_height = paired ? (logical_height + 1u) / 2u : logical_height;
    const uint64_t blocks_x = (static_cast<uint64_t>(width) + format.block_width - 1u) /
                              format.block_width;
    const uint64_t blocks_y = (static_cast<uint64_t>(physical_height) + format.block_height - 1u) /
                              format.block_height;
    uint64_t block_count = 0;
    uint64_t payload_bytes = 0;
    if (blocks_x > std::numeric_limits<uint32_t>::max() ||
        blocks_y > std::numeric_limits<uint32_t>::max() ||
        !checked_mul(blocks_x, blocks_y, block_count) ||
        !checked_mul(block_count, format.block_bytes, payload_bytes)) {
        error = "ASTC stream geometry is too large";
        return false;
    }
    geometry.footprint = footprint;
    geometry.physical_width = width;
    geometry.physical_height = physical_height;
    geometry.logical_height = logical_height;
    geometry.block_width = format.block_width;
    geometry.block_height = format.block_height;
    geometry.blocks_x = static_cast<uint32_t>(blocks_x);
    geometry.blocks_y = static_cast<uint32_t>(blocks_y);
    geometry.payload_bytes = payload_bytes;
    geometry.paired = paired;
    error.clear();
    return true;
}

bool astc_vulkan_make_stream_band(
        const astc_vulkan_stream_geometry & geometry, uint32_t block_row,
        uint32_t block_row_count, astc_vulkan_stream_band & band, std::string & error) {
    band = {};
    if (geometry.blocks_x == 0 || geometry.blocks_y == 0 || block_row >= geometry.blocks_y ||
        block_row_count == 0 || block_row_count > geometry.blocks_y - block_row) {
        error = "invalid ASTC stream band";
        return false;
    }
    const uint64_t bytes_per_row = static_cast<uint64_t>(geometry.blocks_x) * 16u;
    uint64_t payload_offset = 0;
    uint64_t payload_size = 0;
    if (!checked_mul(block_row, bytes_per_row, payload_offset) ||
        !checked_mul(block_row_count, bytes_per_row, payload_size)) {
        error = "ASTC stream band range overflow";
        return false;
    }
    const uint32_t physical_y = block_row * geometry.block_height;
    const uint32_t physical_height = std::min(
        geometry.physical_height - physical_y, block_row_count * geometry.block_height);
    const uint32_t logical_row_base = geometry.paired ? physical_y * 2u : physical_y;
    const uint32_t logical_capacity = geometry.paired ? physical_height * 2u : physical_height;
    const uint32_t logical_row_count = std::min(
        geometry.logical_height - std::min(logical_row_base, geometry.logical_height),
        logical_capacity);
    band.block_row = block_row;
    band.block_row_count = block_row_count;
    band.physical_y = physical_y;
    band.physical_height = physical_height;
    band.logical_row_base = std::min(logical_row_base, geometry.logical_height);
    band.logical_row_count = logical_row_count;
    band.payload_offset = payload_offset;
    band.payload_size = payload_size;
    error.clear();
    return true;
}

bool astc_vulkan_plan_stream(
        const astc_vulkan_stream_geometry & geometry, uint64_t max_resident_payload_bytes,
        std::vector<astc_vulkan_stream_band> & bands, std::string & error) {
    bands.clear();
    if (geometry.blocks_x == 0 || geometry.blocks_y == 0 || max_resident_payload_bytes == 0) {
        error = "invalid ASTC stream budget";
        return false;
    }
    const uint64_t bytes_per_row = static_cast<uint64_t>(geometry.blocks_x) * 16u;
    if (bytes_per_row == 0 || max_resident_payload_bytes < bytes_per_row) {
        error = "ASTC stream budget cannot hold one block row";
        return false;
    }
    const uint32_t max_rows = static_cast<uint32_t>(std::min<uint64_t>(
        geometry.blocks_y, max_resident_payload_bytes / bytes_per_row));
    for (uint32_t row = 0; row < geometry.blocks_y;) {
        const uint32_t count = std::min(max_rows, geometry.blocks_y - row);
        astc_vulkan_stream_band band;
        if (!astc_vulkan_make_stream_band(geometry, row, count, band, error)) {
            bands.clear();
            return false;
        }
        bands.push_back(band);
        row += count;
    }
    error.clear();
    return true;
}

bool astc_vulkan_read_file_range(
        const std::string & path, uint64_t offset, uint64_t size,
        std::vector<uint8_t> & bytes, std::string & error) {
    bytes.clear();
    if (size == 0 || size > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        error = "invalid ASTC cache range size";
        return false;
    }
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        error = "cannot open ASTC cache payload";
        return false;
    }
    const std::streamoff end = file.tellg();
    if (end < 0 || offset > static_cast<uint64_t>(end) || size > static_cast<uint64_t>(end) - offset) {
        error = "ASTC cache range is outside payload";
        return false;
    }
    file.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    bytes.resize(static_cast<size_t>(size));
    file.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(size));
    if (!file || file.gcount() != static_cast<std::streamsize>(size)) {
        bytes.clear();
        error = "cannot read ASTC cache payload range";
        return false;
    }
    error.clear();
    return true;
}

bool astc_vulkan_stream_payload_reader::open(
        const std::string & path, uint64_t tensor_offset,
        const astc_vulkan_stream_geometry & geometry, std::string & error) {
    close();
    if (path.empty() || geometry.payload_bytes == 0) {
        error = "invalid ASTC stream payload reader configuration";
        return false;
    }
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        error = "cannot open ASTC cache payload";
        return false;
    }
    const std::streamoff end = file.tellg();
    if (end < 0 || tensor_offset > static_cast<uint64_t>(end) ||
        geometry.payload_bytes > static_cast<uint64_t>(end) - tensor_offset) {
        error = "ASTC cache tensor range is outside payload";
        return false;
    }
    file.seekg(0, std::ios::beg);
    file_ = std::move(file);
    path_ = path;
    file_size_ = static_cast<uint64_t>(end);
    tensor_offset_ = tensor_offset;
    geometry_ = geometry;
    error.clear();
    return true;
}

bool astc_vulkan_stream_payload_reader::read_band(
        const astc_vulkan_stream_band & band, std::vector<uint8_t> & bytes,
        std::string & error) const {
    if (!ready() || !file_.is_open() || band.block_row >= geometry_.blocks_y || band.block_row_count == 0 ||
        band.block_row_count > geometry_.blocks_y - band.block_row ||
        band.payload_offset > geometry_.payload_bytes ||
        band.payload_size > geometry_.payload_bytes - band.payload_offset ||
        tensor_offset_ > std::numeric_limits<uint64_t>::max() - band.payload_offset ||
        tensor_offset_ + band.payload_offset > file_size_ ||
        band.payload_size > file_size_ - (tensor_offset_ + band.payload_offset) ||
        band.payload_size > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        error = "invalid ASTC stream payload band";
        return false;
    }
    file_.clear();
    file_.seekg(static_cast<std::streamoff>(tensor_offset_ + band.payload_offset), std::ios::beg);
    if (!file_) {
        error = "cannot seek ASTC cache payload range";
        return false;
    }
    bytes.resize(static_cast<size_t>(band.payload_size));
    file_.read(reinterpret_cast<char *>(bytes.data()),
               static_cast<std::streamsize>(band.payload_size));
    if (!file_ || file_.gcount() != static_cast<std::streamsize>(band.payload_size)) {
        bytes.clear();
        error = "cannot read ASTC cache payload range";
        return false;
    }
    error.clear();
    return true;
}
