#pragma once

#include "astc-vulkan-format.h"

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

// Physical, footprint-aligned band shared by D1 and D2. The loader knows
// storage geometry only; representation semantics remain in the dispatch.
struct astc_vulkan_stream_geometry {
    astc_vulkan_footprint footprint = astc_vulkan_footprint::k6x6;
    uint32_t physical_width = 0;
    uint32_t physical_height = 0;
    uint32_t logical_height = 0;
    uint32_t block_width = 0;
    uint32_t block_height = 0;
    uint32_t blocks_x = 0;
    uint32_t blocks_y = 0;
    uint64_t payload_bytes = 0;
    bool paired = false;
};

struct astc_vulkan_stream_band {
    uint32_t block_row = 0;
    uint32_t block_row_count = 0;
    uint32_t physical_y = 0;
    uint32_t physical_height = 0;
    uint32_t logical_row_base = 0;
    uint32_t logical_row_count = 0;
    uint64_t payload_offset = 0;
    uint64_t payload_size = 0;
};

bool astc_vulkan_make_stream_geometry(astc_vulkan_footprint footprint,
                                       uint32_t width, uint32_t logical_height,
                                       bool paired,
                                       astc_vulkan_stream_geometry & geometry,
                                       std::string & error);

bool astc_vulkan_make_stream_band(const astc_vulkan_stream_geometry & geometry,
                                  uint32_t block_row, uint32_t block_row_count,
                                  astc_vulkan_stream_band & band,
                                  std::string & error);

// Builds bands whose payload ranges fit the requested resident budget. The
// budget is a payload budget; Vulkan image/staging alignment is accounted for
// by the caller's broader residency plan.
bool astc_vulkan_plan_stream(const astc_vulkan_stream_geometry & geometry,
                             uint64_t max_resident_payload_bytes,
                             std::vector<astc_vulkan_stream_band> & bands,
                             std::string & error);

// Reads exactly one validated payload range. This is deliberately independent
// of Vulkan so the cache and runtime can share the same range contract.
bool astc_vulkan_read_file_range(const std::string & path, uint64_t offset,
                                 uint64_t size, std::vector<uint8_t> & bytes,
                                 std::string & error);

// Reads physical bands from a tensor range inside a shared payload file. The
// object keeps no payload copy between calls, so peak host memory is bounded
// by the caller's band buffer.
class astc_vulkan_stream_payload_reader {
public:
    bool open(const std::string & path, uint64_t tensor_offset,
              const astc_vulkan_stream_geometry & geometry,
              std::string & error);
    bool read_band(const astc_vulkan_stream_band & band,
                   std::vector<uint8_t> & bytes, std::string & error) const;
    void close() { file_.close(); path_.clear(); file_size_ = 0; tensor_offset_ = 0; geometry_ = {}; }
    bool ready() const { return !path_.empty() && geometry_.payload_bytes != 0; }
    const astc_vulkan_stream_geometry & geometry() const { return geometry_; }

private:
    std::string path_;
    mutable std::ifstream file_;
    uint64_t file_size_ = 0;
    uint64_t tensor_offset_ = 0;
    astc_vulkan_stream_geometry geometry_{};
};
