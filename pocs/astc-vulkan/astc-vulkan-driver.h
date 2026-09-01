#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Sidecar-only metadata for the ASTC Vulkan proof of concept.  This is kept
// independent of ggml-vulkan so the on-disk contract can mature before any
// production backend integration is attempted.

enum class astc_vulkan_footprint : uint8_t {
    k4x4 = 0,
    k5x5 = 1,
    k6x6 = 2,
};

struct astc_vulkan_format_info {
    uint32_t block_width;
    uint32_t block_height;
    uint32_t block_bytes;
};

astc_vulkan_format_info astc_vulkan_format(astc_vulkan_footprint footprint);
uint64_t astc_vulkan_block_count(astc_vulkan_footprint footprint,
                                 uint32_t width, uint32_t height);
uint64_t astc_vulkan_image_bytes(astc_vulkan_footprint footprint,
                                 uint32_t width, uint32_t height);

struct astc_vulkan_tensor_record {
    std::string name;
    uint32_t width = 0;
    uint32_t height = 0;
    astc_vulkan_footprint footprint = astc_vulkan_footprint::k6x6;
    uint64_t byte_offset = 0;
    uint64_t byte_size = 0;
};

struct astc_vulkan_manifest {
    uint32_t version = 1;
    std::string model_fingerprint;
    std::vector<astc_vulkan_tensor_record> tensors;
};

const astc_vulkan_tensor_record * astc_vulkan_find_tensor(
    const astc_vulkan_manifest & manifest, const std::string & name);

struct astc_vulkan_atlas_config {
    uint32_t max_width = 4096;
    uint32_t max_height = 4096;
};

struct astc_vulkan_atlas_placement {
    std::string name;
    astc_vulkan_footprint footprint = astc_vulkan_footprint::k6x6;
    uint32_t page = 0;
    uint32_t x = 0;
    uint32_t y = 0;
    uint32_t width = 0;
    uint32_t height = 0;
};

// Places tensors in deterministic row-major pages. Different ASTC footprints
// never share a page because a Vulkan image has one immutable format.
bool astc_vulkan_pack_atlas(const astc_vulkan_atlas_config & config,
                            const std::vector<astc_vulkan_tensor_record> & tensors,
                            std::vector<astc_vulkan_atlas_placement> & placements,
                            std::string & error);

bool astc_vulkan_validate_manifest(const astc_vulkan_manifest & manifest,
                                   std::string & error);
bool astc_vulkan_write_manifest(const std::string & path,
                                const astc_vulkan_manifest & manifest,
                                std::string & error);
bool astc_vulkan_read_manifest(const std::string & path,
                               astc_vulkan_manifest & manifest,
                               std::string & error);
