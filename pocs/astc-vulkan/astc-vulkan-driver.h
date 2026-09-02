#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "astc-vulkan-format.h"
#include "astc-vulkan-manifest.h"

// Sidecar-only metadata for the ASTC Vulkan proof of concept.  This is kept
// independent of ggml-vulkan so the on-disk contract can mature before any
// production backend integration is attempted.

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
