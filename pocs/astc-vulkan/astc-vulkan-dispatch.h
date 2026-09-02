#pragma once

#include "astc-vulkan-manifest.h"

#include <cstddef>
#include <cstdint>

// Stable interface shared by the sidecar dispatch and astc-ffn-matvec.comp.
// Vulkan push constants are scalar-aligned here, so the C++ and GLSL layouts
// are six consecutive 32-bit values (24 bytes).
struct astc_vulkan_matvec_push_constants {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t sample_index = 0;
    float scale_l = 1.0f;
    float scale_a = 0.0f;
    float offset = 0.0f;
};

static_assert(sizeof(astc_vulkan_matvec_push_constants) == 24,
              "ASTC matvec push constants must match the GLSL block");

enum class astc_vulkan_descriptor_binding : uint32_t {
    kWeights = 0,
    kActivations = 1,
    kOutput = 2,
};

inline astc_vulkan_matvec_push_constants astc_vulkan_make_push_constants(
        uint32_t width, uint32_t height, uint32_t sample_index,
        const astc_vulkan_reconstruction & reconstruction) {
    return {width, height, sample_index, reconstruction.scale_l,
            reconstruction.scale_a, reconstruction.offset};
}
