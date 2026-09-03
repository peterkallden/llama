#include "astc-vulkan-paired.h"

#include <algorithm>

namespace {

float clamp_unorm(float value) {
    return std::clamp(value, 0.0f, 1.0f);
}

} // namespace

const char * astc_vulkan_semantic_density_name(astc_vulkan_semantic_density density) {
    switch (density) {
        case astc_vulkan_semantic_density::d1_scalar: return "scalar-d1";
        case astc_vulkan_semantic_density::d2_paired: return "paired-d2";
    }
    return "unknown";
}

const char * astc_vulkan_paired_layout_name(astc_vulkan_paired_layout layout) {
    switch (layout) {
        case astc_vulkan_paired_layout::rg_b: return "rg-b";
        case astc_vulkan_paired_layout::r_gb: return "r-gb";
    }
    return "unknown";
}

astc_vulkan_rgba_texel astc_vulkan_make_paired_texel(
        float q0, float q1, float steering, astc_vulkan_paired_layout layout) {
    q0 = clamp_unorm(q0);
    q1 = clamp_unorm(q1);
    steering = clamp_unorm(steering);
    if (layout == astc_vulkan_paired_layout::rg_b) return {q0, q0, q1, steering};
    return {q0, q1, q1, steering};
}

float astc_vulkan_paired_weight(
        const astc_vulkan_rgba_texel & texel, unsigned int member,
        astc_vulkan_paired_layout layout) {
    if (layout == astc_vulkan_paired_layout::rg_b) {
        return member == 0 ? 0.5f * (texel.r + texel.g) : texel.b;
    }
    return member == 0 ? texel.r : 0.5f * (texel.g + texel.b);
}
