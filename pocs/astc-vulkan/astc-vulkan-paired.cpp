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

const char * astc_vulkan_paired_semantic_name(astc_vulkan_paired_semantic semantic) {
    switch (semantic) {
        case astc_vulkan_paired_semantic::direct_rgb: return "direct-rgb";
        case astc_vulkan_paired_semantic::luminance_alpha: return "luminance-alpha";
    }
    return "unknown";
}

const char * astc_vulkan_paired_basis_name(astc_vulkan_paired_basis basis) {
    switch (basis) {
        case astc_vulkan_paired_basis::direct: return "direct";
        case astc_vulkan_paired_basis::common_difference: return "common-difference";
    }
    return "unknown";
}

const char * astc_vulkan_paired_steering_basis_name(astc_vulkan_paired_steering_basis basis) {
    switch (basis) {
        case astc_vulkan_paired_steering_basis::neutral: return "neutral";
        case astc_vulkan_paired_steering_basis::x_ramp: return "x-ramp";
        case astc_vulkan_paired_steering_basis::y_ramp: return "y-ramp";
        case astc_vulkan_paired_steering_basis::saddle: return "saddle";
        case astc_vulkan_paired_steering_basis::x_plus_y: return "x-plus-y";
        case astc_vulkan_paired_steering_basis::x_minus_y: return "x-minus-y";
    }
    return "unknown";
}

std::vector<astc_vulkan_paired_steering_factor> astc_vulkan_make_paired_steering_codebook() {
    using basis = astc_vulkan_paired_steering_basis;
    return {{0.0f, basis::neutral},
            {-0.5f, basis::x_ramp}, {0.5f, basis::x_ramp},
            {-0.5f, basis::y_ramp}, {0.5f, basis::y_ramp},
            {-0.5f, basis::saddle}, {0.5f, basis::saddle},
            {-0.5f, basis::x_plus_y}, {0.5f, basis::x_plus_y},
            {-0.5f, basis::x_minus_y}, {0.5f, basis::x_minus_y}};
}

float astc_vulkan_paired_steering_basis_value(
        astc_vulkan_paired_steering_basis basis, float x, float y) {
    switch (basis) {
        case astc_vulkan_paired_steering_basis::neutral: return 0.0f;
        case astc_vulkan_paired_steering_basis::x_ramp: return x;
        case astc_vulkan_paired_steering_basis::y_ramp: return y;
        case astc_vulkan_paired_steering_basis::saddle: return x * y;
        case astc_vulkan_paired_steering_basis::x_plus_y: return 0.5f * (x + y);
        case astc_vulkan_paired_steering_basis::x_minus_y: return 0.5f * (x - y);
    }
    return 0.0f;
}

astc_vulkan_rgba_texel astc_vulkan_make_paired_texel(
        float q0, float q1, float steering, astc_vulkan_paired_layout layout,
        astc_vulkan_paired_basis basis, astc_vulkan_paired_semantic semantic) {
    q0 = clamp_unorm(q0);
    q1 = clamp_unorm(q1);
    steering = clamp_unorm(steering);
    if (semantic == astc_vulkan_paired_semantic::luminance_alpha) {
        // The orientation map decides which logical member receives the
        // luminance lane. Alpha is semantic here, so steering is discarded.
        return layout == astc_vulkan_paired_layout::rg_b ?
            astc_vulkan_rgba_texel{q0, q0, q0, q1} :
            astc_vulkan_rgba_texel{q1, q1, q1, q0};
    }
    const float common = basis == astc_vulkan_paired_basis::common_difference ?
        0.5f * (q0 + q1) : q0;
    const float singleton = basis == astc_vulkan_paired_basis::common_difference ?
        0.5f + 0.5f * (q0 - q1) : q1;
    if (layout == astc_vulkan_paired_layout::rg_b) return {common, common, singleton, steering};
    return {common, singleton, singleton, steering};
}

float astc_vulkan_paired_weight(
        const astc_vulkan_rgba_texel & texel, unsigned int member,
        astc_vulkan_paired_layout layout, astc_vulkan_paired_basis basis,
        astc_vulkan_paired_semantic semantic) {
    if (semantic == astc_vulkan_paired_semantic::luminance_alpha) {
        const float luminance = (texel.r + texel.g + texel.b) / 3.0f;
        if (layout == astc_vulkan_paired_layout::rg_b) {
            return member == 0 ? luminance : texel.a;
        }
        return member == 0 ? texel.a : luminance;
    }
    const float common = layout == astc_vulkan_paired_layout::rg_b ?
        0.5f * (texel.r + texel.g) : texel.r;
    const float singleton = layout == astc_vulkan_paired_layout::rg_b ?
        texel.b : 0.5f * (texel.g + texel.b);
    if (basis == astc_vulkan_paired_basis::common_difference) {
        const float difference = singleton - 0.5f;
        return member == 0 ? common + difference : common - difference;
    }
    if (layout == astc_vulkan_paired_layout::rg_b) {
        return member == 0 ? 0.5f * (texel.r + texel.g) : texel.b;
    }
    return member == 0 ? texel.r : 0.5f * (texel.g + texel.b);
}
