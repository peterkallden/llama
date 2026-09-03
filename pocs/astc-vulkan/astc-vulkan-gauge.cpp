#include "astc-vulkan-gauge.h"

const char * astc_vulkan_gauge_basis_name(astc_vulkan_gauge_basis basis) {
    switch (basis) {
        case astc_vulkan_gauge_basis::constant: return "constant";
        case astc_vulkan_gauge_basis::x_ramp: return "x-ramp";
        case astc_vulkan_gauge_basis::y_ramp: return "y-ramp";
        case astc_vulkan_gauge_basis::saddle: return "saddle";
    }
    return "unknown";
}

float astc_vulkan_gauge_basis_value(astc_vulkan_gauge_basis basis, float x, float y) {
    switch (basis) {
        case astc_vulkan_gauge_basis::constant: return 1.0f;
        case astc_vulkan_gauge_basis::x_ramp: return x;
        case astc_vulkan_gauge_basis::y_ramp: return y;
        case astc_vulkan_gauge_basis::saddle: return x * y;
    }
    return 0.0f;
}

std::vector<astc_vulkan_gauge_factor> astc_vulkan_make_gauge_factors(
        bool scalar_anchored_c_delta, bool scalar_anchored_gauge,
        bool weight_grid_gauge, bool pv_lite_grid, bool pv_lite_coarse_grid) {
    using basis = astc_vulkan_gauge_basis;
    using factor = astc_vulkan_gauge_factor;
    if (scalar_anchored_c_delta) {
        return {{0.0f, 0.0f}, {0.0f, -0.5f}, {0.0f, 0.5f},
                {-0.25f, 0.0f}, {0.25f, 0.0f},
                {-0.25f, 0.25f}, {-0.25f, -0.25f}, {0.25f, 0.25f}};
    }
    if (!scalar_anchored_gauge) {
        return {{0.0f, 0.0f}, {0.0f, -0.5f}, {0.0f, 0.5f}, {0.0f, 1.0f},
                {0.0f, 1.5f}, {0.0f, 2.0f}};
    }
    if (!weight_grid_gauge) {
        return {{0.0f, 0.0f}, {0.0f, -0.25f}, {0.0f, 0.25f},
                {0.0f, -0.5f}, {0.0f, 0.5f}, {0.0f, -0.75f}, {0.0f, 0.75f}};
    }
    if (pv_lite_coarse_grid) {
        return {{0.0f, 0.0f, basis::constant},
                {0.0f, -0.5f, basis::x_ramp}, {0.0f, 0.5f, basis::x_ramp},
                {0.0f, -0.5f, basis::y_ramp}, {0.0f, 0.5f, basis::y_ramp},
                {0.0f, -0.5f, basis::saddle}, {0.0f, 0.5f, basis::saddle}};
    }
    if (pv_lite_grid) {
        return {{0.0f, 0.0f, basis::constant},
                {0.0f, -0.25f, basis::x_ramp}, {0.0f, 0.25f, basis::x_ramp},
                {0.0f, -0.50f, basis::x_ramp}, {0.0f, 0.50f, basis::x_ramp},
                {0.0f, -0.75f, basis::x_ramp}, {0.0f, 0.75f, basis::x_ramp},
                {0.0f, -0.25f, basis::y_ramp}, {0.0f, 0.25f, basis::y_ramp},
                {0.0f, -0.50f, basis::y_ramp}, {0.0f, 0.50f, basis::y_ramp},
                {0.0f, -0.75f, basis::y_ramp}, {0.0f, 0.75f, basis::y_ramp},
                {0.0f, -0.25f, basis::saddle}, {0.0f, 0.25f, basis::saddle},
                {0.0f, -0.50f, basis::saddle}, {0.0f, 0.50f, basis::saddle},
                {0.0f, -0.75f, basis::saddle}, {0.0f, 0.75f, basis::saddle}};
    }
    return {{0.0f, 0.0f, basis::constant},
            {0.0f, -0.5f, basis::x_ramp}, {0.0f, 0.5f, basis::x_ramp},
            {0.0f, -0.5f, basis::y_ramp}, {0.0f, 0.5f, basis::y_ramp},
            {0.0f, -0.5f, basis::saddle}, {0.0f, 0.5f, basis::saddle}};
}

const char * astc_vulkan_gauge_candidate_family(
        bool weight_grid_gauge, bool pv_lite_grid, bool pv_lite_coarse_grid) {
    if (pv_lite_coarse_grid) {
        return "zero-sum-pv-lite-coarse-grid-v1(constant,x-ramp,y-ramp,saddle;gauge=0,+/-0.50)";
    }
    if (pv_lite_grid) {
        return "zero-sum-pv-lite-grid-v1(constant,x-ramp,y-ramp,saddle;gauge=0,+/-0.25,+/-0.50,+/-0.75)";
    }
    return weight_grid_gauge ? "zero-sum-weight-grid-gauge-v1" :
                               "scalar-anchored-gauge-v1";
}
