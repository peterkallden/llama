#pragma once

#include <string>
#include <vector>

// Offline candidate families for scalar-anchored ASTC codec steering.
//
// A gauge perturbation presents L = q + G and A = q - G to ASTC, so the cheap
// semantic decoder preserves q before codec quantization. The ASTC encoder is
// discontinuous, however, so different legal payloads can decode to useful
// neural error directions. The exported payload is selected only after exact
// decode and validation-prefix stopping; this module never affects Vulkan
// runtime decoding.
//
// Suitable for: offline scalar-anchored gauge, weight-grid gauge, and the
// bounded PV-inspired coefficient grids. The scalar member is mandatory in
// every family, preserving an exact per-source/per-footprint fallback.
//
// Research context: PV-Tuning, arXiv:2405.14852, motivates alternating or
// coefficient-space tuning at low bitrates. This PoC's `pv-lite` family is a
// finite candidate grid, not an implementation of full PV-Tuning.
enum class astc_vulkan_gauge_basis { constant, x_ramp, y_ramp, saddle };

struct astc_vulkan_gauge_factor {
    float correction = 0.0f;
    float gauge = 0.0f;
    astc_vulkan_gauge_basis basis = astc_vulkan_gauge_basis::constant;
};

const char * astc_vulkan_gauge_basis_name(astc_vulkan_gauge_basis basis);

// Evaluates the normalized basis at x,y in [-1, 1].
float astc_vulkan_gauge_basis_value(astc_vulkan_gauge_basis basis, float x, float y);

// Produces the exact historical candidate ordering used by the latent smoke.
// The ordering is part of deterministic tie-breaking and must not change
// without an artifact-version change.
std::vector<astc_vulkan_gauge_factor> astc_vulkan_make_gauge_factors(
    bool scalar_anchored_c_delta, bool scalar_anchored_gauge,
    bool weight_grid_gauge, bool pv_lite_grid, bool pv_lite_coarse_grid);

const char * astc_vulkan_gauge_candidate_family(
    bool weight_grid_gauge, bool pv_lite_grid, bool pv_lite_coarse_grid);
