#pragma once

// Offline PV-style alternating optimization for ASTC candidate families.
//
// The P-step proposes a continuous change to a small coefficient vector. The
// V-step is supplied by the caller and must project that vector through the
// exact legal ASTC encode/decode oracle. The objective is also supplied by the
// caller so activation-, model-, or YAQA-aware loss can be tested without
// changing the deployed 128-bit ASTC payload contract.
//
// This is intentionally a bounded coordinate search, not a claim that the
// original PV-Tuning optimizer is reproduced. It is suitable for offline
// low-rate experiments (10x6, 8x8 and below), with scalar/gauge-neutral
// fallback retained by the caller.
//
// Reference: Malinovskii et al., PV-Tuning: Beyond Straight-Through
// Estimation for Extreme LLM Compression, https://arxiv.org/abs/2405.14852

#include <cstdint>
#include <functional>
#include <vector>

using astc_vulkan_pv_projector = std::function<bool(
    const std::vector<float> & continuous, std::vector<float> & deployed)>;
using astc_vulkan_pv_objective = std::function<double(
    const std::vector<float> & deployed)>;

struct astc_vulkan_pv_result {
    std::vector<float> continuous;
    std::vector<float> deployed;
    double objective = 0.0;
    uint32_t accepted_steps = 0;
    uint32_t iterations = 0;
};

// Alternates bounded +/- coordinate P-steps with an exact caller-provided
// discrete V-step. The initial deployed point is always evaluated first; a
// step is accepted only when it strictly improves the objective.
bool astc_vulkan_pv_alternate(
    const std::vector<float> & initial,
    const std::vector<float> & coordinate_steps,
    uint32_t max_iterations,
    const astc_vulkan_pv_projector & project,
    const astc_vulkan_pv_objective & objective,
    astc_vulkan_pv_result & result);
