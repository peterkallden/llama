#pragma once

// Two-sided YAQA-style sensitivity score for an already decoded ASTC error.
// Given E = W - W_hat and row/output and column/input sensitivity matrices,
// computes tr(H_out * E * H_in * E^T). This is an offline candidate-ranking
// objective only; it never runs in a Vulkan shader or changes ASTC payloads.
// Suitable for very low-rate selection (10x8, 10x10 and below), where
// immediate layer activation MSE can mis-rank downstream model impact.
//
// Reference: Tseng, Sun, and De Sa, Model-Preserving Adaptive Rounding,
// https://arxiv.org/abs/2505.22988

#include <cstdint>
#include <vector>

double astc_vulkan_yaqa_two_sided_score(
    const std::vector<float> & error,
    uint32_t rows,
    uint32_t columns,
    const std::vector<double> & output_hessian,
    const std::vector<double> & input_hessian);

// Low-rank trace form. `input_trace` is samples x columns and `output_trace`
// is samples x rows. With H_I = X^T X and H_O = Y^T Y this returns the same
// score as the dense form, while using O(samples * rows * columns) storage.
double astc_vulkan_yaqa_trace_score(
    const std::vector<float> & error,
    uint32_t rows,
    uint32_t columns,
    const std::vector<float> & input_trace,
    const std::vector<float> & output_trace,
    uint32_t samples);
