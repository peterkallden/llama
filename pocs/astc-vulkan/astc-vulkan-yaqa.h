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
