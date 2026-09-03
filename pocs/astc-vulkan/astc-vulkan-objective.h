#pragma once

// Offline neural candidate objectives for exact decoded ASTC error E = W-W_hat.
//
// Weight MSE remains a diagnostic and tail-safety check. Candidate selection at
// low rates instead uses the activation/output geometry seen by the model:
// activation, diagonal output-weighted activation, or a YAQA trace rerank.
// None of these functions run at Vulkan inference time.
//
// Suitable for: D1 low-rate and paired-D2 ASTC selection. The bounded
// activation variants are the default; the two-sided trace score is intended
// as a shortlist reranker, avoiding an expensive objective during candidate
// generation.
//
// References: Frantar et al., GPTQ (https://arxiv.org/abs/2210.17323), and
// Tseng, Sun, De Sa, Model-Preserving Adaptive Rounding
// (https://arxiv.org/abs/2505.22988).

#include <cstdint>
#include <vector>

enum class astc_vulkan_objective : unsigned char {
    activation,
    weighted_activation,
    two_sided_trace,
};

const char * astc_vulkan_objective_name(astc_vulkan_objective objective);

// Returns ||E X^T||_F^2 for input trace X (samples x columns).
double astc_vulkan_activation_score(const std::vector<float> & error, uint32_t rows,
                                    uint32_t columns, const std::vector<float> & input_trace,
                                    uint32_t samples);

// Returns sum_r row_sensitivity[r] * ||E_r X^T||^2. Non-negative diagonal
// sensitivities make this a cheap D2-friendly approximation to two-sided
// scoring without materializing an output Hessian.
double astc_vulkan_weighted_activation_score(
    const std::vector<float> & error, uint32_t rows, uint32_t columns,
    const std::vector<float> & input_trace, uint32_t samples,
    const std::vector<float> & row_sensitivity);
