#include "astc-vulkan-objective.h"

#include <cmath>

const char * astc_vulkan_objective_name(astc_vulkan_objective objective) {
    switch (objective) {
        case astc_vulkan_objective::activation: return "activation";
        case astc_vulkan_objective::weighted_activation: return "weighted-activation";
        case astc_vulkan_objective::two_sided_trace: return "two-sided-trace";
    }
    return "unknown";
}

double astc_vulkan_activation_score(const std::vector<float> & error, uint32_t rows,
                                    uint32_t columns, const std::vector<float> & input_trace,
                                    uint32_t samples) {
    if (rows == 0 || columns == 0 || samples == 0 ||
        error.size() != static_cast<size_t>(rows) * columns ||
        input_trace.size() != static_cast<size_t>(samples) * columns) return NAN;
    double score = 0.0;
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t sample = 0; sample < samples; ++sample) {
            double projected = 0.0;
            for (uint32_t column = 0; column < columns; ++column) {
                projected += static_cast<double>(error[static_cast<size_t>(row) * columns + column]) *
                    input_trace[static_cast<size_t>(sample) * columns + column];
            }
            score += projected * projected;
        }
    }
    return score;
}

double astc_vulkan_weighted_activation_score(
        const std::vector<float> & error, uint32_t rows, uint32_t columns,
        const std::vector<float> & input_trace, uint32_t samples,
        const std::vector<float> & row_sensitivity) {
    if (row_sensitivity.size() != rows) return NAN;
    double score = 0.0;
    if (rows == 0 || columns == 0 || samples == 0 ||
        error.size() != static_cast<size_t>(rows) * columns ||
        input_trace.size() != static_cast<size_t>(samples) * columns) return NAN;
    for (uint32_t row = 0; row < rows; ++row) {
        if (row_sensitivity[row] < 0.0f) return NAN;
        for (uint32_t sample = 0; sample < samples; ++sample) {
            double projected = 0.0;
            for (uint32_t column = 0; column < columns; ++column) {
                projected += static_cast<double>(error[static_cast<size_t>(row) * columns + column]) *
                    input_trace[static_cast<size_t>(sample) * columns + column];
            }
            score += static_cast<double>(row_sensitivity[row]) * projected * projected;
        }
    }
    return score;
}
