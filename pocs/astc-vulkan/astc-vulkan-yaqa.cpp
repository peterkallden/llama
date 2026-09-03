#include "astc-vulkan-yaqa.h"

#include <cmath>

double astc_vulkan_yaqa_two_sided_score(
    const std::vector<float> & error,
    uint32_t rows,
    uint32_t columns,
    const std::vector<double> & output_hessian,
    const std::vector<double> & input_hessian) {
    if (rows == 0 || columns == 0 || error.size() != static_cast<size_t>(rows) * columns ||
        output_hessian.size() != static_cast<size_t>(rows) * rows ||
        input_hessian.size() != static_cast<size_t>(columns) * columns) return NAN;
    double score = 0.0;
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t left = 0; left < columns; ++left) {
            const double left_error = error[static_cast<size_t>(row) * columns + left];
            for (uint32_t other_row = 0; other_row < rows; ++other_row) {
                const double output_weight = output_hessian[static_cast<size_t>(row) * rows + other_row];
                for (uint32_t right = 0; right < columns; ++right) {
                    score += output_weight * left_error *
                        input_hessian[static_cast<size_t>(left) * columns + right] *
                        error[static_cast<size_t>(other_row) * columns + right];
                }
            }
        }
    }
    return score;
}

double astc_vulkan_yaqa_trace_score(
    const std::vector<float> & error,
    uint32_t rows,
    uint32_t columns,
    const std::vector<float> & input_trace,
    const std::vector<float> & output_trace,
    uint32_t samples) {
    if (rows == 0 || columns == 0 || samples == 0 ||
        error.size() != static_cast<size_t>(rows) * columns ||
        input_trace.size() != static_cast<size_t>(samples) * columns ||
        output_trace.size() != static_cast<size_t>(samples) * rows) return NAN;

    // M = E X^T, then Y M = Y E X^T. This is algebraically identical to
    // tr((Y^T Y) E (X^T X) E^T), but avoids dense Hessian materialization.
    std::vector<double> projected(static_cast<size_t>(rows) * samples, 0.0);
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t sample = 0; sample < samples; ++sample) {
            double value = 0.0;
            for (uint32_t column = 0; column < columns; ++column) {
                value += static_cast<double>(error[static_cast<size_t>(row) * columns + column]) *
                    input_trace[static_cast<size_t>(sample) * columns + column];
            }
            projected[static_cast<size_t>(row) * samples + sample] = value;
        }
    }
    double score = 0.0;
    for (uint32_t output_sample = 0; output_sample < samples; ++output_sample) {
        for (uint32_t input_sample = 0; input_sample < samples; ++input_sample) {
            double value = 0.0;
            for (uint32_t row = 0; row < rows; ++row) {
                value += output_trace[static_cast<size_t>(output_sample) * rows + row] *
                    projected[static_cast<size_t>(row) * samples + input_sample];
            }
            score += value * value;
        }
    }
    return score;
}
