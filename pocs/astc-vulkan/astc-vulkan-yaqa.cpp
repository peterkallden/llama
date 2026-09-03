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
