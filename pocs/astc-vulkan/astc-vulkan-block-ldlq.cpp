#include "astc-vulkan-block-ldlq.h"

#include <cmath>
#include <utility>

const char * astc_vulkan_ldlq_order_name(astc_vulkan_ldlq_order order) {
    switch (order) {
        case astc_vulkan_ldlq_order::forward: return "forward";
        case astc_vulkan_ldlq_order::reverse: return "reverse";
        case astc_vulkan_ldlq_order::pivot:   return "pivot";
    }
    return "unknown";
}

bool astc_vulkan_ldlq_solve_damped_block(
    const std::vector<double> & gram,
    uint32_t dimension,
    uint32_t first_column,
    uint32_t count,
    double damping,
    const std::vector<double> & rhs,
    std::vector<double> & output) {
    if (dimension == 0 || count == 0 || first_column + count > dimension ||
        gram.size() != static_cast<size_t>(dimension) * dimension || rhs.size() != count ||
        !std::isfinite(damping) || damping < 0.0) return false;
    std::vector<double> matrix(static_cast<size_t>(count) * count, 0.0);
    for (uint32_t row = 0; row < count; ++row) {
        for (uint32_t column = 0; column < count; ++column) {
            matrix[static_cast<size_t>(row) * count + column] =
                gram[static_cast<size_t>(first_column + row) * dimension + first_column + column];
        }
        matrix[static_cast<size_t>(row) * count + row] += damping;
    }
    output = rhs;
    for (uint32_t pivot = 0; pivot < count; ++pivot) {
        uint32_t best_row = pivot;
        for (uint32_t row = pivot + 1; row < count; ++row) {
            if (std::abs(matrix[static_cast<size_t>(row) * count + pivot]) >
                std::abs(matrix[static_cast<size_t>(best_row) * count + pivot])) best_row = row;
        }
        if (std::abs(matrix[static_cast<size_t>(best_row) * count + pivot]) < 1e-12) return false;
        if (best_row != pivot) {
            for (uint32_t column = pivot; column < count; ++column) {
                std::swap(matrix[static_cast<size_t>(pivot) * count + column],
                          matrix[static_cast<size_t>(best_row) * count + column]);
            }
            std::swap(output[pivot], output[best_row]);
        }
        const double pivot_value = matrix[static_cast<size_t>(pivot) * count + pivot];
        for (uint32_t column = pivot; column < count; ++column) {
            matrix[static_cast<size_t>(pivot) * count + column] /= pivot_value;
        }
        output[pivot] /= pivot_value;
        for (uint32_t row = 0; row < count; ++row) {
            if (row == pivot) continue;
            const double factor = matrix[static_cast<size_t>(row) * count + pivot];
            for (uint32_t column = pivot; column < count; ++column) {
                matrix[static_cast<size_t>(row) * count + column] -=
                    factor * matrix[static_cast<size_t>(pivot) * count + column];
            }
            output[row] -= factor * output[pivot];
        }
    }
    return true;
}
