#include "astc-vulkan-d2-row-scale.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

bool valid_shape(const std::vector<float> & values, unsigned int rows, unsigned int columns,
                 const std::vector<astc_vulkan_d2_row_scale> & scales) {
    return rows != 0 && columns != 0 && values.size() == static_cast<size_t>(rows) * columns &&
           scales.size() == rows;
}

float safe_scale(float value) {
    return std::isfinite(value) && value > 0.0f ? value : 1.0f;
}

} // namespace

std::vector<astc_vulkan_d2_row_scale> astc_vulkan_d2_make_absmax_row_scales(
        const std::vector<float> & weights, unsigned int rows, unsigned int columns) {
    if (rows == 0 || columns == 0 || weights.size() != static_cast<size_t>(rows) * columns) return {};
    std::vector<astc_vulkan_d2_row_scale> result(rows);
    for (unsigned int row = 0; row < rows; ++row) {
        float maximum = 0.0f;
        for (unsigned int column = 0; column < columns; ++column) {
            const float value = weights[static_cast<size_t>(row) * columns + column];
            if (std::isfinite(value)) maximum = std::max(maximum, std::fabs(value));
        }
        result[row].value = safe_scale(maximum);
    }
    return result;
}

std::vector<float> astc_vulkan_d2_normalize_rows(
        const std::vector<float> & weights, unsigned int rows, unsigned int columns,
        const std::vector<astc_vulkan_d2_row_scale> & scales) {
    if (!valid_shape(weights, rows, columns, scales)) return {};
    std::vector<float> result(weights.size());
    for (unsigned int row = 0; row < rows; ++row) {
        const float scale = safe_scale(scales[row].value);
        for (unsigned int column = 0; column < columns; ++column) {
            result[static_cast<size_t>(row) * columns + column] =
                weights[static_cast<size_t>(row) * columns + column] / scale;
        }
    }
    return result;
}

std::vector<float> astc_vulkan_d2_restore_rows(
        const std::vector<float> & normalized, unsigned int rows, unsigned int columns,
        const std::vector<astc_vulkan_d2_row_scale> & scales) {
    if (!valid_shape(normalized, rows, columns, scales)) return {};
    std::vector<float> result(normalized.size());
    for (unsigned int row = 0; row < rows; ++row) {
        const float scale = safe_scale(scales[row].value);
        for (unsigned int column = 0; column < columns; ++column) {
            result[static_cast<size_t>(row) * columns + column] =
                normalized[static_cast<size_t>(row) * columns + column] * scale;
        }
    }
    return result;
}
