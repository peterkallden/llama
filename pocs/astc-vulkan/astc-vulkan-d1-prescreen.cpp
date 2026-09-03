#include "astc-vulkan-d1-prescreen.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

double bits_per_weight(astc_vulkan_footprint footprint) {
    const auto format = astc_vulkan_format(footprint);
    return 128.0 / static_cast<double>(format.block_width * format.block_height);
}

} // namespace

bool astc_vulkan_score_d1_prescreen_cpu(
    const std::vector<float> & weights, uint32_t rows, uint32_t columns,
    const std::vector<float> & column_energy,
    const std::vector<astc_vulkan_d1_prescreen_candidate> & candidates,
    std::vector<astc_vulkan_d1_prescreen_score> & scores) {
    scores.clear();
    if (rows == 0 || columns == 0 || weights.size() != static_cast<size_t>(rows) * columns ||
        column_energy.size() != columns || candidates.empty()) return false;
    double reference_energy = 0.0;
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t column = 0; column < columns; ++column) {
            const double value = weights[static_cast<size_t>(row) * columns + column];
            reference_energy += value * value * std::max(0.0f, column_energy[column]);
        }
    }
    scores.reserve(candidates.size());
    for (const auto candidate : candidates) {
        if (!astc_vulkan_footprint_is_valid(candidate.footprint) || candidate.levels < 2) {
            scores.clear();
            return false;
        }
        const auto format = astc_vulkan_format(candidate.footprint);
        double error = 0.0;
        for (uint32_t y0 = 0; y0 < rows; y0 += format.block_height) {
            for (uint32_t x0 = 0; x0 < columns; x0 += format.block_width) {
                float low = std::numeric_limits<float>::infinity();
                float high = -std::numeric_limits<float>::infinity();
                for (uint32_t y = y0; y < std::min(rows, y0 + format.block_height); ++y) {
                    for (uint32_t x = x0; x < std::min(columns, x0 + format.block_width); ++x) {
                        const float value = weights[static_cast<size_t>(y) * columns + x];
                        low = std::min(low, value);
                        high = std::max(high, value);
                    }
                }
                const float interval = high - low;
                const float step = interval / static_cast<float>(candidate.levels - 1);
                for (uint32_t y = y0; y < std::min(rows, y0 + format.block_height); ++y) {
                    for (uint32_t x = x0; x < std::min(columns, x0 + format.block_width); ++x) {
                        const float value = weights[static_cast<size_t>(y) * columns + x];
                        const float quantized = step > 0.0f
                            ? low + std::round((value - low) / step) * step : low;
                        const double difference = static_cast<double>(value) - quantized;
                        error += difference * difference * std::max(0.0f, column_energy[x]);
                    }
                }
            }
        }
        scores.push_back({candidate, error,
            reference_energy > 0.0 ? error / reference_energy : error, bits_per_weight(candidate.footprint)});
    }
    return true;
}

bool astc_vulkan_select_d1_prescreen_shortlist(
    const std::vector<astc_vulkan_d1_prescreen_score> & scores, uint32_t max_candidates,
    double lambda_bits, std::vector<astc_vulkan_d1_prescreen_score> & shortlist) {
    shortlist.clear();
    if (scores.empty() || max_candidates == 0 || !std::isfinite(lambda_bits) || lambda_bits < 0.0) return false;
    std::vector<size_t> order(scores.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](size_t lhs, size_t rhs) {
        const double left = scores[lhs].weighted_error + lambda_bits * scores[lhs].bits_per_weight;
        const double right = scores[rhs].weighted_error + lambda_bits * scores[rhs].bits_per_weight;
        return left < right;
    });
    const size_t count = std::min<size_t>(max_candidates, scores.size());
    shortlist.reserve(count);
    for (size_t i = 0; i < count; ++i) shortlist.push_back(scores[order[i]]);
    return true;
}
