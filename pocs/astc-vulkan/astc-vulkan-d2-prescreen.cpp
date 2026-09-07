#include "astc-vulkan-d2-prescreen.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

bool valid_candidate(const astc_vulkan_d2_prescreen_candidate & candidate) {
    if (!astc_vulkan_footprint_is_valid(candidate.footprint) || candidate.levels < 2) return false;
    const auto fp = astc_vulkan_format(candidate.footprint);
    return fp.block_height == 5;
}

double bits_per_weight(astc_vulkan_footprint footprint) {
    const auto fp = astc_vulkan_format(footprint);
    return fp.block_width == 0 || fp.block_height == 0
        ? std::numeric_limits<double>::infinity()
        // A paired-D2 texel carries two logical neural weights. The
        // prescreen reports the logical model-storage rate, not the physical
        // ASTC texel rate, so it remains comparable with the cache profiles.
        : 128.0 / static_cast<double>(fp.block_width * fp.block_height * 2u);
}

float row_scale(const std::vector<float> & weights, uint32_t row, uint32_t columns) {
    float scale = 0.0f;
    const size_t offset = static_cast<size_t>(row) * columns;
    for (uint32_t column = 0; column < columns; ++column) {
        scale = std::max(scale, std::abs(weights[offset + column]));
    }
    return scale > 0.0f ? scale : 1.0f;
}

} // namespace

bool astc_vulkan_score_d2_prescreen_cpu(
        const std::vector<float> & weights, uint32_t rows, uint32_t columns,
        const std::vector<float> & column_energy,
        const std::vector<astc_vulkan_d2_prescreen_candidate> & candidates,
        std::vector<astc_vulkan_d2_prescreen_score> & scores) {
    scores.clear();
    if (rows == 0 || columns == 0 || (rows & 1u) != 0 ||
        weights.size() != static_cast<size_t>(rows) * columns ||
        column_energy.size() != columns || candidates.empty()) return false;

    double reference_energy = 0.0;
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t column = 0; column < columns; ++column) {
            const double value = weights[static_cast<size_t>(row) * columns + column];
            reference_energy += value * value * std::max(0.0f, column_energy[column]);
        }
    }

    scores.reserve(candidates.size());
    for (const auto & candidate : candidates) {
        if (!valid_candidate(candidate)) {
            scores.clear();
            return false;
        }
        const auto fp = astc_vulkan_format(candidate.footprint);
        double error = 0.0;
        for (uint32_t y0 = 0; y0 < rows; y0 += fp.block_height * 2u) {
            const uint32_t y_end = std::min(rows, y0 + fp.block_height * 2u);
            for (uint32_t x0 = 0; x0 < columns; x0 += fp.block_width) {
                const uint32_t x_end = std::min(columns, x0 + fp.block_width);
                float low[2] = { std::numeric_limits<float>::infinity(),
                                 std::numeric_limits<float>::infinity() };
                float high[2] = { -std::numeric_limits<float>::infinity(),
                                  -std::numeric_limits<float>::infinity() };
                for (uint32_t row = y0; row < y_end; ++row) {
                    const uint32_t lane = row & 1u;
                    const float scale = candidate.normalization ==
                        astc_vulkan_d2_prescreen_normalization::row_absmax
                        ? row_scale(weights, row, columns) : 1.0f;
                    for (uint32_t column = x0; column < x_end; ++column) {
                        const float value = weights[static_cast<size_t>(row) * columns + column] / scale;
                        low[lane] = std::min(low[lane], value);
                        high[lane] = std::max(high[lane], value);
                    }
                }
                const auto quantized_error = [&](uint32_t row, uint32_t lane, float scale) {
                    const float lo = low[lane];
                    const float interval = high[lane] - lo;
                    const float step = interval / static_cast<float>(candidate.levels - 1);
                    for (uint32_t column = x0; column < x_end; ++column) {
                        const float original = weights[static_cast<size_t>(row) * columns + column];
                        const float value = original / scale;
                        const float reconstructed = step > 0.0f
                            ? lo + std::round((value - lo) / step) * step : lo;
                        const double difference = static_cast<double>(original - reconstructed * scale);
                        error += difference * difference * std::max(0.0f, column_energy[column]);
                    }
                };
                for (uint32_t row = y0; row < y_end; ++row) {
                    const float scale = candidate.normalization ==
                        astc_vulkan_d2_prescreen_normalization::row_absmax
                        ? row_scale(weights, row, columns) : 1.0f;
                    quantized_error(row, row & 1u, scale);
                }
            }
        }
        // Source-derived Alpha is only a codec steering degree of freedom;
        // exact encode/decode must decide its value later.
        const double proxy_cost = error;
        scores.push_back({candidate, error,
            reference_energy > 0.0 ? error / reference_energy : error,
            bits_per_weight(candidate.footprint), proxy_cost});
    }
    return true;
}

bool astc_vulkan_d2_prescreen_calibration_energy(
        const std::vector<float> & trace, uint32_t trace_samples, uint32_t columns,
        uint32_t calibration_samples, std::vector<float> & column_energy) {
    column_energy.clear();
    if (trace_samples == 0 || columns == 0 || calibration_samples == 0 ||
        calibration_samples > trace_samples ||
        trace.size() != static_cast<size_t>(trace_samples) * columns) return false;
    column_energy.assign(columns, 0.0f);
    for (uint32_t sample = 0; sample < calibration_samples; ++sample) {
        for (uint32_t column = 0; column < columns; ++column) {
            const float value = trace[static_cast<size_t>(sample) * columns + column];
            column_energy[column] += value * value;
        }
    }
    return true;
}

bool astc_vulkan_select_d2_prescreen_shortlist(
        const std::vector<astc_vulkan_d2_prescreen_score> & scores,
        uint32_t max_candidates, double lambda_bits,
        std::vector<astc_vulkan_d2_prescreen_score> & shortlist) {
    shortlist.clear();
    if (scores.empty() || max_candidates == 0 || !std::isfinite(lambda_bits) || lambda_bits < 0.0) return false;
    std::vector<size_t> order(scores.size());
    for (size_t index = 0; index < order.size(); ++index) order[index] = index;
    std::stable_sort(order.begin(), order.end(), [&](size_t lhs, size_t rhs) {
        const double left = scores[lhs].proxy_cost + lambda_bits * scores[lhs].bits_per_weight;
        const double right = scores[rhs].proxy_cost + lambda_bits * scores[rhs].bits_per_weight;
        return left < right;
    });
    const size_t count = std::min<size_t>(max_candidates, order.size());
    shortlist.reserve(count);
    std::vector<bool> selected(scores.size(), false);
    bool have_direct = false, have_la = false;
    bool have_unscaled = false, have_scaled = false;
    bool have_neutral_alpha = false, have_source_alpha = false;
    auto coverage_gain = [&](const astc_vulkan_d2_prescreen_candidate & candidate) {
        uint32_t gain = 0;
        gain += candidate.semantic == astc_vulkan_d2_prescreen_semantic::direct ? !have_direct : !have_la;
        gain += candidate.normalization == astc_vulkan_d2_prescreen_normalization::none ? !have_unscaled : !have_scaled;
        gain += candidate.source_derived_alpha ? !have_source_alpha : !have_neutral_alpha;
        return gain;
    };
    auto mark_coverage = [&](const astc_vulkan_d2_prescreen_candidate & candidate) {
        if (candidate.semantic == astc_vulkan_d2_prescreen_semantic::direct) have_direct = true;
        else have_la = true;
        if (candidate.normalization == astc_vulkan_d2_prescreen_normalization::none) have_unscaled = true;
        else have_scaled = true;
        if (candidate.source_derived_alpha) have_source_alpha = true;
        else have_neutral_alpha = true;
    };
    // Greedily take the best candidate that expands categorical coverage.
    while (shortlist.size() < count) {
        size_t index = scores.size();
        uint32_t best_gain = 0;
        for (const size_t candidate_index : order) {
            if (selected[candidate_index]) continue;
            const uint32_t gain = coverage_gain(scores[candidate_index].candidate);
            if (gain > best_gain) { best_gain = gain; index = candidate_index; }
        }
        if (index == scores.size() || best_gain == 0) break;
        selected[index] = true;
        mark_coverage(scores[index].candidate);
        shortlist.push_back(scores[index]);
    }
    for (const size_t index : order) {
        if (shortlist.size() == count) break;
        if (selected[index]) continue;
        selected[index] = true;
        shortlist.push_back(scores[index]);
    }
    return true;
}
