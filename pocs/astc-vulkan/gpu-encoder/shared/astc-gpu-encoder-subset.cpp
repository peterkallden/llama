#include "astc-gpu-encoder-subset.h"

#include <algorithm>
#include <cmath>

namespace {

uint16_t unorm16(float value) {
    return static_cast<uint16_t>(std::lround(value * 65535.0f));
}

void write_bits(uint32_t value, uint32_t bit_count, uint32_t bit_offset,
                std::array<uint8_t, 16> & payload) {
    for (uint32_t bit = 0; bit < bit_count; ++bit) {
        const uint32_t position = bit_offset + bit;
        const uint8_t mask = static_cast<uint8_t>(1u << (position & 7u));
        if (value & (1u << bit)) payload[position >> 3u] |= mask;
        else payload[position >> 3u] &= static_cast<uint8_t>(~mask);
    }
}

uint8_t bit_reverse(uint8_t value) {
    value = static_cast<uint8_t>(((value & 0x55u) << 1u) | ((value >> 1u) & 0x55u));
    value = static_cast<uint8_t>(((value & 0x33u) << 2u) | ((value >> 2u) & 0x33u));
    return static_cast<uint8_t>((value << 4u) | (value >> 4u));
}

std::array<uint8_t, 16> pack_luminance_raw_weights(
    uint32_t block_mode, uint32_t endpoint_format,
    const uint8_t * endpoints, uint32_t endpoint_count,
    const uint8_t * weights, uint32_t weight_count, uint32_t bits_per_weight);

std::array<uint8_t, 16> pack_luminance_raw_weights(
    uint32_t block_mode, uint8_t endpoint_low, uint8_t endpoint_high,
    const uint8_t * weights, uint32_t weight_count, uint32_t bits_per_weight) {
    std::array<uint8_t, 2> endpoints{endpoint_low, endpoint_high};
    return pack_luminance_raw_weights(block_mode, 0u, endpoints.data(), endpoints.size(),
                                      weights, weight_count, bits_per_weight);
}

uint8_t quantize_unorm8(float value) {
    return static_cast<uint8_t>(std::floor(std::clamp(value, 0.0f, 1.0f) * 255.0f + 0.5f));
}

enum class binary_refinement_seed : uint8_t {
    minmax,
    mean_split,
    quartiles,
};

// A deliberately small Lloyd-style solve in the exact discrete domain used
// by our binary 6x6 subset. It is not a general ASTC optimizer: endpoints are
// direct UNORM8 values and every texel keeps a binary weight. Quantizing each
// iteration keeps CPU/GPU output deterministic. The three seeds deliberately
// expose a tiny local candidate bank; all output remains the same legal mode.
void refine_binary_luminance_6x6(
    const astc_gpu_encoder_source_block & source,
    uint8_t & endpoint_low, uint8_t & endpoint_high,
    std::array<uint8_t, 36> & weights, binary_refinement_seed seed) {
    float low = 1.0f;
    float high = 0.0f;
    for (const auto & texel : source.texels) {
        low = std::min(low, texel.rgba[0]);
        high = std::max(high, texel.rgba[0]);
    }
    if (seed == binary_refinement_seed::mean_split) {
        float mean = 0.0f;
        for (const auto & texel : source.texels) mean += texel.rgba[0];
        mean /= static_cast<float>(source.texels.size());
        float low_sum = 0.0f, high_sum = 0.0f;
        uint32_t low_count = 0, high_count = 0;
        for (const auto & texel : source.texels) {
            if (texel.rgba[0] < mean) { low_sum += texel.rgba[0]; ++low_count; }
            else { high_sum += texel.rgba[0]; ++high_count; }
        }
        if (low_count && high_count) {
            low = low_sum / static_cast<float>(low_count);
            high = high_sum / static_cast<float>(high_count);
        }
    } else if (seed == binary_refinement_seed::quartiles) {
        std::array<float, 36> sorted{};
        for (uint32_t index = 0; index < 36; ++index) sorted[index] = source.texels[index].rgba[0];
        std::sort(sorted.begin(), sorted.end());
        low = sorted[9];
        high = sorted[27];
    }
    for (uint32_t iteration = 0; iteration < 3; ++iteration) {
        endpoint_low = quantize_unorm8(low);
        endpoint_high = quantize_unorm8(high);
        const float low_value = static_cast<float>(endpoint_low) / 255.0f;
        const float high_value = static_cast<float>(endpoint_high) / 255.0f;
        const float threshold = 0.5f * (low_value + high_value);
        float low_sum = 0.0f, high_sum = 0.0f;
        uint32_t low_count = 0, high_count = 0;
        for (uint32_t index = 0; index < 36; ++index) {
            const float value = source.texels[index].rgba[0];
            const bool upper = value >= threshold;
            weights[index] = upper ? 1u : 0u;
            if (upper) {
                high_sum += value;
                ++high_count;
            } else {
                low_sum += value;
                ++low_count;
            }
        }
        if (low_count) low = low_sum / static_cast<float>(low_count);
        if (high_count) high = high_sum / static_cast<float>(high_count);
    }
    endpoint_low = quantize_unorm8(low);
    endpoint_high = quantize_unorm8(high);
    const float threshold = 0.5f * (static_cast<float>(endpoint_low) +
                                    static_cast<float>(endpoint_high)) / 255.0f;
    for (uint32_t index = 0; index < 36; ++index) {
        weights[index] = source.texels[index].rgba[0] >= threshold ? 1u : 0u;
    }
}

// D2-LA shares one binary weight plane between its two semantic lanes. These
// small deterministic solves expose alternate legal compromises without
// changing the physical ASTC mode or the D2 runtime decoder.
void refine_luminance_alpha_binary_8x5(
    const astc_gpu_encoder_source_block & source,
    std::array<uint8_t, 4> & endpoints,
    std::array<uint8_t, 40> & weights, binary_refinement_seed seed) {
    std::array<float, 2> low{1.0f, 1.0f};
    std::array<float, 2> high{0.0f, 0.0f};
    std::array<float, 40> scores{};
    float score_sum = 0.0f;
    for (uint32_t index = 0; index < 40; ++index) {
        const auto & texel = source.texels[index];
        low[0] = std::min(low[0], texel.rgba[0]); low[1] = std::min(low[1], texel.rgba[3]);
        high[0] = std::max(high[0], texel.rgba[0]); high[1] = std::max(high[1], texel.rgba[3]);
        scores[index] = 0.5f * (texel.rgba[0] + texel.rgba[3]);
        score_sum += scores[index];
    }
    if (seed != binary_refinement_seed::minmax) {
        float split = score_sum / 40.0f;
        if (seed == binary_refinement_seed::quartiles) {
            auto sorted = scores;
            std::sort(sorted.begin(), sorted.end());
            split = 0.5f * (sorted[10] + sorted[30]);
        }
        std::array<float, 2> lo_sum{}, hi_sum{};
        uint32_t lo_count = 0, hi_count = 0;
        for (uint32_t index = 0; index < 40; ++index) {
            const auto & texel = source.texels[index];
            auto & sum = scores[index] < split ? lo_sum : hi_sum;
            auto & count = scores[index] < split ? lo_count : hi_count;
            sum[0] += texel.rgba[0]; sum[1] += texel.rgba[3]; ++count;
        }
        if (lo_count && hi_count) {
            low = {lo_sum[0] / lo_count, lo_sum[1] / lo_count};
            high = {hi_sum[0] / hi_count, hi_sum[1] / hi_count};
        }
    }
    for (uint32_t iteration = 0; iteration < 3; ++iteration) {
        endpoints = {quantize_unorm8(low[0]), quantize_unorm8(high[0]),
                     quantize_unorm8(low[1]), quantize_unorm8(high[1])};
        std::array<uint32_t, 2> lo_sum{}, hi_sum{};
        uint32_t lo_count = 0, hi_count = 0;
        for (uint32_t index = 0; index < 40; ++index) {
            const auto & texel = source.texels[index];
            const int l = quantize_unorm8(texel.rgba[0]), a = quantize_unorm8(texel.rgba[3]);
            const int d0 = (l - endpoints[0]) * (l - endpoints[0]) + (a - endpoints[2]) * (a - endpoints[2]);
            const int d1 = (l - endpoints[1]) * (l - endpoints[1]) + (a - endpoints[3]) * (a - endpoints[3]);
            const bool upper = d1 <= d0;
            weights[index] = upper ? 1u : 0u;
            auto & sum = upper ? hi_sum : lo_sum;
            auto & count = upper ? hi_count : lo_count;
            sum[0] += l; sum[1] += a; ++count;
        }
        if (lo_count) low = {float((lo_sum[0] + lo_count / 2u) / lo_count) / 255.0f,
                             float((lo_sum[1] + lo_count / 2u) / lo_count) / 255.0f};
        if (hi_count) high = {float((hi_sum[0] + hi_count / 2u) / hi_count) / 255.0f,
                              float((hi_sum[1] + hi_count / 2u) / hi_count) / 255.0f};
    }
    endpoints = {quantize_unorm8(low[0]), quantize_unorm8(high[0]),
                 quantize_unorm8(low[1]), quantize_unorm8(high[1])};
    for (uint32_t index = 0; index < 40; ++index) {
        const auto & texel = source.texels[index];
        const int l = quantize_unorm8(texel.rgba[0]), a = quantize_unorm8(texel.rgba[3]);
        const int d0 = (l - endpoints[0]) * (l - endpoints[0]) + (a - endpoints[2]) * (a - endpoints[2]);
        const int d1 = (l - endpoints[1]) * (l - endpoints[1]) + (a - endpoints[3]) * (a - endpoints[3]);
        weights[index] = d1 <= d0 ? 1u : 0u;
    }
}

std::array<uint8_t, 16> pack_luminance_raw_weights(
    uint32_t block_mode, uint32_t endpoint_format,
    const uint8_t * endpoints, uint32_t endpoint_count,
    const uint8_t * weights, uint32_t weight_count, uint32_t bits_per_weight) {
    std::array<uint8_t, 16> weight_bits{};
    const uint8_t limit = static_cast<uint8_t>((1u << bits_per_weight) - 1u);
    for (uint32_t index = 0; index < weight_count; ++index) {
        if (weights[index] > limit) return {};
        const uint32_t offset = index * bits_per_weight;
        weight_bits[offset >> 3u] |= static_cast<uint8_t>(weights[index] << (offset & 7u));
    }
    std::array<uint8_t, 16> payload{};
    for (uint32_t index = 0; index < payload.size(); ++index) {
        payload[index] = bit_reverse(weight_bits[payload.size() - 1u - index]);
    }
    write_bits(block_mode, 11, 0, payload);
    write_bits(0u, 2, 11, payload); // One partition.
    write_bits(endpoint_format, 4, 13, payload);
    for (uint32_t index = 0; index < endpoint_count; ++index) {
        write_bits(endpoints[index], 8, 17u + index * 8u, payload);
    }
    return payload;
}

} // namespace

const astc_vulkan_astc_mode_descriptor * astc_gpu_exact_subset_audited_mode(
    astc_gpu_exact_subset_kind kind) {
    using gpu_kind = astc_gpu_exact_subset_kind;
    using audited = astc_vulkan_audited_mode;
    switch (kind) {
        case gpu_kind::d1_luminance_binary_6x6:
        case gpu_kind::d1_luminance_binary_refined_6x6:
        case gpu_kind::d1_luminance_binary_mean_refined_6x6:
        case gpu_kind::d1_luminance_binary_quantile_refined_6x6:
            return astc_vulkan_find_audited_mode(audited::d1_luminance_binary_6x6);
        case gpu_kind::d1_luminance_binary_5x5:
            return astc_vulkan_find_audited_mode(audited::d1_luminance_binary_5x5);
        case gpu_kind::d1_luminance_quant4_4x4:
            return astc_vulkan_find_audited_mode(audited::d1_luminance_quant4_4x4);
        case gpu_kind::luminance_alpha_binary_8x5:
        case gpu_kind::luminance_alpha_binary_luminance_weights_8x5:
        case gpu_kind::luminance_alpha_binary_alpha_weights_8x5:
        case gpu_kind::luminance_alpha_binary_refined_8x5:
        case gpu_kind::luminance_alpha_binary_mean_refined_8x5:
        case gpu_kind::luminance_alpha_binary_quantile_refined_8x5:
            return astc_vulkan_find_audited_mode(audited::d2_luminance_alpha_binary_8x5);
        case gpu_kind::luminance_alpha_dual_binary_8x5:
            return astc_vulkan_find_audited_mode(audited::d2_luminance_alpha_dual_binary_8x5);
        case gpu_kind::void_extent_unorm16:
            return nullptr;
    }
    return nullptr;
}

bool astc_gpu_exact_subset_matches_audited_mode(
    const astc_gpu_encoder_request & request) {
    if (request.exact_subset == astc_gpu_exact_subset_kind::void_extent_unorm16) {
        return astc_vulkan_footprint_is_valid(request.footprint);
    }
    const auto * descriptor = astc_gpu_exact_subset_audited_mode(request.exact_subset);
    return descriptor != nullptr && descriptor->footprint == request.footprint &&
        astc_vulkan_audit_mode_budget(*descriptor).legal;
}

std::array<uint8_t, 16> astc_gpu_exact_subset_pack_void_extent_unorm16(
    const std::array<uint16_t, 4> & rgba) {
    // ASTC void-extent UNORM16 header, followed by four little-endian values.
    std::array<uint8_t, 16> payload{0xFC, 0xFD, 0xFF, 0xFF,
                                    0xFF, 0xFF, 0xFF, 0xFF};
    for (uint32_t channel = 0; channel < 4; ++channel) {
        payload[8 + 2 * channel] = static_cast<uint8_t>(rgba[channel] & 0xFFu);
        payload[9 + 2 * channel] = static_cast<uint8_t>(rgba[channel] >> 8u);
    }
    return payload;
}

std::array<uint8_t, 16> astc_gpu_exact_subset_pack_d1_luminance_binary_6x6(
    uint8_t endpoint_low, uint8_t endpoint_high,
    const std::array<uint8_t, 36> & weights) {
    // ASTC normal block mode 0x104: 6x6 grid, QUANT_2 weights, one plane.
    return pack_luminance_raw_weights(0x104u, endpoint_low, endpoint_high,
                                      weights.data(), weights.size(), 1);
}

std::array<uint8_t, 16> astc_gpu_exact_subset_pack_d1_luminance_binary_5x5(
    uint8_t endpoint_low, uint8_t endpoint_high,
    const std::array<uint8_t, 25> & weights) {
    // ASTC normal block mode 0x0e1: 5x5 grid, QUANT_2 weights, one plane.
    return pack_luminance_raw_weights(0x0e1u, endpoint_low, endpoint_high,
                                      weights.data(), weights.size(), 1);
}

std::array<uint8_t, 16> astc_gpu_exact_subset_pack_d1_luminance_quant4_4x4(
    uint8_t endpoint_low, uint8_t endpoint_high,
    const std::array<uint8_t, 16> & weights) {
    // ASTC normal block mode 0x042: 4x4 grid, QUANT_4 weights, one plane.
    return pack_luminance_raw_weights(0x042u, endpoint_low, endpoint_high,
                                      weights.data(), weights.size(), 2);
}

std::array<uint8_t, 16> astc_gpu_exact_subset_pack_luminance_alpha_binary_8x5(
    const std::array<uint8_t, 4> & endpoints,
    const std::array<uint8_t, 40> & weights) {
    // ASTC normal block mode 0x065: 8x5 grid, QUANT_2 weights, one plane.
    // FMT_LUMINANCE_ALPHA provides L0, L1, A0, A1 direct endpoints.
    return pack_luminance_raw_weights(0x065u, 4u, endpoints.data(), endpoints.size(),
                                      weights.data(), weights.size(), 1);
}

std::array<uint8_t, 16> astc_gpu_exact_subset_pack_luminance_alpha_dual_binary_8x5(
    const std::array<uint8_t, 4> & endpoints,
    const std::array<uint8_t, 40> & interleaved_weights) {
    // Mode 0x4c1: 5x4 QUANT_2 grid, two planes. 40 binary values are stored
    // as L/A pairs; plane two's component selector is Alpha (3).
    auto payload = pack_luminance_raw_weights(0x4c1u, 4u, endpoints.data(), endpoints.size(),
                                              interleaved_weights.data(), interleaved_weights.size(), 1);
    write_bits(3u, 2, 86u, payload);
    return payload;
}

bool astc_gpu_exact_subset_encode_cpu(
    const astc_gpu_encoder_request & request,
    std::vector<astc_gpu_exact_subset_block> & blocks) {
    std::vector<astc_gpu_encoder_batch> batches;
    if (!astc_gpu_exact_subset_matches_audited_mode(request) ||
        !astc_gpu_encoder_plan_batches(request, batches)) return false;

    const bool void_extent = request.exact_subset == astc_gpu_exact_subset_kind::void_extent_unorm16;
    const bool binary_6x6 = request.exact_subset == astc_gpu_exact_subset_kind::d1_luminance_binary_6x6 &&
        request.footprint == astc_vulkan_footprint::k6x6;
    const bool refined_binary_6x6 =
        (request.exact_subset == astc_gpu_exact_subset_kind::d1_luminance_binary_refined_6x6 ||
         request.exact_subset == astc_gpu_exact_subset_kind::d1_luminance_binary_mean_refined_6x6 ||
         request.exact_subset == astc_gpu_exact_subset_kind::d1_luminance_binary_quantile_refined_6x6) &&
        request.footprint == astc_vulkan_footprint::k6x6;
    const bool binary_5x5 = request.exact_subset == astc_gpu_exact_subset_kind::d1_luminance_binary_5x5 &&
        request.footprint == astc_vulkan_footprint::k5x5;
    const bool quant4_4x4 = request.exact_subset == astc_gpu_exact_subset_kind::d1_luminance_quant4_4x4 &&
        request.footprint == astc_vulkan_footprint::k4x4;
    const bool luminance_alpha_balanced_8x5 = request.exact_subset == astc_gpu_exact_subset_kind::luminance_alpha_binary_8x5 &&
        request.footprint == astc_vulkan_footprint::k8x5;
    const bool luminance_alpha_luminance_8x5 = request.exact_subset == astc_gpu_exact_subset_kind::luminance_alpha_binary_luminance_weights_8x5 &&
        request.footprint == astc_vulkan_footprint::k8x5;
    const bool luminance_alpha_alpha_8x5 = request.exact_subset == astc_gpu_exact_subset_kind::luminance_alpha_binary_alpha_weights_8x5 &&
        request.footprint == astc_vulkan_footprint::k8x5;
    const bool refined_luminance_alpha_8x5 =
        (request.exact_subset == astc_gpu_exact_subset_kind::luminance_alpha_binary_refined_8x5 ||
         request.exact_subset == astc_gpu_exact_subset_kind::luminance_alpha_binary_mean_refined_8x5 ||
         request.exact_subset == astc_gpu_exact_subset_kind::luminance_alpha_binary_quantile_refined_8x5) &&
        request.footprint == astc_vulkan_footprint::k8x5;
    const bool luminance_alpha_dual_8x5 = request.exact_subset == astc_gpu_exact_subset_kind::luminance_alpha_dual_binary_8x5 &&
        request.footprint == astc_vulkan_footprint::k8x5;
    const bool luminance_alpha_8x5 = luminance_alpha_balanced_8x5 ||
        luminance_alpha_luminance_8x5 || luminance_alpha_alpha_8x5 || refined_luminance_alpha_8x5;
    if (!void_extent && !binary_6x6 && !refined_binary_6x6 && !binary_5x5 && !quant4_4x4 && !luminance_alpha_8x5 && !luminance_alpha_dual_8x5) return false;
    const bool scalar_luminance = binary_6x6 || refined_binary_6x6 || binary_5x5 || quant4_4x4;
    blocks.clear();
    blocks.reserve(request.blocks.size());
    for (const auto & source : request.blocks) {
        std::array<double, 4> sums{};
        float low = 1.0f;
        float high = 0.0f;
        float alpha_low = 1.0f;
        float alpha_high = 0.0f;
        for (const auto & texel : source.texels) {
            for (uint32_t channel = 0; channel < 4; ++channel) {
                const float value = texel.rgba[channel];
                if (!std::isfinite(value) || value < 0.0f || value > 1.0f) return false;
                sums[channel] += value;
            }
            if (scalar_luminance && (texel.rgba[0] != texel.rgba[1] || texel.rgba[0] != texel.rgba[2])) return false;
            low = std::min(low, texel.rgba[0]);
            high = std::max(high, texel.rgba[0]);
            alpha_low = std::min(alpha_low, texel.rgba[3]);
            alpha_high = std::max(alpha_high, texel.rgba[3]);
        }
        astc_gpu_exact_subset_block result;
        result.source_block_id = source.source_block_id;
        result.mode = request.exact_subset;
        if (void_extent) {
            for (uint32_t channel = 0; channel < 4; ++channel) {
                result.unorm16_rgba[channel] = unorm16(
                    static_cast<float>(sums[channel] / source.texels.size()));
            }
            result.payload = astc_gpu_exact_subset_pack_void_extent_unorm16(result.unorm16_rgba);
        } else {
            uint8_t endpoint_low = quantize_unorm8(low);
            uint8_t endpoint_high = quantize_unorm8(high);
            const float threshold = 0.5f * (low + high);
            std::array<uint8_t, 40> weights{};
            const float range = high - low;
            const float alpha_range = alpha_high - alpha_low;
            for (uint32_t index = 0; index < source.texels.size(); ++index) {
                const float normalized = range > 0.0f ? (source.texels[index].rgba[0] - low) / range : 0.0f;
                const float alpha_normalized = alpha_range > 0.0f
                    ? (source.texels[index].rgba[3] - alpha_low) / alpha_range : 0.0f;
                weights[index] = quant4_4x4
                    ? static_cast<uint8_t>(std::lround(std::clamp(normalized, 0.0f, 1.0f) * 3.0f))
                    : (luminance_alpha_8x5
                        ? ((luminance_alpha_luminance_8x5 ? normalized :
                            (luminance_alpha_alpha_8x5 ? alpha_normalized :
                             0.5f * (normalized + alpha_normalized))) >= 0.5f ? 1u : 0u)
                        : (source.texels[index].rgba[0] >= threshold ? 1u : 0u));
            }
            if (refined_binary_6x6) {
                std::array<uint8_t, 36> refined_weights{};
                const auto seed = request.exact_subset == astc_gpu_exact_subset_kind::d1_luminance_binary_mean_refined_6x6
                    ? binary_refinement_seed::mean_split
                    : (request.exact_subset == astc_gpu_exact_subset_kind::d1_luminance_binary_quantile_refined_6x6
                        ? binary_refinement_seed::quartiles : binary_refinement_seed::minmax);
                refine_binary_luminance_6x6(source, endpoint_low, endpoint_high, refined_weights, seed);
                std::copy(refined_weights.begin(), refined_weights.end(), weights.begin());
            }
            result.unorm16_rgba[0] = endpoint_low;
            result.unorm16_rgba[1] = endpoint_high;
            if (luminance_alpha_8x5 || luminance_alpha_dual_8x5) {
                std::array<uint8_t, 4> endpoints{
                    endpoint_low, endpoint_high,
                    static_cast<uint8_t>(std::lround(alpha_low * 255.0f)),
                    static_cast<uint8_t>(std::lround(alpha_high * 255.0f))};
                if (luminance_alpha_dual_8x5) {
                    std::array<uint8_t, 40> dual_weights{};
                    for (uint32_t y = 0; y < 4; ++y) for (uint32_t x = 0; x < 5; ++x) {
                        const uint32_t source_x = (x * 7u + 2u) / 4u;
                        const uint32_t source_y = (y * 4u + 1u) / 3u;
                        const auto & texel = source.texels[source_y * 8u + source_x];
                        const float l = range > 0.0f ? (texel.rgba[0] - low) / range : 0.0f;
                        const float a = alpha_range > 0.0f ? (texel.rgba[3] - alpha_low) / alpha_range : 0.0f;
                        const uint32_t grid_index = y * 5u + x;
                        dual_weights[2u * grid_index] = l >= 0.5f ? 1u : 0u;
                        dual_weights[2u * grid_index + 1u] = a >= 0.5f ? 1u : 0u;
                    }
                    result.payload = astc_gpu_exact_subset_pack_luminance_alpha_dual_binary_8x5(
                        endpoints, dual_weights);
                } else {
                    if (refined_luminance_alpha_8x5) {
                        const auto seed = request.exact_subset == astc_gpu_exact_subset_kind::luminance_alpha_binary_mean_refined_8x5
                            ? binary_refinement_seed::mean_split
                            : (request.exact_subset == astc_gpu_exact_subset_kind::luminance_alpha_binary_quantile_refined_8x5
                                ? binary_refinement_seed::quartiles : binary_refinement_seed::minmax);
                        refine_luminance_alpha_binary_8x5(source, endpoints, weights, seed);
                    }
                    result.payload = astc_gpu_exact_subset_pack_luminance_alpha_binary_8x5(
                        endpoints, weights);
                }
            } else if (binary_6x6 || refined_binary_6x6) {
                std::array<uint8_t, 36> weights_6x6{};
                std::copy_n(weights.begin(), weights_6x6.size(), weights_6x6.begin());
                result.payload = astc_gpu_exact_subset_pack_d1_luminance_binary_6x6(
                    endpoint_low, endpoint_high, weights_6x6);
            } else if (binary_5x5) {
                std::array<uint8_t, 25> weights_5x5{};
                std::copy_n(weights.begin(), weights_5x5.size(), weights_5x5.begin());
                result.payload = astc_gpu_exact_subset_pack_d1_luminance_binary_5x5(
                    endpoint_low, endpoint_high, weights_5x5);
            } else {
                std::array<uint8_t, 16> weights_4x4{};
                std::copy_n(weights.begin(), weights_4x4.size(), weights_4x4.begin());
                result.payload = astc_gpu_exact_subset_pack_d1_luminance_quant4_4x4(
                    endpoint_low, endpoint_high, weights_4x4);
            }
        }
        blocks.push_back(result);
    }
    return true;
}
