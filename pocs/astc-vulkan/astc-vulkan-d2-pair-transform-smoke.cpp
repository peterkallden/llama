// Exact offline oracle for D2-LA pairing and inexpensive Givens transforms.
//
// This intentionally stops before cache serialization or Vulkan dispatch. It
// asks one narrow question: can a pairing and a 2x2 transform produce a more
// useful legal ASTC 8x5 candidate than adjacent direct D2-LA pairing?

#include "astc-vulkan-d2-pair-transform.h"
#include "astc-vulkan-input.h"
#include "astc-vulkan-paired.h"

#include <astcenc.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr unsigned int kWidth = 8;
constexpr unsigned int kPhysicalHeight = 5;
constexpr unsigned int kLogicalHeight = 10;
constexpr unsigned int kSamples = 8;
constexpr float kPi = 3.14159265358979323846f;

struct candidate {
    astc_vulkan_d2_pairing pairing{};
    astc_vulkan_d2_givens_transform transform{};
    float screen_score = 0.0f;
    float activation_mse = std::numeric_limits<float>::infinity();
    bool legal = false;
};

float clamp01(float value) {
    return std::max(0.0f, std::min(1.0f, value));
}

float dot(const float * lhs, const float * rhs, size_t size) {
    float total = 0.0f;
    for (size_t index = 0; index < size; ++index) total += lhs[index] * rhs[index];
    return total;
}

// This fixture deliberately has useful non-adjacent pairs. It is not intended
// as a quality claim for a model; only to exercise the exact candidate chain.
std::array<float, kLogicalHeight * kWidth> make_weights() {
    std::array<float, kLogicalHeight * kWidth> weights{};
    constexpr std::array<unsigned int, kLogicalHeight> mate = {7, 9, 5, 8, 6, 2, 4, 0, 3, 1};
    for (unsigned int row = 0; row < kLogicalHeight; ++row) {
        const unsigned int base = std::min(row, mate[row]);
        for (unsigned int column = 0; column < kWidth; ++column) {
            const float x = static_cast<float>(column) / static_cast<float>(kWidth - 1);
            const float shared = std::sin((base + 1.0f) * (0.7f + 2.1f * x));
            const float detail = 0.08f * std::cos((row + 1.0f) * (1.3f + 3.2f * x));
            weights[row * kWidth + column] = 0.52f + 0.32f * shared + detail;
        }
    }
    return weights;
}

std::array<float, kSamples * kWidth> make_activations() {
    std::array<float, kSamples * kWidth> activations{};
    for (unsigned int sample = 0; sample < kSamples; ++sample) {
        for (unsigned int column = 0; column < kWidth; ++column) {
            activations[sample * kWidth + column] =
                std::sin((sample + 1.0f) * (column + 1.0f) * 0.31f) +
                0.35f * std::cos((sample + 2.0f) * (column + 1.0f) * 0.17f);
        }
    }
    return activations;
}

float pair_screen_score(const std::array<float, kLogicalHeight * kWidth> & weights,
                        const astc_vulkan_d2_pairing & pairing) {
    float score = 0.0f;
    for (unsigned int pair = 0; pair < kPhysicalHeight; ++pair) {
        const float * first = weights.data() + pairing.row_order[2 * pair] * kWidth;
        const float * second = weights.data() + pairing.row_order[2 * pair + 1] * kWidth;
        const float numerator = std::fabs(dot(first, second, kWidth));
        const float denom = std::sqrt(dot(first, first, kWidth) * dot(second, second, kWidth)) + 1e-8f;
        score += numerator / denom;
    }
    return score;
}

bool exact_candidate(const std::array<float, kLogicalHeight * kWidth> & weights,
                     const std::array<float, kSamples * kWidth> & activations,
                     candidate & value) {
    std::array<float, kWidth * kPhysicalHeight> first{};
    std::array<float, kWidth * kPhysicalHeight> second{};
    float low = std::numeric_limits<float>::infinity();
    float high = -std::numeric_limits<float>::infinity();
    for (unsigned int y = 0; y < kPhysicalHeight; ++y) {
        const unsigned int row0 = value.pairing.row_order[2 * y];
        const unsigned int row1 = value.pairing.row_order[2 * y + 1];
        for (unsigned int x = 0; x < kWidth; ++x) {
            float u = 0.0f;
            float v = 0.0f;
            astc_vulkan_d2_pair_forward(weights[row0 * kWidth + x], weights[row1 * kWidth + x],
                                        value.transform, u, v);
            first[y * kWidth + x] = u;
            second[y * kWidth + x] = v;
            low = std::min(low, std::min(u, v));
            high = std::max(high, std::max(u, v));
        }
    }
    const float range = std::max(1e-6f, high - low);
    std::array<float, kWidth * kPhysicalHeight * 4> source{};
    for (unsigned int y = 0; y < kPhysicalHeight; ++y) {
        for (unsigned int x = 0; x < kWidth; ++x) {
            const size_t pixel = (y * kWidth + x) * 4;
            const astc_vulkan_rgba_texel texel = astc_vulkan_make_paired_texel(
                clamp01((first[y * kWidth + x] - low) / range),
                clamp01((second[y * kWidth + x] - low) / range), 0.0f,
                astc_vulkan_paired_layout::rg_b, astc_vulkan_paired_basis::direct,
                astc_vulkan_paired_semantic::luminance_alpha);
            source[pixel + 0] = texel.r;
            source[pixel + 1] = texel.g;
            source[pixel + 2] = texel.b;
            source[pixel + 3] = texel.a;
        }
    }

    astcenc_config config{};
    if (astcenc_config_init(ASTCENC_PRF_LDR, kWidth, kPhysicalHeight, 1,
                            ASTCENC_PRE_FAST, 0, &config) != ASTCENC_SUCCESS) return false;
    config.cw_r_weight = 1.0f / 3.0f;
    config.cw_g_weight = 1.0f / 3.0f;
    config.cw_b_weight = 1.0f / 3.0f;
    config.cw_a_weight = 1.0f;
    astcenc_context * context = nullptr;
    if (astcenc_context_alloc(&config, 1, &context) != ASTCENC_SUCCESS) return false;
    std::array<uint8_t, 16> payload{};
    void * source_slice = source.data();
    astcenc_image image{kWidth, kPhysicalHeight, 1, ASTCENC_TYPE_F32, &source_slice};
    const astcenc_swizzle swizzle{ASTCENC_SWZ_R, ASTCENC_SWZ_G, ASTCENC_SWZ_B, ASTCENC_SWZ_A};
    const astcenc_error encoded = astcenc_compress_image(context, &image, &swizzle,
                                                         payload.data(), payload.size(), 0);
    astcenc_block_info info{};
    const astcenc_error inspected = astcenc_get_block_info(context, payload.data(), &info);
    std::array<float, kWidth * kPhysicalHeight * 4> decoded{};
    void * decoded_slice = decoded.data();
    astcenc_image decoded_image{kWidth, kPhysicalHeight, 1, ASTCENC_TYPE_F32, &decoded_slice};
    const astcenc_error decompressed = encoded == ASTCENC_SUCCESS && inspected == ASTCENC_SUCCESS
        ? astcenc_decompress_image(context, payload.data(), payload.size(), &decoded_image, &swizzle, 0)
        : ASTCENC_ERR_BAD_PARAM;
    astcenc_context_free(context);
    if (encoded != ASTCENC_SUCCESS || inspected != ASTCENC_SUCCESS || decompressed != ASTCENC_SUCCESS) return false;

    std::array<float, kLogicalHeight * kWidth> reconstructed{};
    for (unsigned int y = 0; y < kPhysicalHeight; ++y) {
        const unsigned int row0 = value.pairing.row_order[2 * y];
        const unsigned int row1 = value.pairing.row_order[2 * y + 1];
        for (unsigned int x = 0; x < kWidth; ++x) {
            const size_t pixel = (y * kWidth + x) * 4;
            const astc_vulkan_rgba_texel texel{decoded[pixel], decoded[pixel + 1], decoded[pixel + 2], decoded[pixel + 3]};
            const float u = low + range * astc_vulkan_paired_weight(texel, 0, astc_vulkan_paired_layout::rg_b,
                                                                        astc_vulkan_paired_basis::direct,
                                                                        astc_vulkan_paired_semantic::luminance_alpha);
            const float v = low + range * astc_vulkan_paired_weight(texel, 1, astc_vulkan_paired_layout::rg_b,
                                                                        astc_vulkan_paired_basis::direct,
                                                                        astc_vulkan_paired_semantic::luminance_alpha);
            astc_vulkan_d2_pair_inverse(u, v, value.transform,
                                        reconstructed[row0 * kWidth + x], reconstructed[row1 * kWidth + x]);
        }
    }
    float error = 0.0f;
    for (unsigned int row = 0; row < kLogicalHeight; ++row) {
        for (unsigned int sample = 0; sample < kSamples; ++sample) {
            float delta = 0.0f;
            for (unsigned int x = 0; x < kWidth; ++x) {
                delta += (weights[row * kWidth + x] - reconstructed[row * kWidth + x]) * activations[sample * kWidth + x];
            }
            error += delta * delta;
        }
    }
    value.activation_mse = error / static_cast<float>(kLogicalHeight * kSamples);
    value.legal = true;
    return true;
}

void print_candidate(const char * label, const candidate & value) {
    std::printf("%s mse=%.8g theta_deg=%.1f pairs=", label, value.activation_mse,
                value.transform.radians * 180.0f / kPi);
    for (unsigned int pair = 0; pair < kPhysicalHeight; ++pair) {
        std::printf("%s%u-%u", pair == 0 ? "" : ",", value.pairing.row_order[2 * pair], value.pairing.row_order[2 * pair + 1]);
    }
    std::printf("\n");
}

bool load_real_fixture(const char * model_path, const char * tensor_name, const char * trace_path,
                       std::array<float, kLogicalHeight * kWidth> & weights,
                       std::array<float, kSamples * kWidth> & activations) {
    ggml_vk_astc_loaded_matrix matrix;
    ggml_vk_astc_activation_trace trace;
    std::string error;
    if (!ggml_vk_astc_load_gguf_matrix(model_path, tensor_name, matrix, error) ||
        !ggml_vk_astc_load_activation_trace(trace_path, trace, error) ||
        matrix.rows < kLogicalHeight || matrix.columns < kWidth ||
        trace.samples < kSamples || trace.columns < kWidth) {
        std::fprintf(stderr, "d2-pair-transform real fixture error: %s\n", error.c_str());
        return false;
    }
    for (unsigned int row = 0; row < kLogicalHeight; ++row) {
        for (unsigned int column = 0; column < kWidth; ++column) {
            weights[row * kWidth + column] = matrix.values[static_cast<size_t>(row) * matrix.columns + column];
        }
    }
    for (unsigned int sample = 0; sample < kSamples; ++sample) {
        for (unsigned int column = 0; column < kWidth; ++column) {
            activations[sample * kWidth + column] = trace.values[static_cast<size_t>(sample) * trace.columns + column];
        }
    }
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    std::array<float, kLogicalHeight * kWidth> weights = make_weights();
    std::array<float, kSamples * kWidth> activations = make_activations();
    if (argc != 1 && argc != 7) {
        std::fprintf(stderr, "usage: %s [--model model.gguf --tensor tensor.name --trace input.trace]\n", argv[0]);
        return 2;
    }
    bool real_fixture = false;
    if (argc == 7) {
        if (std::string(argv[1]) != "--model" || std::string(argv[3]) != "--tensor" ||
            std::string(argv[5]) != "--trace" ||
            !load_real_fixture(argv[2], argv[4], argv[6], weights, activations)) return 2;
        real_fixture = true;
    }
    const auto identity = astc_vulkan_d2_identity_pairing();
    auto pairings = astc_vulkan_d2_enumerate_pairings();
    if (pairings.size() != 945 || !astc_vulkan_d2_pairing_is_valid(identity)) return 1;
    std::vector<candidate> screened;
    screened.reserve(pairings.size());
    for (const auto & pairing : pairings) screened.push_back({pairing, {}, pair_screen_score(weights, pairing)});
    std::sort(screened.begin(), screened.end(), [](const candidate & lhs, const candidate & rhs) { return lhs.screen_score > rhs.screen_score; });
    constexpr size_t shortlist_size = 16;
    screened.resize(shortlist_size);
    const bool identity_in_shortlist = std::any_of(screened.begin(), screened.end(), [&](const candidate & value) { return value.pairing.row_order == identity.row_order; });
    if (!identity_in_shortlist) screened.back().pairing = identity;

    candidate baseline{identity};
    candidate common_difference{identity, {kPi / 4.0f}};
    if (!exact_candidate(weights, activations, baseline) || !exact_candidate(weights, activations, common_difference)) return 2;
    candidate pairing_best = baseline;
    candidate givens_best = baseline;
    constexpr std::array<float, 7> angles = {-45.0f, -30.0f, -15.0f, 0.0f, 15.0f, 30.0f, 45.0f};
    unsigned int exact_count = 2;
    for (const auto & screened_value : screened) {
        candidate pairing_only{screened_value.pairing};
        if (!exact_candidate(weights, activations, pairing_only)) return 3;
        ++exact_count;
        if (pairing_only.activation_mse < pairing_best.activation_mse) pairing_best = pairing_only;
        for (const float degrees : angles) {
            candidate transformed{screened_value.pairing, {degrees * kPi / 180.0f}};
            if (!exact_candidate(weights, activations, transformed)) return 4;
            ++exact_count;
            if (transformed.activation_mse < givens_best.activation_mse) givens_best = transformed;
        }
    }
    std::printf("d2-pair-transform fixture=%s matchings=%zu shortlist=%zu exact-candidates=%u\n",
                real_fixture ? "real" : "synthetic", pairings.size(), screened.size(), exact_count);
    print_candidate("baseline-adjacent", baseline);
    print_candidate("fixed-common-difference", common_difference);
    print_candidate("pairing-only", pairing_best);
    print_candidate("pairing-plus-givens", givens_best);
    return 0;
}
