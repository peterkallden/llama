#include <astcenc.h>

#include "astc-vulkan-contract.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

enum class pattern_kind {
    smooth,
    clustered,
    pseudo_random,
    outlier,
};

struct pattern_descriptor {
    const char * name;
    pattern_kind kind;
};

constexpr std::array<pattern_descriptor, 4> kPatterns{{
    { "smooth", pattern_kind::smooth },
    { "clustered", pattern_kind::clustered },
    { "random", pattern_kind::pseudo_random },
    { "outlier", pattern_kind::outlier },
}};

uint32_t level_for(uint32_t x, uint32_t y, uint32_t width, uint32_t height,
                   uint32_t levels, pattern_kind pattern) {
    switch (pattern) {
        case pattern_kind::smooth:
            return static_cast<uint32_t>(std::lround(
                (levels - 1) * static_cast<double>(x + y) / (width + height - 2)));
        case pattern_kind::clustered:
            return ((x / 3) + (y / 3) * 2) % levels;
        case pattern_kind::pseudo_random: {
            uint32_t state = 0x9e3779b9u ^ (x * 0x85ebca6bu) ^ (y * 0xc2b2ae35u);
            state ^= state >> 16;
            state *= 0x7feb352du;
            return (state ^ (state >> 15)) % levels;
        }
        case pattern_kind::outlier:
            return (x == width / 2 && y == height / 2) ||
                (x == width / 2 - 1 && y == height / 2) ? levels - 1 : levels / 2;
    }
    return 0;
}

bool run_case(uint32_t block, uint32_t levels, const pattern_descriptor & pattern) {
    const uint32_t width = block * 4;
    const uint32_t height = block * 4;
    const size_t texel_count = static_cast<size_t>(width) * height;
    std::vector<float> expected(texel_count);
    std::vector<float> input(texel_count * 4);
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            const uint32_t level = level_for(x, y, width, height, levels, pattern.kind);
            const float value = static_cast<float>(level) / (levels - 1);
            const size_t texel = static_cast<size_t>(y) * width + x;
            expected[texel] = value;
            // Replicate the scalar in RGBA. This intentionally measures one
            // scalar per texel; it does not claim four independent values per
            // texel at the same bits/value rate.
            std::fill_n(input.data() + texel * 4, 4, value);
        }
    }

    astcenc_config config{};
    if (astcenc_config_init(ASTCENC_PRF_LDR, block, block, 1,
                            ASTCENC_PRE_THOROUGH, 0, &config) != ASTCENC_SUCCESS) {
        return false;
    }
    astcenc_context * context = nullptr;
    if (astcenc_context_alloc(&config, 1, &context) != ASTCENC_SUCCESS) {
        return false;
    }
    void * input_slice = input.data();
    astcenc_image input_image{ width, height, 1, ASTCENC_TYPE_F32, &input_slice };
    const astcenc_swizzle swizzle{
        ASTCENC_SWZ_R, ASTCENC_SWZ_G, ASTCENC_SWZ_B, ASTCENC_SWZ_A,
    };
    const size_t compressed_bytes = static_cast<size_t>(
        ggml_vk_astc_block_count(width, block) * ggml_vk_astc_block_count(height, block) * 16);
    std::vector<uint8_t> compressed(compressed_bytes);
    astcenc_error status = astcenc_compress_image(
        context, &input_image, &swizzle, compressed.data(), compressed.size(), 0);
    if (status != ASTCENC_SUCCESS) {
        astcenc_context_free(context);
        return false;
    }
    std::vector<float> decoded(input.size());
    void * decoded_slice = decoded.data();
    astcenc_image decoded_image{ width, height, 1, ASTCENC_TYPE_F32, &decoded_slice };
    status = astcenc_decompress_image(
        context, compressed.data(), compressed.size(), &decoded_image, &swizzle, 0);
    astcenc_context_free(context);
    if (status != ASTCENC_SUCCESS) {
        return false;
    }

    double squared_error = 0.0;
    float max_error = 0.0f;
    uint32_t near_levels = 0;
    for (size_t texel = 0; texel < texel_count; ++texel) {
        const float value = decoded[texel * 4];
        const float error = std::fabs(expected[texel] - value);
        squared_error += static_cast<double>(error) * error;
        max_error = std::max(max_error, error);
        const float nearest_level = std::round(value * (levels - 1)) / (levels - 1);
        if (std::fabs(value - nearest_level) <= 0.02f) {
            ++near_levels;
        }
    }
    const double mse = squared_error / texel_count;
    std::printf("ASTC-Q %ux%u levels=%u pattern=%s bytes=%zu bits-per-weight=%.5f "
                "near-level=%.2f%% MSE=%.8g max-error=%.6f\n",
                block, block, levels, pattern.name, compressed_bytes,
                compressed_bytes * 8.0 / texel_count,
                100.0 * near_levels / texel_count, mse, max_error);
    return std::isfinite(mse) && std::isfinite(max_error);
}

} // namespace

int main() {
    bool ok = true;
    for (const uint32_t block : { 4u, 6u }) {
        for (const uint32_t levels : { 3u, 5u, 8u, 16u }) {
            for (const pattern_descriptor & pattern : kPatterns) {
                ok = run_case(block, levels, pattern) && ok;
            }
        }
    }
    if (!ok) {
        std::fprintf(stderr, "ASTC-Q level smoke failed\n");
        return 1;
    }
    std::printf("ASTC-Q level smoke passed\n");
    return 0;
}
