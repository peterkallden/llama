#include <astcenc.h>

#include "astc-vulkan-input.h"
#include "astc-vulkan-paired.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

namespace {

struct decoded_image { uint32_t width = 0, height = 0; std::vector<float> values; };

bool roundtrip(const std::vector<float> & source, uint32_t width, uint32_t height,
               uint32_t block_width, uint32_t block_height, decoded_image & decoded) {
    astcenc_config config{};
    if (astcenc_config_init(ASTCENC_PRF_LDR, block_width, block_height, 1,
                            ASTCENC_PRE_THOROUGH, 0, &config) != ASTCENC_SUCCESS) return false;
    astcenc_context * context = nullptr;
    if (astcenc_context_alloc(&config, 1, &context) != ASTCENC_SUCCESS) return false;
    void * source_slice = const_cast<float *>(source.data());
    astcenc_image source_image{width, height, 1, ASTCENC_TYPE_F32, &source_slice};
    const astcenc_swizzle swizzle{ASTCENC_SWZ_R, ASTCENC_SWZ_G,
                                  ASTCENC_SWZ_B, ASTCENC_SWZ_A};
    const size_t bytes = static_cast<size_t>((width + block_width - 1) / block_width) *
                         ((height + block_height - 1) / block_height) * 16;
    std::vector<uint8_t> payload(bytes);
    bool ok = astcenc_compress_image(context, &source_image, &swizzle,
                                     payload.data(), payload.size(), 0) == ASTCENC_SUCCESS;
    decoded = {width, height, std::vector<float>(static_cast<size_t>(width) * height * 4)};
    void * decoded_slice = decoded.values.data();
    astcenc_image decoded_image{width, height, 1, ASTCENC_TYPE_F32, &decoded_slice};
    if (ok) ok = astcenc_decompress_image(context, payload.data(), payload.size(),
                                          &decoded_image, &swizzle, 0) == ASTCENC_SUCCESS;
    astcenc_context_free(context);
    return ok;
}

float source_q(const ggml_vk_astc_loaded_matrix & matrix, uint32_t row, uint32_t column,
               float minimum, float range) {
    return std::clamp((matrix.values[static_cast<size_t>(row) * matrix.columns + column] - minimum) /
                      (range > 0.0f ? range : 1.0f), 0.0f, 1.0f);
}

double loss(const ggml_vk_astc_loaded_matrix & matrix, const ggml_vk_astc_activation_trace & trace,
            const decoded_image & decoded, uint32_t rows, uint32_t columns,
            float minimum, float range, bool d2, astc_vulkan_paired_layout layout) {
    const float safe_range = range > 0.0f ? range : 1.0f;
    double result = 0.0;
    for (uint32_t sample = 0; sample < trace.samples; ++sample) {
        for (uint32_t row = 0; row < rows; ++row) {
            double output_error = 0.0;
            for (uint32_t column = 0; column < columns; ++column) {
                const uint32_t texel_row = d2 ? row / 2 : row;
                const size_t offset = (static_cast<size_t>(texel_row) * columns + column) * 4;
                const astc_vulkan_rgba_texel texel{decoded.values[offset], decoded.values[offset + 1],
                                                   decoded.values[offset + 2], decoded.values[offset + 3]};
                const float decoded_q = d2 ? astc_vulkan_paired_weight(texel, row & 1u, layout) :
                    (texel.r + texel.g + texel.b) / 3.0f;
                const float error = (source_q(matrix, row, column, minimum, range) - decoded_q) * safe_range;
                output_error += static_cast<double>(error) *
                    trace.values[static_cast<size_t>(sample) * columns + column];
            }
            result += output_error * output_error;
        }
    }
    return result / static_cast<double>(trace.samples * rows);
}

bool run_d1(const ggml_vk_astc_loaded_matrix & matrix, const ggml_vk_astc_activation_trace & trace,
            uint32_t rows, uint32_t columns, float minimum, float range, double & result) {
    std::vector<float> source(static_cast<size_t>(rows) * columns * 4);
    for (uint32_t row = 0; row < rows; ++row) for (uint32_t column = 0; column < columns; ++column) {
        const float q = source_q(matrix, row, column, minimum, range);
        const size_t offset = (static_cast<size_t>(row) * columns + column) * 4;
        source[offset] = source[offset + 1] = source[offset + 2] = source[offset + 3] = q;
    }
    decoded_image decoded;
    if (!roundtrip(source, columns, rows, 10, 8, decoded)) return false;
    result = loss(matrix, trace, decoded, rows, columns, minimum, range, false,
                  astc_vulkan_paired_layout::rg_b);
    return true;
}

bool run_d2(const ggml_vk_astc_loaded_matrix & matrix, const ggml_vk_astc_activation_trace & trace,
            uint32_t rows, uint32_t columns, float minimum, float range,
            astc_vulkan_paired_layout layout, bool steering, double & result) {
    const uint32_t texture_rows = (rows + 1) / 2;
    std::vector<float> source(static_cast<size_t>(texture_rows) * columns * 4);
    for (uint32_t row = 0; row < texture_rows; ++row) for (uint32_t column = 0; column < columns; ++column) {
        const uint32_t row0 = row * 2;
        const uint32_t row1 = std::min(row0 + 1, rows - 1);
        const float alpha = steering ? 0.5f + 0.25f * std::sin(0.13f * column + 0.17f * row) : 0.5f;
        const auto texel = astc_vulkan_make_paired_texel(
            source_q(matrix, row0, column, minimum, range),
            source_q(matrix, row1, column, minimum, range), alpha, layout);
        const size_t offset = (static_cast<size_t>(row) * columns + column) * 4;
        source[offset] = texel.r; source[offset + 1] = texel.g;
        source[offset + 2] = texel.b; source[offset + 3] = texel.a;
    }
    decoded_image decoded;
    if (!roundtrip(source, columns, texture_rows, 8, 5, decoded)) return false;
    result = loss(matrix, trace, decoded, rows, columns, minimum, range, true, layout);
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 7 || std::string(argv[1]) != "--model" ||
        std::string(argv[3]) != "--tensor" || std::string(argv[5]) != "--trace") {
        std::fprintf(stderr, "usage: %s --model model.gguf --tensor name --trace trace\n", argv[0]);
        return 2;
    }
    ggml_vk_astc_loaded_matrix matrix;
    ggml_vk_astc_activation_trace trace;
    std::string error;
    if (!ggml_vk_astc_load_gguf_matrix(argv[2], argv[4], matrix, error) ||
        !ggml_vk_astc_load_activation_trace(argv[6], trace, error) ||
        trace.columns == 0 || trace.columns > matrix.columns || matrix.rows < 2) {
        std::fprintf(stderr, "paired model smoke input error: %s\n", error.c_str());
        return 1;
    }
    const uint32_t rows = std::min<uint32_t>(matrix.rows, 32);
    const uint32_t columns = trace.columns;
    float minimum = std::numeric_limits<float>::infinity();
    float maximum = -std::numeric_limits<float>::infinity();
    for (uint32_t row = 0; row < rows; ++row) for (uint32_t column = 0; column < columns; ++column) {
        const float value = matrix.values[static_cast<size_t>(row) * matrix.columns + column];
        minimum = std::min(minimum, value); maximum = std::max(maximum, value);
    }
    const float range = maximum - minimum;
    double d1 = 0.0;
    if (!run_d1(matrix, trace, rows, columns, minimum, range, d1)) return 1;
    std::printf("paired-model D1 10x8 rate=1.60000 activation-mse=%.8g\n", d1);
    for (const auto layout : {astc_vulkan_paired_layout::rg_b, astc_vulkan_paired_layout::r_gb}) {
        for (const bool steering : {false, true}) {
            double value = 0.0;
            if (!run_d2(matrix, trace, rows, columns, minimum, range, layout, steering, value)) return 1;
            std::printf("paired-model D2 8x5 layout=%s steering=%s rate=1.60000 activation-mse=%.8g\n",
                        astc_vulkan_paired_layout_name(layout), steering ? "yes" : "no", value);
        }
    }
    return 0;
}
