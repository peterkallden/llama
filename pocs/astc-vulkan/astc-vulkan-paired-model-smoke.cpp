#include <astcenc.h>

#include "astc-vulkan-input.h"
#include "astc-vulkan-objective.h"
#include "astc-vulkan-paired.h"
#include "astc-vulkan-yaqa.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

namespace {

#if defined(ASTC_VULKAN_PAIRED_D2_10X5)
constexpr uint32_t kD2BlockWidth = 10;
constexpr uint32_t kD2BlockHeight = 5;
constexpr const char * kD2FootprintName = "10x5";
constexpr double kD2Rate = 1.28;
#else
constexpr uint32_t kD2BlockWidth = 8;
constexpr uint32_t kD2BlockHeight = 5;
constexpr const char * kD2FootprintName = "8x5";
constexpr double kD2Rate = 1.60;
#endif

struct decoded_image { uint32_t width = 0, height = 0; std::vector<float> values; };

struct paired_model_params {
    std::string model;
    std::string tensor;
    std::string input_trace;
    std::string output_trace;
    astc_vulkan_objective objective = astc_vulkan_objective::activation;
};

bool parse_params(int argc, char ** argv, paired_model_params & params) {
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (index + 1 == argc) return false;
        const std::string value = argv[++index];
        if (option == "--model") params.model = value;
        else if (option == "--tensor") params.tensor = value;
        else if (option == "--trace") params.input_trace = value;
        else if (option == "--output-trace") params.output_trace = value;
        else if (option == "--objective") {
            if (value == "activation") params.objective = astc_vulkan_objective::activation;
            else if (value == "yaqa") params.objective = astc_vulkan_objective::two_sided_trace;
            else return false;
        } else return false;
    }
    return !params.model.empty() && !params.tensor.empty() && !params.input_trace.empty() &&
           (params.objective != astc_vulkan_objective::two_sided_trace || !params.output_trace.empty());
}

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

std::vector<float> decoded_error(const ggml_vk_astc_loaded_matrix & matrix,
                                 const decoded_image & decoded, uint32_t rows, uint32_t columns,
                                 float minimum, float range, bool d2,
                                 astc_vulkan_paired_layout layout) {
    const float safe_range = range > 0.0f ? range : 1.0f;
    std::vector<float> error(static_cast<size_t>(rows) * columns);
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t column = 0; column < columns; ++column) {
            const uint32_t texel_row = d2 ? row / 2 : row;
            const size_t offset = (static_cast<size_t>(texel_row) * columns + column) * 4;
            const astc_vulkan_rgba_texel texel{decoded.values[offset], decoded.values[offset + 1],
                                               decoded.values[offset + 2], decoded.values[offset + 3]};
            const float decoded_q = d2 ? astc_vulkan_paired_weight(texel, row & 1u, layout) :
                (texel.r + texel.g + texel.b) / 3.0f;
            error[static_cast<size_t>(row) * columns + column] =
                (source_q(matrix, row, column, minimum, range) - decoded_q) * safe_range;
        }
    }
    return error;
}

double objective_score(const std::vector<float> & error,
                       const ggml_vk_astc_activation_trace & input_trace,
                       const ggml_vk_astc_activation_trace * output_trace,
                       astc_vulkan_objective objective, uint32_t rows, uint32_t columns) {
    if (objective == astc_vulkan_objective::activation) {
        return astc_vulkan_activation_score(error, rows, columns,
                                            input_trace.values, input_trace.samples) /
               static_cast<double>(input_trace.samples * rows);
    }
    if (objective != astc_vulkan_objective::two_sided_trace || output_trace == nullptr) return NAN;
    std::vector<float> output_crop(static_cast<size_t>(output_trace->samples) * rows);
    for (uint32_t sample = 0; sample < output_trace->samples; ++sample) {
        std::copy_n(output_trace->values.data() + static_cast<size_t>(sample) * output_trace->columns,
                    rows, output_crop.data() + static_cast<size_t>(sample) * rows);
    }
    return astc_vulkan_yaqa_trace_score(error, rows, columns, input_trace.values, output_crop,
                                        input_trace.samples);
}

bool run_d1(const ggml_vk_astc_loaded_matrix & matrix, const ggml_vk_astc_activation_trace & trace,
            uint32_t rows, uint32_t columns, float minimum, float range,
            const ggml_vk_astc_activation_trace * output_trace,
            astc_vulkan_objective objective, double & result) {
    std::vector<float> source(static_cast<size_t>(rows) * columns * 4);
    for (uint32_t row = 0; row < rows; ++row) for (uint32_t column = 0; column < columns; ++column) {
        const float q = source_q(matrix, row, column, minimum, range);
        const size_t offset = (static_cast<size_t>(row) * columns + column) * 4;
        source[offset] = source[offset + 1] = source[offset + 2] = source[offset + 3] = q;
    }
    decoded_image decoded;
    if (!roundtrip(source, columns, rows, 10, 8, decoded)) return false;
    result = objective_score(decoded_error(matrix, decoded, rows, columns, minimum, range, false,
                                           astc_vulkan_paired_layout::rg_b),
                             trace, output_trace, objective, rows, columns);
    return true;
}

bool run_d2(const ggml_vk_astc_loaded_matrix & matrix, const ggml_vk_astc_activation_trace & trace,
            uint32_t rows, uint32_t columns, float minimum, float range,
            astc_vulkan_paired_layout layout,
            const astc_vulkan_paired_steering_factor & factor,
            const ggml_vk_astc_activation_trace * output_trace,
            astc_vulkan_objective objective, double & result) {
    const uint32_t texture_rows = (rows + 1) / 2;
    std::vector<float> source(static_cast<size_t>(texture_rows) * columns * 4);
    for (uint32_t row = 0; row < texture_rows; ++row) for (uint32_t column = 0; column < columns; ++column) {
        const uint32_t row0 = row * 2;
        const uint32_t row1 = std::min(row0 + 1, rows - 1);
        const float x = columns > 1 ? 2.0f * column / static_cast<float>(columns - 1) - 1.0f : 0.0f;
        const float y = texture_rows > 1 ? 2.0f * row / static_cast<float>(texture_rows - 1) - 1.0f : 0.0f;
        const float alpha = std::clamp(0.5f + factor.amplitude *
            astc_vulkan_paired_steering_basis_value(factor.basis, x, y), 0.0f, 1.0f);
        const auto texel = astc_vulkan_make_paired_texel(
            source_q(matrix, row0, column, minimum, range),
            source_q(matrix, row1, column, minimum, range), alpha, layout);
        const size_t offset = (static_cast<size_t>(row) * columns + column) * 4;
        source[offset] = texel.r; source[offset + 1] = texel.g;
        source[offset + 2] = texel.b; source[offset + 3] = texel.a;
    }
    decoded_image decoded;
    if (!roundtrip(source, columns, texture_rows, kD2BlockWidth, kD2BlockHeight, decoded)) return false;
    result = objective_score(decoded_error(matrix, decoded, rows, columns, minimum, range, true, layout),
                             trace, output_trace, objective, rows, columns);
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    paired_model_params params;
    if (!parse_params(argc, argv, params)) {
        std::fprintf(stderr, "usage: %s --model model.gguf --tensor name --trace input.trace "
                             "[--objective activation|yaqa --output-trace output.trace]\n", argv[0]);
        return 2;
    }
    ggml_vk_astc_loaded_matrix matrix;
    ggml_vk_astc_activation_trace trace, output_trace;
    std::string error;
    if (!ggml_vk_astc_load_gguf_matrix(params.model, params.tensor, matrix, error) ||
        !ggml_vk_astc_load_activation_trace(params.input_trace, trace, error) ||
        trace.columns == 0 || trace.columns > matrix.columns || matrix.rows < 2) {
        std::fprintf(stderr, "paired model smoke input error: %s\n", error.c_str());
        return 1;
    }
    const ggml_vk_astc_activation_trace * output_trace_ptr = nullptr;
    if (params.objective == astc_vulkan_objective::two_sided_trace) {
        if (!ggml_vk_astc_load_activation_trace(params.output_trace, output_trace, error) ||
            output_trace.samples != trace.samples || output_trace.columns < std::min<uint32_t>(matrix.rows, 32)) {
            std::fprintf(stderr, "paired model smoke output trace error: %s\n", error.c_str());
            return 1;
        }
        output_trace_ptr = &output_trace;
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
    if (!run_d1(matrix, trace, rows, columns, minimum, range, output_trace_ptr,
                params.objective, d1)) return 1;
    std::printf("paired-model D1 10x8 rate=1.60000 objective=%s score=%.8g\n",
                astc_vulkan_objective_name(params.objective), d1);
    const auto codebook = astc_vulkan_make_paired_steering_codebook();
    for (const auto layout : {astc_vulkan_paired_layout::rg_b, astc_vulkan_paired_layout::r_gb}) {
        double best = std::numeric_limits<double>::infinity();
        size_t best_index = 0;
        for (size_t index = 0; index < codebook.size(); ++index) {
            double value = 0.0;
            if (!run_d2(matrix, trace, rows, columns, minimum, range, layout, codebook[index],
                        output_trace_ptr, params.objective, value)) return 1;
            if (value < best) { best = value; best_index = index; }
        }
        const auto & factor = codebook[best_index];
        std::printf("paired-model D2 %s layout=%s objective=%s rate=%.5f "
                    "best=%s amplitude=%g score=%.8g candidates=%zu\n",
                    kD2FootprintName,
                    astc_vulkan_paired_layout_name(layout),
                    astc_vulkan_objective_name(params.objective),
                    kD2Rate, astc_vulkan_paired_steering_basis_name(factor.basis), factor.amplitude,
                    best, codebook.size());
    }
    return 0;
}
