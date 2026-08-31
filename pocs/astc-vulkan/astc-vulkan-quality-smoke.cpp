#include <astcenc.h>

#include "astc-vulkan-input.h"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace {

constexpr float kQualityGateRelativeActivationMse = 0.10f;

struct activation_set {
    uint32_t samples = 0;
    uint32_t columns = 0;
    std::vector<float> values;
};

struct result {
    uint32_t block_width = 0;
    uint32_t block_height = 0;
    size_t astc_bytes = 0;
    size_t metadata_bytes = 0;
    size_t residual_bytes = 0;
    bool block_affine = false;
    const char * transform_name = "linear";
    double mse = 0.0;
    double activation_relative_mse = 0.0;
    double corrected_mse = 0.0;
    double corrected_activation_relative_mse = 0.0;
};

enum class transform_mode {
    linear,
    signed_sqrt,
};

float encode_value(float value, float offset, float scale, transform_mode mode) {
    const float normalized = std::clamp((value - offset) / scale, 0.0f, 1.0f);
    if (mode == transform_mode::linear) return normalized;
    const float centered = 2.0f * normalized - 1.0f;
    return 0.5f + 0.5f * std::copysign(std::sqrt(std::fabs(centered)), centered);
}

float decode_value(float value, float offset, float scale, transform_mode mode) {
    if (mode == transform_mode::linear) return value * scale + offset;
    const float centered = 2.0f * value - 1.0f;
    const float normalized = 0.5f + 0.5f * std::copysign(centered * centered, centered);
    return normalized * scale + offset;
}

bool make_default_activations(uint32_t columns, activation_set & activations) {
    activations.samples = 8;
    activations.columns = columns;
    activations.values.resize(static_cast<size_t>(activations.samples) * columns);
    uint32_t state = 0x9e3779b9u;
    for (uint32_t sample = 0; sample < activations.samples; ++sample) {
        for (uint32_t column = 0; column < columns; ++column) {
            state = state * 1664525u + 1013904223u;
            const float noise = (static_cast<float>(state >> 8) / 16777215.0f - 0.5f) * 0.2f;
            activations.values[static_cast<size_t>(sample) * columns + column] =
                0.5f * std::sin(0.013f * (sample + 1) * (column + 1)) + noise;
        }
    }
    return true;
}

double activation_relative_mse(const std::vector<float> & reference,
                               const std::vector<float> & candidate,
                               uint32_t rows, uint32_t columns,
                               const activation_set & activations) {
    double squared_error = 0.0;
    double reference_energy = 0.0;
    for (uint32_t sample = 0; sample < activations.samples; ++sample) {
        const float * input = activations.values.data() + static_cast<size_t>(sample) * columns;
        for (uint32_t row = 0; row < rows; ++row) {
            double expected = 0.0;
            double actual = 0.0;
            for (uint32_t column = 0; column < columns; ++column) {
                const size_t index = static_cast<size_t>(row) * columns + column;
                expected += reference[index] * input[column];
                actual += candidate[index] * input[column];
            }
            const double error = expected - actual;
            squared_error += error * error;
            reference_energy += expected * expected;
        }
    }
    return squared_error / std::max(reference_energy, 1e-12);
}

double elementwise_mse(const std::vector<float> & reference,
                       const std::vector<float> & candidate) {
    double error = 0.0;
    for (size_t i = 0; i < reference.size(); ++i) {
        const double delta = static_cast<double>(reference[i]) - candidate[i];
        error += delta * delta;
    }
    return error / std::max<size_t>(reference.size(), 1);
}

bool encode_format(const std::vector<float> & weights,
                   uint32_t rows, uint32_t columns,
                   const activation_set & activations,
                   uint32_t block_width, uint32_t block_height,
                   bool block_affine,
                   float residual_fraction,
                   transform_mode transform,
                   result & output) {
    const uint32_t texel_columns = (columns + 3) / 4;
    const uint32_t padded_columns = texel_columns * 4;
    const uint32_t blocks_x = (texel_columns + block_width - 1) / block_width;
    const uint32_t blocks_y = (rows + block_height - 1) / block_height;
    const size_t block_count = static_cast<size_t>(blocks_x) * blocks_y;
    float minimum = std::numeric_limits<float>::infinity();
    float maximum = -std::numeric_limits<float>::infinity();
    for (float value : weights) {
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
    }
    const float scale = std::max(maximum - minimum, 1e-6f);
    std::vector<float> block_offsets(block_count, minimum);
    std::vector<float> block_scales(block_count, scale);
    if (block_affine) {
        std::fill(block_offsets.begin(), block_offsets.end(), std::numeric_limits<float>::infinity());
        std::vector<float> block_maximum(block_count, -std::numeric_limits<float>::infinity());
        for (uint32_t row = 0; row < rows; ++row) {
            const uint32_t block_y = row / block_height;
            for (uint32_t column = 0; column < columns; ++column) {
                const uint32_t block_x = (column / 4) / block_width;
                const size_t block = static_cast<size_t>(block_y) * blocks_x + block_x;
                const float value = weights[static_cast<size_t>(row) * columns + column];
                block_offsets[block] = std::min(block_offsets[block], value);
                block_maximum[block] = std::max(block_maximum[block], value);
            }
        }
        for (size_t block = 0; block < block_count; ++block) {
            block_scales[block] = std::max(block_maximum[block] - block_offsets[block], 1e-6f);
        }
    }
    std::vector<float> texels(static_cast<size_t>(texel_columns) * rows * 4, 0.0f);
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t column = 0; column < columns; ++column) {
            const size_t source = static_cast<size_t>(row) * columns + column;
            const size_t destination = static_cast<size_t>(row) * padded_columns + column;
            const size_t block = static_cast<size_t>(row / block_height) * blocks_x +
                                 (column / 4) / block_width;
            texels[destination] = encode_value(weights[source], block_offsets[block],
                                                block_scales[block], transform);
        }
    }

    astcenc_config config{};
    if (astcenc_config_init(ASTCENC_PRF_LDR, block_width, block_height, 1,
                            ASTCENC_PRE_MEDIUM, 0, &config) != ASTCENC_SUCCESS) {
        return false;
    }
    astcenc_context * context = nullptr;
    if (astcenc_context_alloc(&config, 1, &context) != ASTCENC_SUCCESS) {
        return false;
    }
    void * input_slice = texels.data();
    astcenc_image input{ texel_columns, rows, 1, ASTCENC_TYPE_F32, &input_slice };
    const astcenc_swizzle swizzle{ ASTCENC_SWZ_R, ASTCENC_SWZ_G,
                                   ASTCENC_SWZ_B, ASTCENC_SWZ_A };
    output.astc_bytes = block_count * 16;
    output.metadata_bytes = (block_affine ? block_count : 1) * 2 * sizeof(float);
    std::vector<uint8_t> compressed(output.astc_bytes);
    if (astcenc_compress_image(context, &input, &swizzle, compressed.data(),
                               compressed.size(), 0) != ASTCENC_SUCCESS) {
        astcenc_context_free(context);
        return false;
    }
    std::vector<float> decoded(texels.size());
    void * decoded_slice = decoded.data();
    astcenc_image decoded_image{ texel_columns, rows, 1, ASTCENC_TYPE_F32, &decoded_slice };
    const bool decoded_ok = astcenc_decompress_image(context, compressed.data(),
                                                      compressed.size(), &decoded_image,
                                                      &swizzle, 0) == ASTCENC_SUCCESS;
    astcenc_context_free(context);
    if (!decoded_ok) return false;

    std::vector<float> reconstructed(weights.size());
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t column = 0; column < columns; ++column) {
            const size_t block = static_cast<size_t>(row / block_height) * blocks_x +
                                 (column / 4) / block_width;
            reconstructed[static_cast<size_t>(row) * columns + column] = decode_value(
                decoded[static_cast<size_t>(row) * padded_columns + column],
                block_offsets[block], block_scales[block], transform);
        }
    }
    output.mse = elementwise_mse(weights, reconstructed);
    output.activation_relative_mse = activation_relative_mse(
        weights, reconstructed, rows, columns, activations);

    const size_t residual_count = residual_fraction <= 0.0f ? 0 : std::max<size_t>(1,
        static_cast<size_t>(std::ceil(residual_fraction * weights.size())));
    std::vector<size_t> order(weights.size());
    std::iota(order.begin(), order.end(), 0);
    if (residual_count > 0) {
        std::partial_sort(order.begin(), order.begin() + residual_count, order.end(),
                          [&](size_t lhs, size_t rhs) {
                              return std::fabs(weights[lhs] - reconstructed[lhs]) >
                                     std::fabs(weights[rhs] - reconstructed[rhs]);
                          });
    }
    std::vector<float> corrected = reconstructed;
    for (size_t i = 0; i < residual_count; ++i) {
        const size_t index = order[i];
        corrected[index] = weights[index];
    }
    output.residual_bytes = residual_count * (sizeof(uint32_t) + sizeof(float));
    output.corrected_mse = elementwise_mse(weights, corrected);
    output.corrected_activation_relative_mse = activation_relative_mse(
        weights, corrected, rows, columns, activations);
    output.block_width = block_width;
    output.block_height = block_height;
    output.block_affine = block_affine;
    output.transform_name = transform == transform_mode::linear ? "linear" : "signed-sqrt";
    return std::isfinite(output.mse) && std::isfinite(output.activation_relative_mse);
}

bool make_q4_reference(const std::vector<float> & weights,
                       uint32_t rows, uint32_t columns,
                       std::vector<float> & reconstructed,
                       size_t & bytes) {
    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q4_0, columns);
    std::vector<uint8_t> quantized(row_bytes * rows);
    ggml_quantize_chunk(GGML_TYPE_Q4_0, weights.data(), quantized.data(),
                        0, rows, columns, nullptr);
    const ggml_type_traits * traits = ggml_get_type_traits(GGML_TYPE_Q4_0);
    if (traits == nullptr || traits->to_float == nullptr) return false;
    reconstructed.resize(weights.size());
    traits->to_float(quantized.data(), reconstructed.data(), static_cast<int64_t>(weights.size()));
    bytes = quantized.size();
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 5 && argc != 7 && argc != 9) {
        std::fprintf(stderr, "usage: %s --model path --tensor name [--trace path] [--residual-percent n]\n", argv[0]);
        return 2;
    }
    std::string model;
    std::string tensor;
    std::string trace_path;
    float residual_fraction = 0.01f;
    for (int i = 1; i < argc; i += 2) {
        const std::string option = argv[i];
        if (option == "--model") model = argv[i + 1];
        else if (option == "--tensor") tensor = argv[i + 1];
        else if (option == "--trace") trace_path = argv[i + 1];
        else if (option == "--residual-percent") {
            char * end = nullptr;
            const float percent = std::strtof(argv[i + 1], &end);
            if (end == argv[i + 1] || *end != '\0' || percent < 0.0f || percent > 100.0f) {
                std::fprintf(stderr, "invalid residual percentage: %s\n", argv[i + 1]);
                return 2;
            }
            residual_fraction = percent / 100.0f;
        }
        else {
            std::fprintf(stderr, "unknown or misplaced option: %s\n", option.c_str());
            return 2;
        }
    }
    if (model.empty() || tensor.empty()) return 2;
    ggml_vk_astc_loaded_matrix matrix;
    std::string error;
    if (!ggml_vk_astc_load_gguf_matrix(model, tensor, matrix, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    activation_set activations;
    if (!trace_path.empty()) {
        ggml_vk_astc_activation_trace trace;
        if (!ggml_vk_astc_load_activation_trace(trace_path, trace, error) ||
            trace.columns != matrix.columns) {
            std::fprintf(stderr, "invalid activation trace: %s\n", error.c_str());
            return 1;
        }
        activations.samples = trace.samples;
        activations.columns = trace.columns;
        activations.values = std::move(trace.values);
    } else if (!make_default_activations(matrix.columns, activations)) {
        return 1;
    }

    size_t q4_bytes = 0;
    std::vector<float> q4;
    if (!make_q4_reference(matrix.values, matrix.rows, matrix.columns, q4, q4_bytes)) {
        std::fprintf(stderr, "Q4_0 reference conversion failed\n");
        return 1;
    }
    const double q4_mse = elementwise_mse(matrix.values, q4);
    const double q4_activation = activation_relative_mse(
        matrix.values, q4, matrix.rows, matrix.columns, activations);
    std::printf("tensor=%s rows=%u columns=%u FP16-bytes=%zu Q4_0-bytes=%zu\n",
                tensor.c_str(), matrix.rows, matrix.columns,
                matrix.values.size() * sizeof(ggml_fp16_t), q4_bytes);
    std::printf("Q4_0 MSE %.8g activation-relative-MSE %.8g\n", q4_mse, q4_activation);

    bool gate_passed = true;
    for (const uint32_t block : { 4u, 6u }) {
        for (const bool block_affine : { false, true }) {
            for (const transform_mode transform : { transform_mode::linear,
                                                     transform_mode::signed_sqrt }) {
            result output;
            if (!encode_format(matrix.values, matrix.rows, matrix.columns, activations,
                               block, block, block_affine, residual_fraction, transform, output)) {
                std::fprintf(stderr, "ASTC %ux%u encode/decode failed\n", block, block);
                return 1;
            }
            const size_t total_bytes = output.astc_bytes + output.metadata_bytes;
            const size_t corrected_total = total_bytes + output.residual_bytes;
            std::printf("ASTC %ux%u mode=%s transform=%s residual=%.3g%% bytes=%zu (+%zu residual=%zu) MSE %.8g activation-relative-MSE %.8g corrected-MSE %.8g corrected-activation-relative-MSE %.8g\n",
                        block, block, block_affine ? "block-affine" : "global", output.transform_name,
                        residual_fraction * 100.0f,
                        total_bytes, output.astc_bytes, output.residual_bytes,
                        output.mse, output.activation_relative_mse, output.corrected_mse,
                        output.corrected_activation_relative_mse);
            std::printf("ASTC %ux%u mode=%s transform=%s total-with-residual-bytes=%zu\n",
                        block, block, block_affine ? "block-affine" : "global",
                        output.transform_name, corrected_total);
            gate_passed = gate_passed &&
                output.corrected_activation_relative_mse <= kQualityGateRelativeActivationMse;
            }
        }
    }
    std::printf("quality-gate corrected activation-relative-MSE <= %.3f: %s\n",
                kQualityGateRelativeActivationMse, gate_passed ? "PASS" : "FAIL");
    return gate_passed ? 0 : 3;
}
