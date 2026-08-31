#include <astcenc.h>

#include "astc-vulkan-input.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct activations {
    uint32_t samples = 8;
    uint32_t columns = 0;
    std::vector<float> values;
};

double activation_relative_mse(const std::vector<float> & reference,
                               const std::vector<float> & candidate,
                               uint32_t rows, uint32_t columns,
                               const activations & inputs) {
    double error_sum = 0.0;
    double reference_sum = 0.0;
    for (uint32_t sample = 0; sample < inputs.samples; ++sample) {
        const float * input = inputs.values.data() + static_cast<size_t>(sample) * columns;
        for (uint32_t row = 0; row < rows; ++row) {
            double expected = 0.0;
            double actual = 0.0;
            for (uint32_t column = 0; column < columns; ++column) {
                const size_t index = static_cast<size_t>(row) * columns + column;
                expected += reference[index] * input[column];
                actual += candidate[index] * input[column];
            }
            const double delta = expected - actual;
            error_sum += delta * delta;
            reference_sum += expected * expected;
        }
    }
    return error_sum / std::max(reference_sum, 1e-12);
}

double elementwise_mse(const std::vector<float> & reference,
                       const std::vector<float> & candidate) {
    double sum = 0.0;
    for (size_t i = 0; i < reference.size(); ++i) {
        const double delta = reference[i] - candidate[i];
        sum += delta * delta;
    }
    return sum / std::max<size_t>(reference.size(), 1);
}

std::vector<float> make_default_activations(uint32_t columns) {
    activations result;
    result.columns = columns;
    result.values.resize(static_cast<size_t>(result.samples) * columns);
    uint32_t state = 0x9e3779b9u;
    for (uint32_t sample = 0; sample < result.samples; ++sample) {
        for (uint32_t column = 0; column < columns; ++column) {
            state = state * 1664525u + 1013904223u;
            const float noise = (static_cast<float>(state >> 8) / 16777215.0f - 0.5f) * 0.2f;
            result.values[static_cast<size_t>(sample) * columns + column] =
                0.5f * std::sin(0.013f * (sample + 1) * (column + 1)) + noise;
        }
    }
    return result.values;
}

std::vector<float> quantize_levels(const std::vector<float> & weights, uint32_t levels) {
    std::vector<float> result(weights.size());
    float minimum = *std::min_element(weights.begin(), weights.end());
    float maximum = *std::max_element(weights.begin(), weights.end());
    if (levels == 3) {
        const float scale = std::max(std::fabs(minimum), std::fabs(maximum));
        minimum = -scale;
        maximum = scale;
    }
    const float step = (maximum - minimum) / (levels - 1);
    for (size_t i = 0; i < weights.size(); ++i) {
        const float level = step > 0.0f ? std::round((weights[i] - minimum) / step) : 0.0f;
        result[i] = minimum + std::clamp(level, 0.0f, static_cast<float>(levels - 1)) * step;
    }
    return result;
}

bool astc_roundtrip(const std::vector<float> & quantized,
                    uint32_t rows, uint32_t columns, uint32_t block,
                    std::vector<float> & reconstructed, size_t & compressed_bytes) {
    const uint32_t blocks_x = (columns + block - 1) / block;
    const uint32_t blocks_y = (rows + block - 1) / block;
    compressed_bytes = static_cast<size_t>(blocks_x) * blocks_y * 16;
    const float minimum = *std::min_element(quantized.begin(), quantized.end());
    const float maximum = *std::max_element(quantized.begin(), quantized.end());
    const float scale = std::max(maximum - minimum, 1e-6f);
    std::vector<float> input(static_cast<size_t>(rows) * columns * 4);
    for (size_t i = 0; i < quantized.size(); ++i) {
        const float normalized = (quantized[i] - minimum) / scale;
        std::fill_n(input.data() + i * 4, 4, normalized);
    }
    astcenc_config config{};
    if (astcenc_config_init(ASTCENC_PRF_LDR, block, block, 1,
                            ASTCENC_PRE_THOROUGH, 0, &config) != ASTCENC_SUCCESS) return false;
    astcenc_context * context = nullptr;
    if (astcenc_context_alloc(&config, 1, &context) != ASTCENC_SUCCESS) return false;
    void * input_slice = input.data();
    astcenc_image input_image{ columns, rows, 1, ASTCENC_TYPE_F32, &input_slice };
    const astcenc_swizzle swizzle{ ASTCENC_SWZ_R, ASTCENC_SWZ_G,
                                   ASTCENC_SWZ_B, ASTCENC_SWZ_A };
    std::vector<uint8_t> compressed(compressed_bytes);
    astcenc_error status = astcenc_compress_image(
        context, &input_image, &swizzle, compressed.data(), compressed.size(), 0);
    if (status != ASTCENC_SUCCESS) {
        astcenc_context_free(context);
        return false;
    }
    std::vector<float> decoded(input.size());
    void * decoded_slice = decoded.data();
    astcenc_image decoded_image{ columns, rows, 1, ASTCENC_TYPE_F32, &decoded_slice };
    status = astcenc_decompress_image(
        context, compressed.data(), compressed.size(), &decoded_image, &swizzle, 0);
    astcenc_context_free(context);
    if (status != ASTCENC_SUCCESS) return false;
    reconstructed.resize(quantized.size());
    for (size_t i = 0; i < reconstructed.size(); ++i) {
        reconstructed[i] = decoded[i * 4] * scale + minimum;
    }
    return true;
}

bool parse_levels(const std::string & text, std::vector<uint32_t> & levels) {
    std::stringstream stream(text);
    std::string token;
    std::vector<uint32_t> parsed;
    while (std::getline(stream, token, ',')) {
        char * end = nullptr;
        const unsigned long value = std::strtoul(token.c_str(), &end, 10);
        if (token.empty() || end == nullptr || *end != '\0' || value < 2 || value > UINT32_MAX) {
            return false;
        }
        parsed.push_back(static_cast<uint32_t>(value));
    }
    if (parsed.empty()) return false;
    levels = std::move(parsed);
    return true;
}

bool parse_blocks(const std::string & text, std::vector<uint32_t> & blocks) {
    std::vector<uint32_t> parsed;
    if (!parse_levels(text, parsed)) return false;
    for (const uint32_t block : parsed) {
        if (block != 4 && block != 5 && block != 6) return false;
    }
    blocks = std::move(parsed);
    return true;
}

bool parse_args(int argc, char ** argv, std::string & model, std::string & tensor,
                std::string & trace, std::vector<uint32_t> & levels,
                std::vector<uint32_t> & blocks) {
    if (argc < 5 || argc % 2 == 0) return false;
    for (int i = 1; i < argc; i += 2) {
        const std::string option = argv[i];
        if (option == "--model") model = argv[i + 1];
        else if (option == "--tensor") tensor = argv[i + 1];
        else if (option == "--trace") trace = argv[i + 1];
        else if (option == "--levels") {
            if (!parse_levels(argv[i + 1], levels)) return false;
        }
        else if (option == "--blocks") {
            if (!parse_blocks(argv[i + 1], blocks)) return false;
        }
        else return false;
    }
    return !model.empty() && !tensor.empty();
}

} // namespace

int main(int argc, char ** argv) {
    std::string model;
    std::string tensor;
    std::string trace_path;
    std::vector<uint32_t> levels = { 3u, 5u, 8u, 16u };
    std::vector<uint32_t> blocks = { 4u, 6u };
    if (!parse_args(argc, argv, model, tensor, trace_path, levels, blocks)) {
        std::fprintf(stderr,
                     "usage: %s --model path --tensor name [--trace path] [--levels 3,5,8,16] [--blocks 4,5,6]\n",
                     argv[0]);
        return 2;
    }
    ggml_vk_astc_loaded_matrix matrix;
    std::string error;
    if (!ggml_vk_astc_load_gguf_matrix(model, tensor, matrix, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    activations inputs;
    inputs.columns = matrix.columns;
    if (!trace_path.empty()) {
        ggml_vk_astc_activation_trace loaded;
        if (!ggml_vk_astc_load_activation_trace(trace_path, loaded, error) ||
            loaded.columns != matrix.columns) {
            std::fprintf(stderr, "invalid activation trace: %s\n", error.c_str());
            return 1;
        }
        inputs.samples = loaded.samples;
        inputs.values = std::move(loaded.values);
    } else {
        inputs.values = make_default_activations(matrix.columns);
    }

    for (const uint32_t level_count : levels) {
        const std::vector<float> quantized = quantize_levels(matrix.values, level_count);
        std::printf("native-quant levels=%u packed-ideal-bytes=%zu MSE=%.8g activation-relative-MSE=%.8g\n",
                    level_count,
                    static_cast<size_t>(std::ceil(matrix.values.size() * std::log2(level_count) / 8.0)),
                    elementwise_mse(matrix.values, quantized),
                    activation_relative_mse(matrix.values, quantized,
                                             matrix.rows, matrix.columns, inputs));
        for (const uint32_t block : blocks) {
            std::vector<float> reconstructed;
            size_t compressed_bytes = 0;
            if (!astc_roundtrip(quantized, matrix.rows, matrix.columns, block,
                                reconstructed, compressed_bytes)) {
                std::fprintf(stderr, "ASTC-Q roundtrip failed for %ux%u levels=%u\n",
                             block, block, level_count);
                return 1;
            }
            std::printf("ASTC-Q %ux%u levels=%u bytes=%zu bits-per-weight=%.5f MSE=%.8g activation-relative-MSE=%.8g\n",
                        block, block, level_count, compressed_bytes,
                        compressed_bytes * 8.0 / matrix.values.size(),
                        elementwise_mse(matrix.values, reconstructed),
                        activation_relative_mse(matrix.values, reconstructed,
                                                 matrix.rows, matrix.columns, inputs));
        }
    }
    return 0;
}
