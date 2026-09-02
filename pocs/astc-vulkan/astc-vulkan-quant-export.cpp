#include "astc-vulkan-input.h"

#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

namespace {

bool write_bytes(const std::string & path, const std::vector<uint8_t> & bytes) {
    std::ofstream output(path, std::ios::binary);
    if (!output) return false;
    output.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    return output.good();
}

} // namespace

int main(int argc, char ** argv) {
    std::string model;
    std::string tensor;
    std::string type_name;
    std::string output;
    std::string trace_path;
    uint32_t max_rows = 0;
    uint32_t max_columns = 0;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (index + 1 >= argc) break;
        if (option == "--model") model = argv[++index];
        else if (option == "--tensor") tensor = argv[++index];
        else if (option == "--type") type_name = argv[++index];
        else if (option == "--output") output = argv[++index];
        else if (option == "--trace") trace_path = argv[++index];
        else if (option == "--max-rows") max_rows = static_cast<uint32_t>(std::stoul(argv[++index]));
        else if (option == "--max-columns") max_columns = static_cast<uint32_t>(std::stoul(argv[++index]));
        else {
            std::fprintf(stderr, "usage: %s --model path --tensor name --type f32|q4_0|tq1_0|tq2_0 --output path [--max-rows N --max-columns N --trace path]\n", argv[0]);
            return 2;
        }
    }
    const bool export_f32 = type_name == "f32";
    const ggml_type type = type_name == "q4_0" ? GGML_TYPE_Q4_0 :
                           type_name == "tq1_0" ? GGML_TYPE_TQ1_0 :
                           type_name == "tq2_0" ? GGML_TYPE_TQ2_0 :
                           export_f32 ? GGML_TYPE_F32 : GGML_TYPE_COUNT;
    if (model.empty() || tensor.empty() || output.empty() || type == GGML_TYPE_COUNT) {
        std::fprintf(stderr, "usage: %s --model path --tensor name --type f32|q4_0|tq1_0|tq2_0 --output path [--max-rows N --max-columns N --trace path]\n", argv[0]);
        return 2;
    }
    ggml_vk_astc_loaded_matrix matrix;
    std::string error;
    if (!ggml_vk_astc_load_gguf_matrix(model, tensor, matrix, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    const uint32_t rows = max_rows == 0 ? matrix.rows : std::min(matrix.rows, max_rows);
    const uint32_t columns = max_columns == 0 ? matrix.columns : std::min(matrix.columns, max_columns);
    if (rows != matrix.rows || columns != matrix.columns) {
        std::vector<float> cropped(static_cast<size_t>(rows) * columns);
        for (uint32_t row = 0; row < rows; ++row) {
            std::copy_n(matrix.values.data() + static_cast<size_t>(row) * matrix.columns, columns,
                        cropped.data() + static_cast<size_t>(row) * columns);
        }
        matrix.rows = rows;
        matrix.columns = columns;
        matrix.values = std::move(cropped);
    }
    if (export_f32) {
        if (!write_bytes(output, std::vector<uint8_t>(
                reinterpret_cast<const uint8_t *>(matrix.values.data()),
                reinterpret_cast<const uint8_t *>(matrix.values.data()) +
                    matrix.values.size() * sizeof(float)))) {
            std::fprintf(stderr, "cannot write %s\n", output.c_str());
            return 1;
        }
        std::printf("quant-export type=f32 rows=%u columns=%u bytes=%zu bits-per-weight=32.00000 output=%s\n",
                    matrix.rows, matrix.columns, matrix.values.size() * sizeof(float), output.c_str());
        return 0;
    }
    const uint32_t block_elements = type == GGML_TYPE_Q4_0 ? 32 : 256;
    if (matrix.columns % block_elements != 0) {
        std::fprintf(stderr, "%s columns=%u is not divisible by %u\n",
                     type_name.c_str(), matrix.columns, block_elements);
        return 1;
    }
    const size_t row_bytes = ggml_row_size(type, matrix.columns);
    std::vector<uint8_t> packed(row_bytes * matrix.rows);
    if (ggml_quantize_chunk(type, matrix.values.data(), packed.data(),
                            0, matrix.rows, matrix.columns, nullptr) != packed.size()) {
        std::fprintf(stderr, "quantization failed\n");
        return 1;
    }
    if (!write_bytes(output, packed)) {
        std::fprintf(stderr, "cannot write %s\n", output.c_str());
        return 1;
    }
    const ggml_type_traits * traits = ggml_get_type_traits(type);
    double element_mse = NAN;
    double activation_mse = NAN;
    if (!trace_path.empty() && traits != nullptr && traits->to_float != nullptr) {
        std::vector<float> reconstructed(matrix.values.size());
        traits->to_float(packed.data(), reconstructed.data(), static_cast<int64_t>(reconstructed.size()));
        double squared_error = 0.0;
        for (size_t i = 0; i < reconstructed.size(); ++i) {
            const double error_value = static_cast<double>(matrix.values[i]) - reconstructed[i];
            squared_error += error_value * error_value;
        }
        element_mse = squared_error / reconstructed.size();
        ggml_vk_astc_activation_trace trace;
        if (!ggml_vk_astc_load_activation_trace(trace_path, trace, error) || trace.columns < matrix.columns) {
            std::fprintf(stderr, "invalid activation trace: %s\n", error.c_str());
            return 1;
        }
        double error_energy = 0.0;
        double reference_energy = 0.0;
        for (uint32_t sample = 0; sample < trace.samples; ++sample) {
            for (uint32_t row = 0; row < matrix.rows; ++row) {
                double reference_dot = 0.0;
                double error_dot = 0.0;
                for (uint32_t column = 0; column < matrix.columns; ++column) {
                    const float input = trace.values[static_cast<size_t>(sample) * trace.columns + column];
                    const size_t index = static_cast<size_t>(row) * matrix.columns + column;
                    reference_dot += matrix.values[index] * input;
                    error_dot += (matrix.values[index] - reconstructed[index]) * input;
                }
                reference_energy += reference_dot * reference_dot;
                error_energy += error_dot * error_dot;
            }
        }
        activation_mse = error_energy / std::max(reference_energy, 1e-12);
    }
    std::printf("quant-export type=%s rows=%u columns=%u bytes=%zu bits-per-weight=%.5f element-MSE=%.8g activation-relative-MSE=%.8g output=%s\n",
                type_name.c_str(), matrix.rows, matrix.columns, packed.size(),
                packed.size() * 8.0 / matrix.values.size(), element_mse, activation_mse, output.c_str());
    return 0;
}
