#include "astc-vulkan-input.h"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

struct activations {
    uint32_t samples = 8;
    uint32_t columns = 0;
    std::vector<float> values;
};

double elementwise_mse(const std::vector<float> & reference,
                       const std::vector<float> & candidate) {
    double sum = 0.0;
    for (size_t i = 0; i < reference.size(); ++i) {
        const double delta = reference[i] - candidate[i];
        sum += delta * delta;
    }
    return sum / std::max<size_t>(reference.size(), 1);
}

double activation_relative_mse(const ggml_vk_astc_loaded_matrix & reference,
                               const ggml_vk_astc_loaded_matrix & candidate,
                               const activations & inputs) {
    double error_sum = 0.0;
    double reference_sum = 0.0;
    for (uint32_t sample = 0; sample < inputs.samples; ++sample) {
        const float * input = inputs.values.data() + static_cast<size_t>(sample) * reference.columns;
        for (uint32_t row = 0; row < reference.rows; ++row) {
            double expected = 0.0;
            double actual = 0.0;
            for (uint32_t column = 0; column < reference.columns; ++column) {
                const size_t index = static_cast<size_t>(row) * reference.columns + column;
                expected += reference.values[index] * input[column];
                actual += candidate.values[index] * input[column];
            }
            const double delta = expected - actual;
            error_sum += delta * delta;
            reference_sum += expected * expected;
        }
    }
    return error_sum / std::max(reference_sum, 1e-12);
}

std::vector<float> make_default_activations(uint32_t columns) {
    std::vector<float> result(static_cast<size_t>(8) * columns);
    uint32_t state = 0x9e3779b9u;
    for (uint32_t sample = 0; sample < 8; ++sample) {
        for (uint32_t column = 0; column < columns; ++column) {
            state = state * 1664525u + 1013904223u;
            const float noise = (static_cast<float>(state >> 8) / 16777215.0f - 0.5f) * 0.2f;
            result[static_cast<size_t>(sample) * columns + column] =
                0.5f * std::sin(0.013f * (sample + 1) * (column + 1)) + noise;
        }
    }
    return result;
}

bool find_tensor_info(const std::string & path, const std::string & tensor,
                      ggml_vk_astc_tensor_info & result, std::string & error) {
    std::vector<ggml_vk_astc_tensor_info> tensors;
    if (!ggml_vk_astc_list_gguf_tensors(path, tensors, error)) return false;
    const auto it = std::find_if(tensors.begin(), tensors.end(), [&](const auto & info) {
        return info.name == tensor;
    });
    if (it == tensors.end()) {
        error = "tensor not found in candidate model";
        return false;
    }
    result = *it;
    return true;
}

bool parse_args(int argc, char ** argv, std::string & reference_model,
                std::string & candidate_model, std::string & tensor,
                std::string & trace) {
    if (argc < 7 || argc % 2 == 0) return false;
    for (int i = 1; i < argc; i += 2) {
        const std::string option = argv[i];
        if (option == "--reference-model") reference_model = argv[i + 1];
        else if (option == "--candidate-model") candidate_model = argv[i + 1];
        else if (option == "--tensor") tensor = argv[i + 1];
        else if (option == "--trace") trace = argv[i + 1];
        else return false;
    }
    return !reference_model.empty() && !candidate_model.empty() && !tensor.empty();
}

} // namespace

int main(int argc, char ** argv) {
    std::string reference_model;
    std::string candidate_model;
    std::string tensor;
    std::string trace_path;
    if (!parse_args(argc, argv, reference_model, candidate_model, tensor, trace_path)) {
        std::fprintf(stderr,
                     "usage: %s --reference-model path --candidate-model path --tensor name [--trace path]\n",
                     argv[0]);
        return 2;
    }

    ggml_vk_astc_loaded_matrix reference;
    ggml_vk_astc_loaded_matrix candidate;
    std::string error;
    if (!ggml_vk_astc_load_gguf_matrix(reference_model, tensor, reference, error) ||
        !ggml_vk_astc_load_gguf_matrix(candidate_model, tensor, candidate, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    if (reference.rows != candidate.rows || reference.columns != candidate.columns) {
        std::fprintf(stderr, "reference and candidate tensor dimensions differ\n");
        return 1;
    }

    activations inputs;
    inputs.columns = reference.columns;
    if (!trace_path.empty()) {
        ggml_vk_astc_activation_trace trace;
        if (!ggml_vk_astc_load_activation_trace(trace_path, trace, error) ||
            trace.columns != reference.columns) {
            std::fprintf(stderr, "invalid activation trace: %s\n", error.c_str());
            return 1;
        }
        inputs.samples = trace.samples;
        inputs.values = std::move(trace.values);
    } else {
        inputs.values = make_default_activations(inputs.columns);
    }

    ggml_vk_astc_tensor_info info;
    if (!find_tensor_info(candidate_model, tensor, info, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    const double bits_per_weight = info.bytes * 8.0 / candidate.values.size();
    std::printf("format-baseline tensor=%s rows=%u columns=%u candidate-type=%s "
                "stored-bytes=%zu bits-per-weight=%.5f MSE=%.8g activation-relative-MSE=%.8g\n",
                tensor.c_str(), candidate.rows, candidate.columns,
                ggml_type_name(static_cast<ggml_type>(info.type)), info.bytes, bits_per_weight,
                elementwise_mse(reference.values, candidate.values),
                activation_relative_mse(reference, candidate, inputs));
    return 0;
}
