#include "astc-vulkan-input.h"

#include "ggml.h"
#include "gguf.h"

#include <cassert>
#include <cstdio>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

constexpr int kTernaryBlockElements = 256;

std::string temp_path(const char * suffix) {
    return "/tmp/astc-vulkan-input-" + std::to_string(getpid()) + suffix;
}

bool write_fixture(const std::string & path) {
    gguf_context * file = gguf_init_empty();
    if (file == nullptr) return false;
    ggml_init_params params{ 1024 * 1024, nullptr, false };
    ggml_context * context = ggml_init(params);
    if (context == nullptr) {
        gguf_free(file);
        return false;
    }
    ggml_tensor * tensor = ggml_new_tensor_2d(context, GGML_TYPE_F32, 4, 3);
    ggml_set_name(tensor, "astc.test.f32");
    for (int i = 0; i < 12; ++i) static_cast<float *>(tensor->data)[i] = 0.25f * (i - 4);
    gguf_add_tensor(file, tensor);
    ggml_tensor * half = ggml_new_tensor_2d(context, GGML_TYPE_F16, 4, 3);
    ggml_set_name(half, "astc.test.f16");
    for (int i = 0; i < 12; ++i) static_cast<ggml_fp16_t *>(half->data)[i] = ggml_fp32_to_fp16(0.125f * (i - 3));
    gguf_add_tensor(file, half);

    std::vector<float> ternary(kTernaryBlockElements);
    for (int i = 0; i < kTernaryBlockElements; ++i) ternary[i] = static_cast<float>((i % 3) - 1);
    for (const ggml_type type : { GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0 }) {
        ggml_tensor * quantized = ggml_new_tensor_2d(context, type, kTernaryBlockElements, 1);
        ggml_set_name(quantized, type == GGML_TYPE_TQ1_0 ? "astc.test.tq1" : "astc.test.tq2");
        if (ggml_quantize_chunk(type, ternary.data(), quantized->data,
                                0, 1, kTernaryBlockElements, nullptr) != ggml_nbytes(quantized)) {
            ggml_free(context);
            gguf_free(file);
            return false;
        }
        gguf_add_tensor(file, quantized);
    }
    const bool ok = gguf_write_to_file(file, path.c_str(), false);
    ggml_free(context);
    gguf_free(file);
    return ok;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 1) {
        if (argc == 3 && std::string(argv[1]) == "--list") {
            std::vector<ggml_vk_astc_tensor_info> tensors;
            std::string error;
            if (!ggml_vk_astc_list_gguf_tensors(argv[2], tensors, error)) {
                std::fprintf(stderr, "%s\n", error.c_str());
                return 1;
            }
            for (const auto & tensor : tensors) {
                std::printf("%s %ux%u type=%s bytes=%zu rank2=%s\n",
                            tensor.name.c_str(), tensor.rows, tensor.columns,
                            ggml_type_name(static_cast<ggml_type>(tensor.type)),
                            tensor.bytes, tensor.rank2 ? "yes" : "no");
            }
            return 0;
        }
        if (argc != 5 || std::string(argv[1]) != "--model" || std::string(argv[3]) != "--tensor") {
            std::fprintf(stderr, "usage: %s [--list path | --model path --tensor name]\n", argv[0]);
            return 2;
        }
        ggml_vk_astc_loaded_matrix matrix;
        std::string error;
        if (!ggml_vk_astc_load_gguf_matrix(argv[2], argv[4], matrix, error)) {
            std::fprintf(stderr, "%s\n", error.c_str());
            return 1;
        }
        std::printf("loaded GGUF tensor %ux%u\n", matrix.rows, matrix.columns);
        return 0;
    }

    const std::string gguf_path = temp_path(".gguf");
    const std::string trace_path = temp_path(".trace");
    assert(write_fixture(gguf_path));
    ggml_vk_astc_activation_trace expected_trace;
    expected_trace.samples = 2;
    expected_trace.columns = 4;
    expected_trace.values = { 1, 2, 3, 4, -1, -2, -3, -4 };
    std::string error;
    assert(ggml_vk_astc_write_activation_trace(trace_path, expected_trace, error));
    ggml_vk_astc_loaded_matrix matrix;
    assert(ggml_vk_astc_load_gguf_matrix(gguf_path, "astc.test.f32", matrix, error));
    assert(matrix.rows == 3 && matrix.columns == 4 && matrix.values[0] == -1.0f);
    assert(ggml_vk_astc_load_gguf_matrix(gguf_path, "astc.test.f16", matrix, error));
    assert(matrix.rows == 3 && matrix.columns == 4 && matrix.values[0] == -0.375f);
    assert(ggml_vk_astc_load_gguf_matrix(gguf_path, "astc.test.tq1", matrix, error));
    assert(matrix.rows == 1 && matrix.columns == kTernaryBlockElements && matrix.values[0] == -1.0f && matrix.values[2] == 1.0f);
    assert(ggml_vk_astc_load_gguf_matrix(gguf_path, "astc.test.tq2", matrix, error));
    assert(matrix.rows == 1 && matrix.columns == kTernaryBlockElements && matrix.values[0] == -1.0f && matrix.values[2] == 1.0f);
    ggml_vk_astc_activation_trace trace;
    assert(ggml_vk_astc_load_activation_trace(trace_path, trace, error));
    assert(trace.samples == 2 && trace.columns == 4 && trace.values[7] == -4.0f);
    std::remove(gguf_path.c_str());
    std::remove(trace_path.c_str());
    std::printf("ASTC GGUF/input smoke passed\n");
    return 0;
}
