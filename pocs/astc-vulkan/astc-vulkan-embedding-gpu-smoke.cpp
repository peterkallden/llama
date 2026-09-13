#include "astc-vulkan-embedding-provider.h"
#include "astc-vulkan-embedding-gpu.h"
#include "astc-vulkan-shared-device.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

#ifndef ASTC_VULKAN_EMBEDDING_SHADER_PATH
#define ASTC_VULKAN_EMBEDDING_SHADER_PATH "astc-embedding-get-rows.comp.spv"
#endif

int main(int argc, char ** argv) {
    const std::string root = argc > 1 ? argv[1] :
        "/home/prbm/models/Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf.astc-vulkan.d1-6x6-50m";
    astc_vulkan_embedding_provider cpu;
    std::string error;
    if (!cpu.prepare(root, error)) { std::fprintf(stderr, "embedding GPU smoke: %s\n", error.c_str()); return 77; }
    astc_vulkan_shared_device device;
    if (!device.init(error)) { std::fprintf(stderr, "embedding GPU smoke: %s\n", error.c_str()); return 77; }
    std::vector<uint8_t> shader_bytes;
    {
        FILE * file = std::fopen(ASTC_VULKAN_EMBEDDING_SHADER_PATH, "rb");
        if (file == nullptr) { std::fprintf(stderr, "embedding GPU smoke: missing SPIR-V\n"); return 77; }
        std::fseek(file, 0, SEEK_END); const long size = std::ftell(file); std::fseek(file, 0, SEEK_SET);
        if (size <= 0 || size % 4 != 0) { std::fclose(file); return 77; }
        shader_bytes.resize(static_cast<size_t>(size));
        const bool ok = std::fread(shader_bytes.data(), 1, shader_bytes.size(), file) == shader_bytes.size();
        std::fclose(file); if (!ok) return 77;
    }
    std::vector<uint32_t> spirv(shader_bytes.size() / 4);
    std::memcpy(spirv.data(), shader_bytes.data(), shader_bytes.size());
    astc_vulkan_embedding_gpu_session gpu;
    if (!gpu.init(device.physical_device(), device.device(), device.queue(), device.queue_family(),
                  cpu.payload(), cpu.affine(), cpu.vocabulary(), cpu.dimensions(), cpu.tiles_per_token(), spirv, error)) {
        std::fprintf(stderr, "embedding GPU smoke: %s\n", error.c_str()); return 77;
    }
    std::vector<int32_t> tokens{0, 1, 42, 1234, static_cast<int32_t>(cpu.vocabulary() - 1)};
    std::vector<float> reference(tokens.size() * cpu.dimensions());
    const auto cpu_begin = std::chrono::steady_clock::now();
    constexpr int repeats = 3;
    for (int repeat = 0; repeat < repeats; ++repeat) {
        if (!cpu.run("token_embd.weight", tokens.data(), static_cast<uint32_t>(tokens.size()), reference.data(), cpu.dimensions())) return 1;
    }
    const double cpu_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cpu_begin).count() / repeats;
    std::vector<float> actual;
    const auto gpu_begin = std::chrono::steady_clock::now();
    for (int repeat = 0; repeat < repeats; ++repeat) {
        if (!gpu.run(tokens, actual, error)) { std::fprintf(stderr, "embedding GPU smoke: %s\n", error.c_str()); return 1; }
    }
    const double gpu_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - gpu_begin).count() / repeats;
    double sum = 0.0; float max_error = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        const float diff = std::fabs(actual[i] - reference[i]);
        max_error = std::max(max_error, diff); sum += static_cast<double>(diff) * diff;
    }
    const double rmse = std::sqrt(sum / actual.size());
    std::printf("embedding GPU smoke passed: tokens=%zu dims=%u atlas=%ux%u token_columns=%u max_abs=%.9g rmse=%.9g cpu_ms=%.3f gpu_ms=%.3f gpu_vectors_per_s=%.1f\n",
                tokens.size(), cpu.dimensions(), gpu.atlas_width(), gpu.atlas_height(), gpu.token_columns(), max_error, rmse,
                cpu_ms, gpu_ms, 1000.0 * tokens.size() / gpu_ms);
    return max_error < 2.0e-3f ? 0 : 1;
}
