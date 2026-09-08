#include "astc-gpu-d1-source.h"
#include "astc-gpu-encoder-exact-dispatch.h"
#include "astc-gpu-encoder-vulkan-verify.h"

#include <astcenc.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 7) return 2;
    constexpr uint32_t block_width = 6;
    constexpr uint32_t block_height = 6;
    constexpr uint32_t width = 12;
    constexpr uint32_t height = 6;
    std::vector<float> weights(width * height);
    for (uint32_t index = 0; index < weights.size(); ++index) {
        weights[index] = 0.10f + 0.80f * static_cast<float>(index % 9u) / 8.0f;
    }
    std::vector<astc_gpu_encoder_source_block> source;
    if (!astc_gpu_d1_build_scalar_source_blocks(
            astc_vulkan_footprint::k6x6, weights, height, width, source)) return 1;
    const auto source_luminance_mse = [&](const std::vector<astc_gpu_encoder_finished_block> & decoded) {
        if (decoded.size() != source.size()) return -1.0;
        double total = 0.0;
        for (size_t block = 0; block < decoded.size(); ++block) {
            for (uint32_t texel = 0; texel < 36; ++texel) {
                const double delta = source[block].texels[texel].rgba[0] -
                    decoded[block].decoded_rgba[texel * 4];
                total += delta * delta;
            }
        }
        return total / 72.0;
    };
    astc_gpu_encoder_request request;
    request.mode = astc_gpu_encode_mode::exact_subset;
    request.footprint = astc_vulkan_footprint::k6x6;
    request.max_blocks_per_batch = 1;
    request.blocks = source;

    std::vector<astc_gpu_exact_subset_block> cpu_blocks;
    std::vector<astc_gpu_exact_subset_block> gpu_blocks;
    std::string error;
    if (!astc_gpu_exact_subset_encode_cpu(request, cpu_blocks) ||
        !astc_gpu_exact_subset_encode_gpu_default(argv[1], request, gpu_blocks, error) ||
        cpu_blocks.size() != 2 || gpu_blocks.size() != 2) {
        std::fprintf(stderr, "GPU exact subset payload mismatch: %s\n", error.c_str());
        return 1;
    }
    for (size_t index = 0; index < cpu_blocks.size(); ++index) {
        if (cpu_blocks[index].payload != gpu_blocks[index].payload) return 1;
    }

    astcenc_config config{};
    if (astcenc_config_init(ASTCENC_PRF_LDR, block_width, block_height, 1, ASTCENC_PRE_FAST, 0, &config) != ASTCENC_SUCCESS) return 1;
    astcenc_context * context = nullptr;
#if defined(GGML_VK_ASTC_EXPERIMENTAL_NEURAL_RANK)
    if (astcenc_context_alloc(&config, 1, &context, nullptr) != ASTCENC_SUCCESS) return 1;
#else
    if (astcenc_context_alloc(&config, 1, &context) != ASTCENC_SUCCESS) return 1;
#endif
    astcenc_block_info info{};
    const astcenc_swizzle swizzle{ASTCENC_SWZ_R, ASTCENC_SWZ_G, ASTCENC_SWZ_B, ASTCENC_SWZ_A};
    std::vector<astc_gpu_encoder_finished_block> finished;
    for (const auto & gpu_block : gpu_blocks) {
        if (astcenc_get_block_info(context, gpu_block.payload.data(), &info) != ASTCENC_SUCCESS) {
            astcenc_context_free(context);
            return 1;
        }
        astc_gpu_encoder_finished_block result;
        result.source_block_id = gpu_block.source_block_id;
        result.footprint = astc_vulkan_footprint::k6x6;
        result.payload = gpu_block.payload;
        result.decoded_rgba.resize(block_width * block_height * 4);
        void * decoded = result.decoded_rgba.data();
        astcenc_image image{block_width, block_height, 1, ASTCENC_TYPE_F32, &decoded};
        if (astcenc_decompress_image(context, result.payload.data(), result.payload.size(),
                                     &image, &swizzle, 0) != ASTCENC_SUCCESS) {
            astcenc_context_free(context);
            return 1;
        }
        finished.push_back(std::move(result));
    }
    for (const auto & block : finished) for (float value : block.decoded_rgba) if (!std::isfinite(value)) return 1;

    double mse = 0.0;
    float max_abs = 0.0f;
    if (!astc_gpu_encoder_verify_d1_vulkan_decode_default(
            argv[5], finished, 2, mse, max_abs, error)) {
        std::fprintf(stderr, "GPU exact subset Vulkan decode failed: %s\n", error.c_str());
        return 1;
    }
    std::printf("GPU exact void extent 6x6: legal=1 payload-equal=1 decode-mse=%.8g max-abs=%.8g\n",
                mse, max_abs);

    request.exact_subset = astc_gpu_exact_subset_kind::d1_luminance_binary_6x6;
    if (!astc_gpu_exact_subset_encode_cpu(request, cpu_blocks) ||
        !astc_gpu_exact_subset_encode_gpu_default(argv[2], request, gpu_blocks, error) ||
        cpu_blocks.size() != 2 || gpu_blocks.size() != 2) {
        std::fprintf(stderr, "GPU exact normal subset payload mismatch: %s\n", error.c_str());
        return 1;
    }
    for (size_t index = 0; index < cpu_blocks.size(); ++index) {
        if (cpu_blocks[index].payload != gpu_blocks[index].payload ||
            astcenc_get_block_info(context, gpu_blocks[index].payload.data(), &info) != ASTCENC_SUCCESS ||
            info.is_error_block) return 1;
    }
    finished.clear();
    for (const auto & gpu_block : gpu_blocks) {
        astc_gpu_encoder_finished_block result;
        result.source_block_id = gpu_block.source_block_id;
        result.footprint = astc_vulkan_footprint::k6x6;
        result.payload = gpu_block.payload;
        result.decoded_rgba.resize(block_width * block_height * 4);
        void * decoded = result.decoded_rgba.data();
        astcenc_image image{block_width, block_height, 1, ASTCENC_TYPE_F32, &decoded};
        if (astcenc_decompress_image(context, result.payload.data(), result.payload.size(),
                                     &image, &swizzle, 0) != ASTCENC_SUCCESS) return 1;
        finished.push_back(std::move(result));
    }
    if (!astc_gpu_encoder_verify_d1_vulkan_decode_default(
            argv[5], finished, 2, mse, max_abs, error)) {
        std::fprintf(stderr, "GPU exact normal subset Vulkan decode failed: %s\n", error.c_str());
        return 1;
    }
    const double normal_source_mse = source_luminance_mse(finished);
    if (normal_source_mse < 0.0) return 1;

    request.exact_subset = astc_gpu_exact_subset_kind::d1_luminance_binary_refined_6x6;
    if (!astc_gpu_exact_subset_encode_cpu(request, cpu_blocks) ||
        !astc_gpu_exact_subset_encode_gpu_default(argv[6], request, gpu_blocks, error) ||
        cpu_blocks.size() != 2 || gpu_blocks.size() != 2) return 1;
    finished.clear();
    for (size_t index = 0; index < gpu_blocks.size(); ++index) {
        if (cpu_blocks[index].payload != gpu_blocks[index].payload ||
            astcenc_get_block_info(context, gpu_blocks[index].payload.data(), &info) != ASTCENC_SUCCESS ||
            info.is_error_block) return 1;
        astc_gpu_encoder_finished_block result;
        result.source_block_id = gpu_blocks[index].source_block_id;
        result.footprint = astc_vulkan_footprint::k6x6;
        result.payload = gpu_blocks[index].payload;
        result.decoded_rgba.resize(block_width * block_height * 4);
        void * decoded = result.decoded_rgba.data();
        astcenc_image image{block_width, block_height, 1, ASTCENC_TYPE_F32, &decoded};
        if (astcenc_decompress_image(context, result.payload.data(), result.payload.size(),
                                     &image, &swizzle, 0) != ASTCENC_SUCCESS) return 1;
        finished.push_back(std::move(result));
    }
    if (!astc_gpu_encoder_verify_d1_vulkan_decode_default(
            argv[5], finished, 2, mse, max_abs, error)) return 1;
    const double refined_source_mse = source_luminance_mse(finished);
    if (refined_source_mse < 0.0 || refined_source_mse > normal_source_mse + 1e-7) return 1;
    std::printf("GPU exact refined D1 6x6: legal=1 payload-equal=1 source-mse=%.8g baseline-mse=%.8g decode-mse=%.8g max-abs=%.8g\n",
                refined_source_mse, normal_source_mse, mse, max_abs);
    astcenc_context_free(context);
    std::printf("GPU exact normal D1 6x6: legal=1 payload-equal=1 blocks=%zu decode-mse=%.8g max-abs=%.8g\n",
                gpu_blocks.size(), mse, max_abs);

    constexpr uint32_t width_5x5 = 10;
    constexpr uint32_t height_5x5 = 5;
    std::vector<float> weights_5x5(width_5x5 * height_5x5);
    for (uint32_t index = 0; index < weights_5x5.size(); ++index) {
        weights_5x5[index] = 0.05f + 0.90f * static_cast<float>((index * 5u) % 17u) / 16.0f;
    }
    std::vector<astc_gpu_encoder_source_block> source_5x5;
    if (!astc_gpu_d1_build_scalar_source_blocks(
            astc_vulkan_footprint::k5x5, weights_5x5, height_5x5, width_5x5, source_5x5)) return 1;
    request.footprint = astc_vulkan_footprint::k5x5;
    request.exact_subset = astc_gpu_exact_subset_kind::d1_luminance_binary_5x5;
    request.blocks = source_5x5;
    if (!astc_gpu_exact_subset_encode_cpu(request, cpu_blocks) ||
        !astc_gpu_exact_subset_encode_gpu_default(argv[3], request, gpu_blocks, error) ||
        cpu_blocks.size() != 2 || gpu_blocks.size() != 2) return 1;
    astcenc_config config_5x5{};
    if (astcenc_config_init(ASTCENC_PRF_LDR, 5, 5, 1, ASTCENC_PRE_FAST, 0, &config_5x5) != ASTCENC_SUCCESS) return 1;
    context = nullptr;
#if defined(GGML_VK_ASTC_EXPERIMENTAL_NEURAL_RANK)
    if (astcenc_context_alloc(&config_5x5, 1, &context, nullptr) != ASTCENC_SUCCESS) return 1;
#else
    if (astcenc_context_alloc(&config_5x5, 1, &context) != ASTCENC_SUCCESS) return 1;
#endif
    finished.clear();
    for (size_t index = 0; index < gpu_blocks.size(); ++index) {
        if (cpu_blocks[index].payload != gpu_blocks[index].payload ||
            astcenc_get_block_info(context, gpu_blocks[index].payload.data(), &info) != ASTCENC_SUCCESS ||
            info.is_error_block) return 1;
        astc_gpu_encoder_finished_block result;
        result.source_block_id = gpu_blocks[index].source_block_id;
        result.footprint = astc_vulkan_footprint::k5x5;
        result.payload = gpu_blocks[index].payload;
        result.decoded_rgba.resize(5 * 5 * 4);
        void * decoded = result.decoded_rgba.data();
        astcenc_image image{5, 5, 1, ASTCENC_TYPE_F32, &decoded};
        if (astcenc_decompress_image(context, result.payload.data(), result.payload.size(),
                                     &image, &swizzle, 0) != ASTCENC_SUCCESS) return 1;
        finished.push_back(std::move(result));
    }
    astcenc_context_free(context);
    if (!astc_gpu_encoder_verify_d1_vulkan_decode_default(
            argv[5], finished, 2, mse, max_abs, error)) return 1;
    std::printf("GPU exact normal D1 5x5: legal=1 payload-equal=1 blocks=%zu decode-mse=%.8g max-abs=%.8g\n",
                gpu_blocks.size(), mse, max_abs);

    constexpr uint32_t width_4x4 = 8;
    constexpr uint32_t height_4x4 = 4;
    std::vector<float> weights_4x4(width_4x4 * height_4x4);
    for (uint32_t index = 0; index < weights_4x4.size(); ++index) {
        weights_4x4[index] = 0.05f + 0.90f * static_cast<float>((index * 3u) % 13u) / 12.0f;
    }
    std::vector<astc_gpu_encoder_source_block> source_4x4;
    if (!astc_gpu_d1_build_scalar_source_blocks(
            astc_vulkan_footprint::k4x4, weights_4x4, height_4x4, width_4x4, source_4x4)) return 1;
    request.footprint = astc_vulkan_footprint::k4x4;
    request.exact_subset = astc_gpu_exact_subset_kind::d1_luminance_quant4_4x4;
    request.blocks = source_4x4;
    if (!astc_gpu_exact_subset_encode_cpu(request, cpu_blocks) ||
        !astc_gpu_exact_subset_encode_gpu_default(argv[4], request, gpu_blocks, error) ||
        cpu_blocks.size() != 2 || gpu_blocks.size() != 2) return 1;
    astcenc_config config_4x4{};
    if (astcenc_config_init(ASTCENC_PRF_LDR, 4, 4, 1, ASTCENC_PRE_FAST, 0, &config_4x4) != ASTCENC_SUCCESS) return 1;
    context = nullptr;
#if defined(GGML_VK_ASTC_EXPERIMENTAL_NEURAL_RANK)
    if (astcenc_context_alloc(&config_4x4, 1, &context, nullptr) != ASTCENC_SUCCESS) return 1;
#else
    if (astcenc_context_alloc(&config_4x4, 1, &context) != ASTCENC_SUCCESS) return 1;
#endif
    finished.clear();
    for (size_t index = 0; index < gpu_blocks.size(); ++index) {
        if (cpu_blocks[index].payload != gpu_blocks[index].payload ||
            astcenc_get_block_info(context, gpu_blocks[index].payload.data(), &info) != ASTCENC_SUCCESS ||
            info.is_error_block) return 1;
        astc_gpu_encoder_finished_block result;
        result.source_block_id = gpu_blocks[index].source_block_id;
        result.footprint = astc_vulkan_footprint::k4x4;
        result.payload = gpu_blocks[index].payload;
        result.decoded_rgba.resize(4 * 4 * 4);
        void * decoded = result.decoded_rgba.data();
        astcenc_image image{4, 4, 1, ASTCENC_TYPE_F32, &decoded};
        if (astcenc_decompress_image(context, result.payload.data(), result.payload.size(),
                                     &image, &swizzle, 0) != ASTCENC_SUCCESS) return 1;
        finished.push_back(std::move(result));
    }
    astcenc_context_free(context);
    if (!astc_gpu_encoder_verify_d1_vulkan_decode_default(
            argv[5], finished, 2, mse, max_abs, error)) return 1;
    std::printf("GPU exact normal D1 4x4: legal=1 payload-equal=1 blocks=%zu decode-mse=%.8g max-abs=%.8g\n",
                gpu_blocks.size(), mse, max_abs);
    return 0;
}
