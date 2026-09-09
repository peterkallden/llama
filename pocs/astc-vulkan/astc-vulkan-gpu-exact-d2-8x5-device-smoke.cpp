#include "astc-gpu-d2-source.h"
#include "astc-gpu-d2-exact-subset.h"
#include "astc-gpu-encoder-exact-dispatch.h"
#include "astc-gpu-encoder-vulkan-verify.h"

#include <astcenc.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#ifndef ASTC_VULKAN_D2_EXACT_WIDTH
#define ASTC_VULKAN_D2_EXACT_WIDTH 8
#endif

namespace {

constexpr astc_vulkan_footprint kTestFootprint =
#if ASTC_VULKAN_D2_EXACT_WIDTH == 6
    astc_vulkan_footprint::k6x5;
#elif ASTC_VULKAN_D2_EXACT_WIDTH == 8
    astc_vulkan_footprint::k8x5;
#elif ASTC_VULKAN_D2_EXACT_WIDTH == 10
    astc_vulkan_footprint::k10x5;
#else
#error "Unsupported D2 exact test footprint"
#endif

constexpr uint32_t kTestWidth = ASTC_VULKAN_D2_EXACT_WIDTH;
constexpr uint32_t kTestHeight = 5u;

} // namespace

int main(int argc, char ** argv) {
    // 8x5 also verifies its extended fitting bank.  The new 6x5/10x5
    // profiles intentionally start with the audited base + dual controls.
    if (argc != 7 && argc != 4) return 2;
    constexpr uint32_t logical_rows = 10;
    constexpr uint32_t logical_columns = 2u * kTestWidth;
    std::vector<float> weights(size_t(logical_rows) * logical_columns);
    for (uint32_t row = 0; row < logical_rows; ++row) for (uint32_t column = 0; column < logical_columns; ++column) {
        const uint32_t pattern = (row * 7u + column * 3u) % 19u;
        weights[size_t(row) * logical_columns + column] = 0.05f + 0.90f * float(pattern) / 18.0f;
    }
    std::vector<astc_vulkan_paired_layout> layouts;
    std::vector<astc_gpu_encoder_source_block> source;
    if (!astc_gpu_d2_make_uniform_layout_map(
            kTestFootprint, logical_rows, logical_columns,
            astc_vulkan_paired_layout::rg_b, layouts) ||
        !astc_gpu_d2_build_paired_source_blocks(
            kTestFootprint, weights, logical_rows, logical_columns,
            layouts, {}, astc_vulkan_paired_semantic::luminance_alpha, source)) return 1;
    astc_gpu_d2_exact_subset_bank candidate_bank;
    if (!astc_gpu_d2_build_luminance_alpha_exact_subset_bank(source, 1, candidate_bank)) return 1;
    const auto & one_plane_request = candidate_bank.one_plane;
    const auto & dual_plane_request = candidate_bank.alpha_dual_plane;
    std::vector<astc_gpu_exact_subset_block> cpu_blocks;
    std::vector<astc_gpu_exact_subset_block> gpu_blocks;
    std::string error;
    if (!astc_gpu_exact_subset_encode_cpu(one_plane_request, cpu_blocks) ||
        !astc_gpu_exact_subset_encode_gpu_default(argv[1], one_plane_request, gpu_blocks, error) ||
        cpu_blocks.size() != 2 || gpu_blocks.size() != 2) {
        std::fprintf(stderr, "D2 exact subset encode failed: %s\n", error.c_str());
        return 1;
    }
    astcenc_config config{};
    if (astcenc_config_init(ASTCENC_PRF_LDR, kTestWidth, kTestHeight, 1, ASTCENC_PRE_FAST, 0, &config) != ASTCENC_SUCCESS) return 1;
    astcenc_context * context = nullptr;
#if defined(GGML_VK_ASTC_EXPERIMENTAL_NEURAL_RANK)
    if (astcenc_context_alloc(&config, 1, &context, nullptr) != ASTCENC_SUCCESS) return 1;
#else
    if (astcenc_context_alloc(&config, 1, &context) != ASTCENC_SUCCESS) return 1;
#endif
    const astcenc_swizzle swizzle{ASTCENC_SWZ_R, ASTCENC_SWZ_G, ASTCENC_SWZ_B, ASTCENC_SWZ_A};
    std::vector<astc_gpu_encoder_finished_block> finished;
    for (size_t index = 0; index < gpu_blocks.size(); ++index) {
        if (cpu_blocks[index].payload != gpu_blocks[index].payload) return 1;
        astcenc_block_info info{};
        if (astcenc_get_block_info(context, gpu_blocks[index].payload.data(), &info) != ASTCENC_SUCCESS ||
            info.is_error_block || info.color_endpoint_modes[0] != 4u || info.partition_count != 1u) return 1;
        astc_gpu_encoder_finished_block result;
        result.source_block_id = gpu_blocks[index].source_block_id;
        result.footprint = kTestFootprint;
        result.payload = gpu_blocks[index].payload;
        result.decoded_rgba.resize(kTestWidth * kTestHeight * 4);
        void * decoded = result.decoded_rgba.data();
        astcenc_image image{kTestWidth, kTestHeight, 1, ASTCENC_TYPE_F32, &decoded};
        if (astcenc_decompress_image(context, result.payload.data(), result.payload.size(),
                                     &image, &swizzle, 0) != ASTCENC_SUCCESS) return 1;
        finished.push_back(std::move(result));
    }
    astcenc_context_free(context);
    double mse = 0.0;
    float max_abs = 0.0f;
    if (!astc_gpu_encoder_verify_d1_vulkan_decode_default(
            argv[argc - 1], finished, 2, mse, max_abs, error)) {
        std::fprintf(stderr, "D2 exact subset Vulkan decode failed: %s\n", error.c_str());
        return 1;
    }
    std::printf("GPU exact D2-LA %ux5: legal=1 payload-equal=1 blocks=%zu decode-mse=%.8g max-abs=%.8g\n",
                kTestWidth, gpu_blocks.size(), mse, max_abs);

    const auto check_one_plane_variant = [&](const char * name,
                                             const astc_gpu_encoder_request & variant,
                                             const char * shader_path) {
        std::vector<astc_gpu_exact_subset_block> cpu_variant;
        std::vector<astc_gpu_exact_subset_block> gpu_variant;
        if (!astc_gpu_exact_subset_encode_cpu(variant, cpu_variant) ||
            !astc_gpu_exact_subset_encode_gpu_default(shader_path, variant, gpu_variant, error) ||
            cpu_variant.size() != gpu_variant.size()) { std::fprintf(stderr, "%s dispatch: %s\n", name, error.c_str()); return false; }
        astcenc_context * check_context = nullptr;
#if defined(GGML_VK_ASTC_EXPERIMENTAL_NEURAL_RANK)
        if (astcenc_context_alloc(&config, 1, &check_context, nullptr) != ASTCENC_SUCCESS) return false;
#else
        if (astcenc_context_alloc(&config, 1, &check_context) != ASTCENC_SUCCESS) return false;
#endif
        bool valid = true;
        for (size_t index = 0; index < gpu_variant.size(); ++index) {
            astcenc_block_info info{};
            valid = valid && cpu_variant[index].payload == gpu_variant[index].payload &&
                astcenc_get_block_info(check_context, gpu_variant[index].payload.data(), &info) == ASTCENC_SUCCESS &&
                !info.is_error_block && !info.is_dual_plane_block &&
                info.color_endpoint_modes[0] == 4u && info.partition_count == 1u;
            if (cpu_variant[index].payload != gpu_variant[index].payload) {
                std::fprintf(stderr, "%s payload mismatch at %zu: cpu=%02x%02x gpu=%02x%02x\n", name, index,
                    cpu_variant[index].payload[0], cpu_variant[index].payload[1],
                    gpu_variant[index].payload[0], gpu_variant[index].payload[1]);
            }
        }
        astcenc_context_free(check_context);
        if (!valid) std::fprintf(stderr, "%s payload mismatch or illegal block\n", name);
        return valid;
    };
    if (argc == 7 &&
        (!check_one_plane_variant("luminance", candidate_bank.one_plane_luminance_weights, argv[2]) ||
         !check_one_plane_variant("alpha", candidate_bank.one_plane_alpha_weights, argv[3]) ||
         !check_one_plane_variant("refined", candidate_bank.one_plane_refined, argv[4]))) return 1;

    const char * dual_shader = argc == 7 ? argv[5] : argv[2];
    if (!astc_gpu_exact_subset_encode_cpu(dual_plane_request, cpu_blocks) ||
        !astc_gpu_exact_subset_encode_gpu_default(dual_shader, dual_plane_request, gpu_blocks, error) ||
        cpu_blocks.size() != 2 || gpu_blocks.size() != 2) return 1;
    context = nullptr;
#if defined(GGML_VK_ASTC_EXPERIMENTAL_NEURAL_RANK)
    if (astcenc_context_alloc(&config, 1, &context, nullptr) != ASTCENC_SUCCESS) return 1;
#else
    if (astcenc_context_alloc(&config, 1, &context) != ASTCENC_SUCCESS) return 1;
#endif
    finished.clear();
    for (size_t index = 0; index < gpu_blocks.size(); ++index) {
        if (cpu_blocks[index].payload != gpu_blocks[index].payload) return 1;
        astcenc_block_info info{};
        if (astcenc_get_block_info(context, gpu_blocks[index].payload.data(), &info) != ASTCENC_SUCCESS ||
            info.is_error_block || !info.is_dual_plane_block || info.dual_plane_component != 3u ||
            info.color_endpoint_modes[0] != 4u || info.partition_count != 1u) return 1;
        astc_gpu_encoder_finished_block result;
        result.source_block_id = gpu_blocks[index].source_block_id;
        result.footprint = kTestFootprint;
        result.payload = gpu_blocks[index].payload;
        result.decoded_rgba.resize(kTestWidth * kTestHeight * 4);
        void * decoded = result.decoded_rgba.data();
        astcenc_image image{kTestWidth, kTestHeight, 1, ASTCENC_TYPE_F32, &decoded};
        if (astcenc_decompress_image(context, result.payload.data(), result.payload.size(),
                                     &image, &swizzle, 0) != ASTCENC_SUCCESS) return 1;
        finished.push_back(std::move(result));
    }
    astcenc_context_free(context);
    if (!astc_gpu_encoder_verify_d1_vulkan_decode_default(
            argv[argc - 1], finished, 2, mse, max_abs, error)) return 1;
    std::printf("GPU exact D2-LA dual-plane %ux5: legal=1 payload-equal=1 blocks=%zu decode-mse=%.8g max-abs=%.8g\n",
                kTestWidth, gpu_blocks.size(), mse, max_abs);
    return 0;
}
