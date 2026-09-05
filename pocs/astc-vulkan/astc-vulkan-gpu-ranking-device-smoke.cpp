#include "astc-vulkan-gpu-ranking-dispatch.h"

#include <vulkan/vulkan.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

std::vector<uint32_t> read_spirv(const char * path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) return {};
    const std::streamsize size = input.tellg();
    if (size <= 0 || size % sizeof(uint32_t) != 0) return {};
    std::vector<uint32_t> result(static_cast<size_t>(size) / sizeof(uint32_t));
    input.seekg(0); input.read(reinterpret_cast<char *>(result.data()), size);
    return input ? result : std::vector<uint32_t>{};
}

std::array<uint8_t, 16> constant_astc(const std::array<uint16_t, 4> & values) {
    // Valid 2D LDR ASTC void-extent block with a constant UNORM16 RGBA color.
    std::array<uint8_t, 16> block{0xfc, 0xfd, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    for (uint32_t channel = 0; channel < 4; ++channel) {
        block[8 + channel * 2] = static_cast<uint8_t>(values[channel] & 0xffu);
        block[9 + channel * 2] = static_cast<uint8_t>(values[channel] >> 8u);
    }
    return block;
}

bool select_device(VkInstance instance, VkPhysicalDevice & physical, uint32_t & family) {
    uint32_t count = 0;
    if (vkEnumeratePhysicalDevices(instance, &count, nullptr) != VK_SUCCESS || count == 0) return false;
    std::vector<VkPhysicalDevice> devices(count);
    if (vkEnumeratePhysicalDevices(instance, &count, devices.data()) != VK_SUCCESS) return false;
    for (const auto device : devices) {
        if (!astc_vulkan_supports_sampled_transfer_extent(device, VK_FORMAT_ASTC_8x5_UNORM_BLOCK, 16, 5)) continue;
        uint32_t queues = 0; vkGetPhysicalDeviceQueueFamilyProperties(device, &queues, nullptr);
        std::vector<VkQueueFamilyProperties> properties(queues);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queues, properties.data());
        for (uint32_t i = 0; i < queues; ++i) if ((properties[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
            physical = device; family = i; return true;
        }
    }
    return false;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 3 || argc > 5) {
        std::fprintf(stderr, "usage: %s <paired-candidate-delta.spv> <paired-proposal-gain.spv> [d1-candidate-delta.spv] [local-select.spv]\n", argv[0]);
        return 2;
    }
    const auto delta_spirv = read_spirv(argv[1]);
    const auto gain_spirv = read_spirv(argv[2]);
    const auto d1_delta_spirv = argc >= 4 ? read_spirv(argv[3]) : std::vector<uint32_t>{};
    const auto local_select_spirv = argc == 5 ? read_spirv(argv[4]) : std::vector<uint32_t>{};
    if (delta_spirv.empty() || gain_spirv.empty() || (argc >= 4 && d1_delta_spirv.empty()) ||
        (argc == 5 && local_select_spirv.empty())) {
        std::fprintf(stderr, "GPU ranking smoke skipped: missing SPIR-V\n");
        return 77;
    }
    const VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr,
        "astc-vulkan-gpu-ranking-device-smoke", 1, "llama.cpp", 1, VK_API_VERSION_1_1};
    const VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, nullptr, 0,
        &application, 0, nullptr, 0, nullptr};
    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&instance_info, nullptr, &instance) != VK_SUCCESS) return 77;
    VkPhysicalDevice physical = VK_NULL_HANDLE; uint32_t family = UINT32_MAX;
    if (!select_device(instance, physical, family)) { vkDestroyInstance(instance, nullptr); return 77; }
    constexpr float priority = 1.0f;
    const VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0,
        family, 1, &priority};
    const VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, nullptr, 0, 1,
        &queue_info, 0, nullptr, 0, nullptr, nullptr};
    VkDevice device = VK_NULL_HANDLE;
    if (vkCreateDevice(physical, &device_info, nullptr, &device) != VK_SUCCESS) { vkDestroyInstance(instance, nullptr); return 77; }
    VkQueue queue = VK_NULL_HANDLE; vkGetDeviceQueue(device, family, 0, &queue);

    const astc_vulkan_gpu_ranking_candidate neutral_rg_b{constant_astc({0x4000, 0x6000, 0x2000, 0x8000}), astc_vulkan_paired_layout::rg_b};
    const astc_vulkan_gpu_ranking_candidate alternate_rg_b{constant_astc({0x6000, 0xa000, 0xc000, 0x8000}), astc_vulkan_paired_layout::rg_b};
    const astc_vulkan_gpu_ranking_candidate neutral_r_gb{constant_astc({0x2000, 0x4000, 0x3000, 0x8000}), astc_vulkan_paired_layout::r_gb};
    const astc_vulkan_gpu_ranking_candidate alternate_r_gb{constant_astc({0x5000, 0x9000, 0xd000, 0x8000}), astc_vulkan_paired_layout::r_gb};
    const std::vector<std::vector<astc_vulkan_gpu_ranking_candidate>> pools{
        {neutral_rg_b, alternate_rg_b}, {neutral_r_gb, alternate_r_gb}};
    astc_vulkan_gpu_ranking_atlas atlas;
    std::string error;
    const bool packed = astc_vulkan_build_gpu_ranking_atlas(astc_vulkan_footprint::k8x5, 2, pools, atlas);
    astc_vulkan_gpu_ranking_session session;
    const bool initialized = packed && session.init(physical, device, queue, family, atlas, 1, 8, 20, 2,
        delta_spirv, gain_spirv, local_select_spirv, error);
    const std::vector<float> activations{1,2,3,4,5,6,7,8, 0.5f,0.5f,0.5f,0.5f,0.5f,0.5f,0.5f,0.5f};
    std::vector<float> residuals(40);
    for (uint32_t sample = 0; sample < 2; ++sample) for (uint32_t row = 0; row < 20; ++row) {
        residuals[sample * 20 + row] = 0.125f * static_cast<float>((sample + 1) * (row + 1));
    }
    std::vector<float> deltas;
    const bool ran = initialized && session.run(activations, 1.0f, deltas, error);
    std::vector<float> gains;
    const bool gained = ran && session.run_proposal_gains(activations, residuals, 1.0f, gains, error);
    if (!gained || deltas.size() != 80 || gains.size() != 4) {
        std::fprintf(stderr, "GPU ranking smoke failed: %s\n", error.c_str());
        return 1;
    }
    const float sum = 36.0f;
    const float sample_scale = 4.0f;
    const float rg_b_delta[2] = {
        (static_cast<float>(0x2000u) + static_cast<float>(0x4000u)) / 2.0f / 65535.0f,
        static_cast<float>(0xa000u) / 65535.0f};
    const float r_gb_delta[2] = {
        static_cast<float>(0x3000u) / 65535.0f,
        (static_cast<float>(0x5000u) + static_cast<float>(0xa000u)) / 2.0f / 65535.0f};
    for (uint32_t candidate = 0; candidate < 4; ++candidate) for (uint32_t sample = 0; sample < 2; ++sample) {
        const bool alternate = candidate == 1 || candidate == 3;
        const float * deltas_for_layout = candidate < 2 ? rg_b_delta : r_gb_delta;
        for (uint32_t row = 0; row < 10; ++row) {
            const uint32_t member = row & 1u;
            const float expected = alternate ? deltas_for_layout[member] * (sample == 0 ? sum : sample_scale) : 0.0f;
            const float actual = deltas[(candidate * 2 + sample) * 10 + row];
            if (std::fabs(actual - expected) > 2e-3f) {
                std::fprintf(stderr, "GPU ranking smoke mismatch c=%u s=%u row=%u: %.6f != %.6f\n", candidate, sample, row, actual, expected);
                return 1;
            }
        }
    }
    for (uint32_t candidate = 0; candidate < 4; ++candidate) {
        const uint32_t source_block_y = candidate < 2 ? 0 : 1;
        float expected_gain = 0.0f;
        for (uint32_t sample = 0; sample < 2; ++sample) for (uint32_t row = 0; row < 10; ++row) {
            const float delta = deltas[(candidate * 2 + sample) * 10 + row];
            const float residual = residuals[sample * 20 + source_block_y * 10 + row];
            expected_gain += delta * (2.0f * residual - delta);
        }
        if (std::fabs(gains[candidate] - expected_gain) > 2e-3f) {
            std::fprintf(stderr, "GPU proposal gain mismatch c=%u: %.6f != %.6f\n",
                candidate, gains[candidate], expected_gain);
            return 1;
        }
    }
    if (!local_select_spirv.empty()) {
        std::vector<uint32_t> selected;
        if (!session.upload_inputs(activations, residuals, error)) {
            std::fprintf(stderr, "GPU resident-input upload failed: %s\n", error.c_str());
            return 1;
        }
        if (!session.run_proposal_gains_and_local_selection(
                activations, residuals, 1.0f, selected, error) || selected.size() != 2) {
            std::fprintf(stderr, "GPU local selection smoke failed: %s\n", error.c_str());
            return 1;
        }
        const uint32_t expected_selected_0 = gains[1] > gains[0] ? 1u : 0u;
        const uint32_t expected_selected_1 = gains[3] > gains[2] ? 3u : 2u;
        if (selected[0] != expected_selected_0 || selected[1] != expected_selected_1) {
            std::fprintf(stderr, "GPU local selection mismatch: [%u,%u] != [%u,%u]\n",
                selected[0], selected[1], expected_selected_0, expected_selected_1);
            return 1;
        }
    }
    astc_vulkan_gpu_ranking_atlas updated_atlas = atlas;
    updated_atlas.payload[8] ^= 0x10u;
    if (!session.update_batch(updated_atlas, error)) {
        std::fprintf(stderr, "GPU ranking batch update failed: %s\n", error.c_str());
        return 1;
    }
    std::vector<float> updated_deltas;
    std::vector<float> updated_gains;
    if (!session.run(activations, 1.0f, updated_deltas, error) ||
        !session.run_proposal_gains(activations, residuals, 1.0f, updated_gains, error) ||
        updated_deltas.size() != deltas.size() || updated_gains.size() != gains.size()) {
        std::fprintf(stderr, "GPU ranking updated-batch run failed: %s\n", error.c_str());
        return 1;
    }
    bool changed = false;
    for (size_t i = 0; i < deltas.size(); ++i) changed |= std::fabs(updated_deltas[i] - deltas[i]) > 1e-5f;
    if (!changed) {
        std::fprintf(stderr, "GPU ranking updated-batch smoke did not observe a payload change\n");
        return 1;
    }

    // Each alternate five-row format uses the same logical D2 shader semantics,
    // but gets a separate atlas because physical block width is part of the
    // push-constant contract. Test every format the device advertises.
    const auto run_secondary = [&](astc_vulkan_footprint footprint, VkFormat format,
                                   uint32_t width, const char * name) {
        if (!astc_vulkan_supports_sampled_transfer_extent(physical, format, width * 2, 5)) {
            std::printf("GPU paired-D2 %s unsupported\n", name);
            return true;
        }
        session.reset();
        astc_vulkan_gpu_ranking_atlas secondary;
        if (!astc_vulkan_build_gpu_ranking_atlas(footprint, 2, pools, secondary) ||
            !session.init(physical, device, queue, family, secondary, 1, width, 20, 2,
                          delta_spirv, gain_spirv, local_select_spirv, error)) {
            std::fprintf(stderr, "GPU D2 %s init failed: %s\n", name, error.c_str());
            return false;
        }
        std::vector<float> secondary_activations(width * 2, 0.5f);
        float sum = 0.0f;
        for (uint32_t column = 0; column < width; ++column) {
            secondary_activations[column] = static_cast<float>(column + 1);
            sum += secondary_activations[column];
        }
        std::vector<float> secondary_deltas, secondary_gains;
        if (!session.run(secondary_activations, 1.0f, secondary_deltas, error) ||
            !session.run_proposal_gains(secondary_activations, residuals, 1.0f, secondary_gains, error) ||
            secondary_deltas.size() != 80 || secondary_gains.size() != 4) {
            std::fprintf(stderr, "GPU D2 %s run failed: %s\n", name, error.c_str());
            return false;
        }
        const float sample_scale = 0.5f * static_cast<float>(width);
        for (uint32_t candidate = 0; candidate < 4; ++candidate) {
            const bool alternate = candidate == 1 || candidate == 3;
            const float * deltas_for_layout = candidate < 2 ? rg_b_delta : r_gb_delta;
            const uint32_t source_block_y = candidate < 2 ? 0 : 1;
            float expected_gain = 0.0f;
            for (uint32_t sample = 0; sample < 2; ++sample) for (uint32_t row = 0; row < 10; ++row) {
                const float expected = alternate ? deltas_for_layout[row & 1u] *
                    (sample == 0 ? sum : sample_scale) : 0.0f;
                const float actual = secondary_deltas[(candidate * 2 + sample) * 10 + row];
                if (std::fabs(actual - expected) > 2e-3f) {
                    std::fprintf(stderr, "GPU D2 %s mismatch c=%u s=%u row=%u: %.6f != %.6f\n",
                        name, candidate, sample, row, actual, expected);
                    return false;
                }
                const float residual = residuals[sample * 20 + source_block_y * 10 + row];
                expected_gain += actual * (2.0f * residual - actual);
            }
            if (std::fabs(secondary_gains[candidate] - expected_gain) > 2e-3f) {
                std::fprintf(stderr, "GPU D2 %s gain mismatch c=%u: %.6f != %.6f\n",
                    name, candidate, secondary_gains[candidate], expected_gain);
                return false;
            }
        }
        return true;
    };
    if (!run_secondary(astc_vulkan_footprint::k6x5, VK_FORMAT_ASTC_6x5_UNORM_BLOCK, 6, "6x5") ||
        !run_secondary(astc_vulkan_footprint::k10x5, VK_FORMAT_ASTC_10x5_UNORM_BLOCK, 10, "10x5")) {
        return 1;
    }

    // D1 uses the same atlas/session resources but one logical output row per
    // physical ASTC row. Exercise all three low/mid-rate scalar footprints.
    const auto run_d1 = [&](astc_vulkan_footprint footprint, VkFormat format,
                            uint32_t width, uint32_t height, const char * name) {
        if (argc != 4 || !astc_vulkan_supports_sampled_transfer_extent(
                physical, format, width * 2, height * 2)) {
            std::printf("GPU D1 %s unsupported or shader not supplied\n", name);
            return true;
        }
        const auto neutral = astc_vulkan_gpu_d1_ranking_candidate{
            constant_astc({0x3000, 0x3000, 0x3000, 0xffff})};
        const auto alternate = astc_vulkan_gpu_d1_ranking_candidate{
            constant_astc({0x6800, 0x6800, 0x6800, 0xffff})};
        const std::vector<std::vector<astc_vulkan_gpu_d1_ranking_candidate>> d1_pools{
            {neutral, alternate}, {neutral, alternate}};
        astc_vulkan_gpu_d1_ranking_atlas d1_atlas;
        if (!astc_vulkan_build_gpu_d1_ranking_atlas(
                footprint, astc_vulkan_d1_semantic_decoder::scalar, 2,
                d1_pools, d1_atlas)) {
            std::fprintf(stderr, "GPU D1 %s atlas build failed\n", name);
            return false;
        }
        session.reset();
        if (!session.init_d1(physical, device, queue, family, d1_atlas, 1,
                             width, height * 2, 2, d1_delta_spirv, gain_spirv,
                             local_select_spirv, error)) {
            std::fprintf(stderr, "GPU D1 %s init failed: %s\n", name, error.c_str());
            return false;
        }
        std::vector<float> d1_activations(width * 2, 1.0f);
        std::vector<float> d1_residuals(height * 2 * 2, 0.25f);
        std::vector<float> d1_deltas, d1_gains;
        if (!session.run(d1_activations, 1.0f, d1_deltas, error) ||
            !session.run_proposal_gains(d1_activations, d1_residuals, 1.0f, d1_gains, error) ||
            d1_deltas.size() != 4u * 2u * height || d1_gains.size() != 4) {
            std::fprintf(stderr, "GPU D1 %s run failed: %s\n", name, error.c_str());
            return false;
        }
        for (float value : d1_deltas) if (!std::isfinite(value)) return false;
        // Candidate zero is the baseline and must have zero delta. The
        // alternate constant block must produce a non-zero semantic delta.
        for (uint32_t index = 0; index < 2u * height; ++index) {
            if (std::fabs(d1_deltas[index]) > 2e-3f) {
                std::fprintf(stderr, "GPU D1 %s baseline delta is non-zero\n", name);
                return false;
            }
        }
        bool changed = false;
        for (size_t index = 2u * height; index < 4u * height; ++index)
            changed |= std::fabs(d1_deltas[index]) > 1e-3f;
        if (!changed) {
            std::fprintf(stderr, "GPU D1 %s alternate delta is zero\n", name);
            return false;
        }
        // Exact CPU oracle: all activations are one and every residual is
        // 0.25, so a constant scalar block has delta=(q1-q0)*width.
        const float delta_q = (static_cast<float>(0x6800u) - static_cast<float>(0x3000u)) / 65535.0f;
        const float expected_delta = delta_q * static_cast<float>(width);
        const float expected_gain = static_cast<float>(2 * height) *
            expected_delta * (0.5f - expected_delta);
        for (uint32_t candidate = 0; candidate < 4; ++candidate) {
            const bool alternate_candidate = candidate == 1 || candidate == 3;
            for (uint32_t sample = 0; sample < 2; ++sample) {
                for (uint32_t row = 0; row < height; ++row) {
                    const float expected = alternate_candidate ? expected_delta : 0.0f;
                    const float actual = d1_deltas[(candidate * 2 + sample) * height + row];
                    if (std::fabs(actual - expected) > 2e-3f) {
                        std::fprintf(stderr,
                            "GPU D1 %s oracle delta mismatch c=%u s=%u row=%u: %.6f != %.6f\n",
                            name, candidate, sample, row, actual, expected);
                        return false;
                    }
                }
            }
            const float expected_candidate_gain = alternate_candidate ? expected_gain : 0.0f;
            if (std::fabs(d1_gains[candidate] - expected_candidate_gain) > 2e-3f) {
                std::fprintf(stderr,
                    "GPU D1 %s oracle gain mismatch c=%u: %.6f != %.6f\n",
                    name, candidate, d1_gains[candidate], expected_candidate_gain);
                return false;
            }
        }
        if (!local_select_spirv.empty()) {
            std::vector<uint32_t> selected;
            if (!session.upload_inputs(d1_activations, d1_residuals, error)) {
                std::fprintf(stderr, "GPU D1 %s resident-input upload failed\n", name);
                return false;
            }
            const uint32_t expected_selected_0 = d1_gains[1] > d1_gains[0] ? 1u : 0u;
            const uint32_t expected_selected_1 = d1_gains[3] > d1_gains[2] ? 3u : 2u;
            if (!session.run_proposal_gains_and_local_selection(
                    d1_activations, d1_residuals, 1.0f, selected, error) || selected.size() != 2 ||
                selected[0] != expected_selected_0 || selected[1] != expected_selected_1) {
                std::fprintf(stderr, "GPU D1 %s local selection mismatch\n", name);
                return false;
            }
        }
        std::printf("GPU D1 %s semantic ranking passed (deltas=%zu gains=%zu)\n",
                    name, d1_deltas.size(), d1_gains.size());
        return true;
    };
    if (!run_d1(astc_vulkan_footprint::k4x4, VK_FORMAT_ASTC_4x4_UNORM_BLOCK, 4, 4, "4x4") ||
        !run_d1(astc_vulkan_footprint::k5x5, VK_FORMAT_ASTC_5x5_UNORM_BLOCK, 5, 5, "5x5") ||
        !run_d1(astc_vulkan_footprint::k6x6, VK_FORMAT_ASTC_6x6_UNORM_BLOCK, 6, 6, "6x6")) return 1;
    session.reset();
    vkDeviceWaitIdle(device);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    std::printf("GPU paired-D2 6x5/8x5/10x5 batch delta/proposal-gain smoke passed\n");
    return 0;
}
