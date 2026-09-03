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
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <paired-candidate-delta.spv> <paired-proposal-gain.spv>\n", argv[0]);
        return 2;
    }
    const auto delta_spirv = read_spirv(argv[1]);
    const auto gain_spirv = read_spirv(argv[2]);
    if (delta_spirv.empty() || gain_spirv.empty()) {
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
        delta_spirv, gain_spirv, error);
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
    session.reset();
    vkDeviceWaitIdle(device);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    std::printf("GPU paired-D2 batch delta/proposal-gain smoke passed\n");
    return 0;
}
