#include "astc-vulkan-yaqa.h"
#include "astc-vulkan-yaqa-dispatch.h"

#include <vulkan/vulkan.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>

namespace {

std::vector<uint32_t> read_spirv(const char * path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) return {};
    const auto bytes = input.tellg();
    if (bytes <= 0 || bytes % static_cast<std::streamsize>(sizeof(uint32_t)) != 0) return {};
    std::vector<uint32_t> code(static_cast<size_t>(bytes) / sizeof(uint32_t));
    input.seekg(0);
    input.read(reinterpret_cast<char *>(code.data()), bytes);
    return input ? code : std::vector<uint32_t>{};
}

bool choose_compute(VkInstance instance, VkPhysicalDevice & physical, uint32_t & family) {
    uint32_t count = 0;
    if (vkEnumeratePhysicalDevices(instance, &count, nullptr) != VK_SUCCESS || count == 0) return false;
    std::vector<VkPhysicalDevice> devices(count);
    if (vkEnumeratePhysicalDevices(instance, &count, devices.data()) != VK_SUCCESS) return false;
    for (int pass = 0; pass < 2; ++pass) {
        for (VkPhysicalDevice candidate : devices) {
            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(candidate, &properties);
            if ((pass == 0) != (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)) continue;
            uint32_t queue_count = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queue_count, nullptr);
            std::vector<VkQueueFamilyProperties> queues(queue_count);
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queue_count, queues.data());
            for (uint32_t i = 0; i < queue_count; ++i) {
                if ((queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
                    physical = candidate;
                    family = i;
                    return true;
                }
            }
        }
    }
    return false;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 3) return 2;
    const auto partial_spirv = read_spirv(argv[1]);
    const auto reduce_spirv = read_spirv(argv[2]);
    if (partial_spirv.empty() || reduce_spirv.empty()) return 77;

    constexpr uint32_t candidates = 7;
    constexpr uint32_t rows = 4;
    constexpr uint32_t columns = 6;
    constexpr uint32_t samples = 8;
    std::vector<float> errors(static_cast<size_t>(candidates) * rows * columns);
    std::vector<float> input(static_cast<size_t>(samples) * columns);
    std::vector<float> output(static_cast<size_t>(samples) * rows);
    for (size_t i = 0; i < errors.size(); ++i) errors[i] = std::sin(static_cast<float>(i) * .17f) * .4f;
    for (size_t i = 0; i < input.size(); ++i) input[i] = std::cos(static_cast<float>(i) * .11f) * .7f;
    for (size_t i = 0; i < output.size(); ++i) output[i] = std::sin(static_cast<float>(i) * .07f) * .6f;

    const VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr,
        "astc-yaqa-session-smoke", 1, "llama.cpp", 1, VK_API_VERSION_1_1};
    const VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        nullptr, 0, &app, 0, nullptr, 0, nullptr};
    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&instance_info, nullptr, &instance) != VK_SUCCESS) return 77;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    uint32_t family = UINT32_MAX;
    if (!choose_compute(instance, physical, family)) {
        vkDestroyInstance(instance, nullptr);
        return 77;
    }
    constexpr float priority = 1.0f;
    const VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        nullptr, 0, family, 1, &priority};
    const VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        nullptr, 0, 1, &queue_info, 0, nullptr, 0, nullptr, nullptr};
    VkDevice device = VK_NULL_HANDLE;
    if (vkCreateDevice(physical, &device_info, nullptr, &device) != VK_SUCCESS) {
        vkDestroyInstance(instance, nullptr);
        return 77;
    }
    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, family, 0, &queue);

    astc_vulkan_yaqa_session session;
    std::string error;
    bool ok = session.init(physical, device, queue, family, rows, columns, samples,
        partial_spirv, reduce_spirv, candidates, error);
    ok = ok && session.upload_traces(input, output, error);
    std::vector<float> actual;
    ok = ok && session.run(errors, candidates, actual, error);
    std::vector<float> expected(candidates);
    for (uint32_t candidate = 0; candidate < candidates; ++candidate) {
        const auto begin = errors.begin() + static_cast<size_t>(candidate) * rows * columns;
        const auto end = begin + static_cast<size_t>(rows) * columns;
        expected[candidate] = static_cast<float>(astc_vulkan_yaqa_trace_score(
            std::vector<float>(begin, end), rows, columns, input, output, samples));
    }
    if (ok) {
        // Run a second batch without re-uploading traces. This is the resident
        // contract that the monolithic batch smoke could not exercise.
        for (float & value : errors) value *= 0.5f;
        ok = session.run(errors, candidates, actual, error);
        for (float & value : expected) value *= 0.25f;
    }
    session.reset();
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    if (!ok) {
        std::fprintf(stderr, "YAQA resident session failed: %s\n", error.c_str());
        return 1;
    }
    for (uint32_t i = 0; i < candidates; ++i) {
        if (std::fabs(actual[i] - expected[i]) > 2e-3f) {
            std::fprintf(stderr, "YAQA resident mismatch candidate %u: %.6f != %.6f\n",
                i, actual[i], expected[i]);
            return 1;
        }
    }
    std::printf("YAQA resident session GPU/CPU equivalence passed for %u candidates (two batches)\n",
        candidates);
    return 0;
}
