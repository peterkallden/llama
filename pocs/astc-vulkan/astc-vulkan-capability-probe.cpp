#include <vulkan/vulkan.h>

#include <cstdio>
#include <vector>

namespace {

bool sampled_astc_supported(VkPhysicalDevice device, VkFormat format) {
    VkFormatProperties properties{};
    vkGetPhysicalDeviceFormatProperties(device, format, &properties);
    return (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) != 0;
}

} // namespace

int main() {
    const VkApplicationInfo application_info{
        VK_STRUCTURE_TYPE_APPLICATION_INFO,
        nullptr,
        "astc-vulkan-capability-probe",
        1,
        "llama.cpp ASTC Vulkan PoC",
        1,
        VK_API_VERSION_1_0,
    };
    const VkInstanceCreateInfo instance_info{
        VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        nullptr,
        0,
        &application_info,
        0,
        nullptr,
        0,
        nullptr,
    };

    VkInstance instance = VK_NULL_HANDLE;
    const VkResult instance_result = vkCreateInstance(&instance_info, nullptr, &instance);
    if (instance_result != VK_SUCCESS) {
        std::fprintf(stderr, "ASTC Vulkan probe skipped: vkCreateInstance returned %d\n",
                     static_cast<int>(instance_result));
        return 77;
    }

    uint32_t device_count = 0;
    VkResult result = vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
    if (result != VK_SUCCESS || device_count == 0) {
        std::fprintf(stderr, "ASTC Vulkan probe skipped: no physical device\n");
        vkDestroyInstance(instance, nullptr);
        return 77;
    }

    std::vector<VkPhysicalDevice> devices(device_count);
    result = vkEnumeratePhysicalDevices(instance, &device_count, devices.data());
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "ASTC Vulkan probe skipped: device enumeration failed\n");
        vkDestroyInstance(instance, nullptr);
        return 77;
    }

    bool supports_astc = false;
    for (VkPhysicalDevice device : devices) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(device, &properties);

        const bool supports_4x4 = sampled_astc_supported(
            device, VK_FORMAT_ASTC_4x4_UNORM_BLOCK);
        const bool supports_5x5 = sampled_astc_supported(
            device, VK_FORMAT_ASTC_5x5_UNORM_BLOCK);
        const bool supports_6x6 = sampled_astc_supported(
            device, VK_FORMAT_ASTC_6x6_UNORM_BLOCK);
        const bool supports_8x5 = sampled_astc_supported(
            device, VK_FORMAT_ASTC_8x5_UNORM_BLOCK);
        const bool supports_8x6 = sampled_astc_supported(
            device, VK_FORMAT_ASTC_8x6_UNORM_BLOCK);
        const bool supports_10x6 = sampled_astc_supported(
            device, VK_FORMAT_ASTC_10x6_UNORM_BLOCK);
        const bool supports_8x8 = sampled_astc_supported(
            device, VK_FORMAT_ASTC_8x8_UNORM_BLOCK);
        const bool supports_10x8 = sampled_astc_supported(
            device, VK_FORMAT_ASTC_10x8_UNORM_BLOCK);
        std::printf("%s: ASTC 4x4 sampled=%s, ASTC 5x5 sampled=%s, ASTC 6x6 sampled=%s, ASTC 8x5 sampled=%s, ASTC 8x6 sampled=%s, ASTC 10x6 sampled=%s, ASTC 8x8 sampled=%s, ASTC 10x8 sampled=%s\n",
                    properties.deviceName,
                    supports_4x4 ? "yes" : "no",
                    supports_5x5 ? "yes" : "no",
                    supports_6x6 ? "yes" : "no",
                    supports_8x5 ? "yes" : "no",
                    supports_8x6 ? "yes" : "no",
                    supports_10x6 ? "yes" : "no",
                    supports_8x8 ? "yes" : "no",
                    supports_10x8 ? "yes" : "no");
        supports_astc = supports_astc || (supports_4x4 && supports_6x6);
    }

    vkDestroyInstance(instance, nullptr);
    if (!supports_astc) {
        std::fprintf(stderr, "ASTC Vulkan probe skipped: no device supports both formats\n");
        return 77;
    }
    return 0;
}
