#include "astc-vulkan-budget.h"

#include <algorithm>
#include <fstream>
#include <limits>
#include <sstream>
#include <vector>

namespace {

bool valid_fraction(float fraction) {
    return fraction > 0.0f && fraction <= 1.0f;
}

uint64_t fraction_of(uint64_t bytes, float fraction) {
    return static_cast<uint64_t>(static_cast<long double>(bytes) * fraction);
}

bool extension_available(VkPhysicalDevice physical_device, const char * name) {
    uint32_t count = 0;
    if (vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &count, nullptr) != VK_SUCCESS) return false;
    std::vector<VkExtensionProperties> properties(count);
    if (vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &count, properties.data()) != VK_SUCCESS) return false;
    return std::any_of(properties.begin(), properties.end(), [name](const VkExtensionProperties & property) {
        return std::string(property.extensionName) == name;
    });
}

} // namespace

bool astc_vulkan_query_host_memory_budget(float fraction,
                                          astc_vulkan_memory_budget & result,
                                          std::string & error) {
    if (!valid_fraction(fraction)) {
        error = "ASTC memory budget fraction must be in (0, 1]";
        return false;
    }
    result = {};
    result.fraction = fraction;
#if defined(__linux__)
    std::ifstream meminfo("/proc/meminfo");
    std::string label;
    uint64_t kib = 0;
    std::string unit;
    while (meminfo >> label >> kib >> unit) {
        if (label == "MemAvailable:") {
            if (kib > std::numeric_limits<uint64_t>::max() / 1024) break;
            result.host_available_bytes = kib * 1024;
            result.host_limit_bytes = fraction_of(result.host_available_bytes, fraction);
            error.clear();
            return true;
        }
    }
#endif
    error = "ASTC host memory auto-detection is unavailable";
    return false;
}

bool astc_vulkan_query_memory_budget(VkPhysicalDevice physical_device,
                                     float fraction,
                                     astc_vulkan_memory_budget & result,
                                     std::string & error) {
    if (physical_device == VK_NULL_HANDLE) {
        error = "ASTC memory budget requires a physical device";
        return false;
    }
    if (!valid_fraction(fraction)) {
        error = "ASTC memory budget fraction must be in (0, 1]";
        return false;
    }
    astc_vulkan_memory_budget host;
    std::string ignored_host_error;
    const bool have_host = astc_vulkan_query_host_memory_budget(fraction, host, ignored_host_error);
    result = host;
    result.fraction = fraction;

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physical_device, &properties);
    result.integrated_gpu = properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;

    VkPhysicalDeviceMemoryBudgetPropertiesEXT memory_budget{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 memory_properties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
    if (extension_available(physical_device, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) {
        memory_properties.pNext = &memory_budget;
        result.uses_vk_ext_memory_budget = true;
    }
    vkGetPhysicalDeviceMemoryProperties2(physical_device, &memory_properties);
    uint64_t best = 0;
    for (uint32_t index = 0; index < memory_properties.memoryProperties.memoryHeapCount; ++index) {
        const VkMemoryHeap & heap = memory_properties.memoryProperties.memoryHeaps[index];
        if ((heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) == 0) continue;
        const uint64_t available = result.uses_vk_ext_memory_budget ? memory_budget.heapBudget[index] : heap.size;
        best = std::max(best, available);
    }
    if (best == 0) {
        error = "ASTC memory budget found no device-local Vulkan heap";
        return false;
    }
    result.device_available_bytes = best;
    result.device_limit_bytes = fraction_of(best, fraction);
    result.effective_device_limit_bytes = result.device_limit_bytes;
    if (result.integrated_gpu && have_host && result.host_limit_bytes != 0) {
        result.effective_device_limit_bytes = std::min(result.effective_device_limit_bytes,
                                                        result.host_limit_bytes);
    }
    error.clear();
    return true;
}

bool astc_vulkan_budget_can_reserve(const astc_vulkan_memory_budget & budget,
                                    uint64_t resident_device_bytes,
                                    uint64_t image_bytes,
                                    uint64_t staging_bytes,
                                    std::string & error) {
    if (budget.effective_device_limit_bytes == 0 ||
        resident_device_bytes > budget.effective_device_limit_bytes ||
        image_bytes > budget.effective_device_limit_bytes - resident_device_bytes) {
        error = "ASTC image exceeds the configured Vulkan memory budget";
        return false;
    }
    if (budget.host_limit_bytes != 0 && staging_bytes > budget.host_limit_bytes) {
        error = "ASTC upload staging exceeds the configured host memory budget";
        return false;
    }
    error.clear();
    return true;
}
