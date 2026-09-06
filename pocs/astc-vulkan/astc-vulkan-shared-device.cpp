#include "astc-vulkan-shared-device.h"

#include "astc-vulkan-resource.h"

#include <vector>

astc_vulkan_shared_device::~astc_vulkan_shared_device() {
    reset();
}

void astc_vulkan_shared_device::reset() {
    if (owns_device_ && device_ != VK_NULL_HANDLE) vkDestroyDevice(device_, nullptr);
    if (owns_instance_ && instance_ != VK_NULL_HANDLE) vkDestroyInstance(instance_, nullptr);
    instance_ = VK_NULL_HANDLE;
    physical_device_ = VK_NULL_HANDLE;
    device_ = VK_NULL_HANDLE;
    queue_ = VK_NULL_HANDLE;
    queue_family_ = UINT32_MAX;
    memory_budget_ = {};
    owns_instance_ = false;
    owns_device_ = false;
}

bool astc_vulkan_shared_device::init_borrowed(
        VkPhysicalDevice physical_device, VkDevice device, VkQueue queue,
        uint32_t queue_family, std::string & error) {
    reset();
    if (physical_device == VK_NULL_HANDLE || device == VK_NULL_HANDLE ||
        queue == VK_NULL_HANDLE || queue_family == UINT32_MAX) {
        error = "invalid borrowed Vulkan device handles";
        return false;
    }
    physical_device_ = physical_device;
    device_ = device;
    queue_ = queue;
    queue_family_ = queue_family;
    if (!astc_vulkan_query_memory_budget(physical_device_, ASTC_VULKAN_DEFAULT_MEMORY_FRACTION,
                                         memory_budget_, error)) {
        reset();
        return false;
    }
    error.clear();
    return true;
}

bool astc_vulkan_shared_device::init(std::string & error) {
    reset();
    const VkApplicationInfo app_info{
        VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr, "astc-vulkan-runtime", 1,
        "llama.cpp ASTC Vulkan runtime", 1, VK_API_VERSION_1_0};
    const VkInstanceCreateInfo instance_info{
        VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, nullptr, 0, &app_info,
        0, nullptr, 0, nullptr};
    if (vkCreateInstance(&instance_info, nullptr, &instance_) != VK_SUCCESS) {
        error = "failed to create ASTC Vulkan runtime instance";
        reset();
        return false;
    }
    owns_instance_ = true;
    uint32_t device_count = 0;
    if (vkEnumeratePhysicalDevices(instance_, &device_count, nullptr) != VK_SUCCESS || device_count == 0) {
        error = "no Vulkan physical device available for ASTC runtime";
        reset();
        return false;
    }
    std::vector<VkPhysicalDevice> devices(device_count);
    if (vkEnumeratePhysicalDevices(instance_, &device_count, devices.data()) != VK_SUCCESS) {
        error = "failed to enumerate ASTC Vulkan runtime devices";
        reset();
        return false;
    }
    for (VkPhysicalDevice candidate : devices) {
        uint32_t queue_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queue_count, nullptr);
        std::vector<VkQueueFamilyProperties> queues(queue_count);
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queue_count, queues.data());
        for (uint32_t index = 0; index < queue_count; ++index) {
            if ((queues[index].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) continue;
            // D1 6x6 is the baseline runtime requirement. Additional
            // footprints are checked individually at bind time.
            if (!astc_vulkan_supports_sampled_transfer(candidate,
                    astc_vulkan_vk_format(static_cast<uint8_t>(astc_vulkan_footprint::k6x6)))) continue;
            physical_device_ = candidate;
            queue_family_ = index;
            break;
        }
        if (physical_device_ != VK_NULL_HANDLE) break;
    }
    if (physical_device_ == VK_NULL_HANDLE) {
        error = "no Vulkan device supports ASTC-6x6 sampling and compute";
        reset();
        return false;
    }
    constexpr float queue_priority = 1.0f;
    const VkDeviceQueueCreateInfo queue_info{
        VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0, queue_family_, 1, &queue_priority};
    const VkDeviceCreateInfo device_info{
        VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, nullptr, 0, 1, &queue_info,
        0, nullptr, 0, nullptr, nullptr};
    if (vkCreateDevice(physical_device_, &device_info, nullptr, &device_) != VK_SUCCESS) {
        error = "failed to create ASTC Vulkan runtime device";
        reset();
        return false;
    }
    owns_device_ = true;
    vkGetDeviceQueue(device_, queue_family_, 0, &queue_);
    if (!astc_vulkan_query_memory_budget(physical_device_, ASTC_VULKAN_DEFAULT_MEMORY_FRACTION,
                                         memory_budget_, error)) {
        reset();
        return false;
    }
    error.clear();
    return true;
}

bool astc_vulkan_shared_device::supports(astc_vulkan_footprint footprint) const {
    return ready() && astc_vulkan_footprint_is_valid(footprint) &&
        astc_vulkan_supports_sampled_transfer(physical_device_,
            astc_vulkan_vk_format(static_cast<uint8_t>(footprint)));
}
