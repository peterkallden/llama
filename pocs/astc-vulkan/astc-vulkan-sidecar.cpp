#include "astc-vulkan-sidecar.h"

#include "astc-vulkan-resource.h"

#include <vector>

astc_vulkan_sidecar::~astc_vulkan_sidecar() {
    reset();
}

void astc_vulkan_sidecar::reset() {
    dispatch_.reset();
    adapter_.reset();
    if (device_ != VK_NULL_HANDLE) vkDestroyDevice(device_, nullptr);
    if (instance_ != VK_NULL_HANDLE) vkDestroyInstance(instance_, nullptr);
    instance_ = VK_NULL_HANDLE;
    physical_device_ = VK_NULL_HANDLE;
    device_ = VK_NULL_HANDLE;
    queue_ = VK_NULL_HANDLE;
    queue_family_ = UINT32_MAX;
    binding_ = {};
}

bool astc_vulkan_sidecar::init(astc_vulkan_footprint footprint, std::string & error) {
    reset();
    const VkApplicationInfo app_info{
        VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr, "astc-vulkan-sidecar", 1,
        "llama.cpp ASTC Vulkan sidecar", 1, VK_API_VERSION_1_0};
    const VkInstanceCreateInfo instance_info{
        VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, nullptr, 0, &app_info,
        0, nullptr, 0, nullptr};
    if (vkCreateInstance(&instance_info, nullptr, &instance_) != VK_SUCCESS) {
        error = "failed to create ASTC Vulkan sidecar instance";
        reset();
        return false;
    }
    uint32_t device_count = 0;
    if (vkEnumeratePhysicalDevices(instance_, &device_count, nullptr) != VK_SUCCESS ||
        device_count == 0) {
        error = "no Vulkan physical device available for ASTC sidecar";
        reset();
        return false;
    }
    std::vector<VkPhysicalDevice> devices(device_count);
    if (vkEnumeratePhysicalDevices(instance_, &device_count, devices.data()) != VK_SUCCESS) {
        error = "failed to enumerate ASTC Vulkan sidecar devices";
        reset();
        return false;
    }
    const VkFormat format = astc_vulkan_vk_format(static_cast<uint8_t>(footprint));
    for (VkPhysicalDevice candidate : devices) {
        if (!astc_vulkan_supports_sampled_transfer(candidate, format)) continue;
        uint32_t queue_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queue_count, nullptr);
        std::vector<VkQueueFamilyProperties> queues(queue_count);
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queue_count, queues.data());
        for (uint32_t index = 0; index < queue_count; ++index) {
            if ((queues[index].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) continue;
            physical_device_ = candidate;
            queue_family_ = index;
            break;
        }
        if (physical_device_ != VK_NULL_HANDLE) break;
    }
    if (physical_device_ == VK_NULL_HANDLE) {
        error = "no Vulkan device supports sampled ASTC and compute";
        reset();
        return false;
    }
    constexpr float queue_priority = 1.0f;
    const VkDeviceQueueCreateInfo queue_info{
        VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0,
        queue_family_, 1, &queue_priority};
    const VkDeviceCreateInfo device_info{
        VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, nullptr, 0, 1, &queue_info,
        0, nullptr, 0, nullptr, nullptr};
    if (vkCreateDevice(physical_device_, &device_info, nullptr, &device_) != VK_SUCCESS) {
        error = "failed to create ASTC Vulkan sidecar device";
        reset();
        return false;
    }
    vkGetDeviceQueue(device_, queue_family_, 0, &queue_);
    footprint_ = footprint;
    error.clear();
    return true;
}

bool astc_vulkan_sidecar::load_manifest(const std::string & path, std::string & error) {
    astc_vulkan_manifest loaded;
    if (!astc_vulkan_read_manifest(path, loaded, error)) return false;
    return set_manifest(loaded, error);
}

bool astc_vulkan_sidecar::set_manifest(const astc_vulkan_manifest & manifest,
                                       std::string & error) {
    if (!astc_vulkan_validate_manifest(manifest, error)) return false;
    dispatch_.reset();
    adapter_.reset();
    binding_ = {};
    manifest_ = manifest;
    error.clear();
    return true;
}

bool astc_vulkan_sidecar::bind_tensor(
        const std::string & tensor_name, uint32_t expected_columns, uint32_t expected_rows,
        const std::vector<uint8_t> & payload, astc_vulkan_ffn_binding & binding,
        std::string & error) {
    if (!ready()) {
        error = "ASTC Vulkan sidecar is not initialized";
        return false;
    }
    dispatch_.reset();
    adapter_.reset();
    binding_ = {};
    if (!adapter_.prepare(manifest_, tensor_name, expected_columns, true,
                          expected_rows, binding, error)) return false;
    if (binding.status != astc_vulkan_binding_status::kReady) return true;
    if (binding.record.footprint != footprint_) {
        binding.status = astc_vulkan_binding_status::kFallback;
        binding.fallback_reason = "tensor footprint does not match sidecar format";
        error.clear();
        return true;
    }
    if (!adapter_.upload(physical_device_, device_, queue_, queue_family_,
                         binding, payload, error)) return false;
    binding_ = binding;
    error.clear();
    return true;
}

bool astc_vulkan_sidecar::run(const std::vector<uint32_t> & spirv,
                              const std::vector<float> & activations,
                              std::vector<float> & output, std::string & error) {
    if (!ready() || binding_.status != astc_vulkan_binding_status::kReady) {
        error = "ASTC Vulkan sidecar has no ready tensor";
        return false;
    }
    if (!dispatch_.init(physical_device_, device_, queue_, queue_family_,
                        adapter_.session(), spirv, binding_.record.width,
                        binding_.record.height,
                        static_cast<uint32_t>(activations.size() / binding_.record.width),
                        error)) return false;
    return dispatch_.run(activations, binding_.reconstruction, output, error);
}
