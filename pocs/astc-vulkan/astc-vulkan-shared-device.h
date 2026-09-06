#pragma once

#include "astc-vulkan-budget.h"
#include "astc-vulkan-format.h"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>

// Owns the Vulkan lifetime shared by all artifacts in one ASTC cache overlay.
// Images, descriptors and dispatch sessions remain tensor-local; the Vulkan
// instance/device/queue must never be recreated per tensor.
class astc_vulkan_shared_device {
public:
    astc_vulkan_shared_device() = default;
    ~astc_vulkan_shared_device();
    astc_vulkan_shared_device(const astc_vulkan_shared_device &) = delete;
    astc_vulkan_shared_device & operator=(const astc_vulkan_shared_device &) = delete;

    bool init(std::string & error);
    bool supports(astc_vulkan_footprint footprint) const;
    void reset();

    bool ready() const { return device_ != VK_NULL_HANDLE; }
    VkPhysicalDevice physical_device() const { return physical_device_; }
    VkDevice device() const { return device_; }
    VkQueue queue() const { return queue_; }
    uint32_t queue_family() const { return queue_family_; }
    const astc_vulkan_memory_budget & memory_budget() const { return memory_budget_; }

private:
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queue_family_ = UINT32_MAX;
    astc_vulkan_memory_budget memory_budget_{};
};
