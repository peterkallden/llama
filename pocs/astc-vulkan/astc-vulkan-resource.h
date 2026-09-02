#pragma once

#include <vulkan/vulkan.h>
#include "astc-vulkan-driver.h"

#include <cstdint>
#include <string>
#include <vector>

struct astc_vulkan_image_resources {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
};

uint32_t astc_vulkan_find_memory_type(VkPhysicalDevice physical_device,
                                      uint32_t type_bits,
                                      VkMemoryPropertyFlags properties);
bool astc_vulkan_supports_sampled_transfer(VkPhysicalDevice physical_device,
                                            VkFormat format);
bool astc_vulkan_create_sampled_image(VkPhysicalDevice physical_device,
                                       VkDevice device, VkFormat format,
                                       uint32_t width, uint32_t height,
                                       astc_vulkan_image_resources & resources);
void astc_vulkan_destroy_sampled_image(VkDevice device,
                                       astc_vulkan_image_resources & resources);

VkFormat astc_vulkan_vk_format(uint8_t footprint);

// Owns one sampled ASTC image. The first runtime integration intentionally
// uses one tensor per image; atlas placement can be added without changing the
// upload or descriptor ownership contract.
class astc_vulkan_texture {
public:
    astc_vulkan_texture() = default;
    ~astc_vulkan_texture();
    astc_vulkan_texture(const astc_vulkan_texture &) = delete;
    astc_vulkan_texture & operator=(const astc_vulkan_texture &) = delete;

    bool upload(VkPhysicalDevice physical_device, VkDevice device, VkQueue queue,
                uint32_t queue_family, uint8_t footprint, uint32_t width,
                uint32_t height, const std::vector<uint8_t> & payload,
                std::string & error);
    void reset();
    VkImageView view() const { return resources_.view; }
    VkSampler sampler() const { return resources_.sampler; }
    uint32_t width() const { return width_; }
    uint32_t height() const { return height_; }

private:
    VkDevice device_ = VK_NULL_HANDLE;
    astc_vulkan_image_resources resources_{};
    uint32_t width_ = 0;
    uint32_t height_ = 0;
};

// Binds one validated manifest record to one uploaded sampled image. The
// shader consumes reconstruction separately, keeping storage and math policy
// independent and allowing L+A or scalar modes without changing the loader.
class astc_vulkan_tensor_session {
public:
    bool upload(VkPhysicalDevice physical_device, VkDevice device, VkQueue queue,
                uint32_t queue_family, const astc_vulkan_tensor_record & record,
                const astc_vulkan_reconstruction & reconstruction,
                const std::vector<uint8_t> & payload, std::string & error);
    void reset() { texture_.reset(); record_ = {}; reconstruction_ = {}; }
    const astc_vulkan_texture & texture() const { return texture_; }
    const astc_vulkan_tensor_record & record() const { return record_; }
    const astc_vulkan_reconstruction & reconstruction() const { return reconstruction_; }

private:
    astc_vulkan_texture texture_;
    astc_vulkan_tensor_record record_{};
    astc_vulkan_reconstruction reconstruction_{};
};
