#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>

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
