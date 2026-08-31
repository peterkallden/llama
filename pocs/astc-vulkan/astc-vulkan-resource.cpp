#include "astc-vulkan-resource.h"

#include <limits>

uint32_t astc_vulkan_find_memory_type(VkPhysicalDevice physical_device,
                                      uint32_t type_bits,
                                      VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memory_properties{};
    vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);
    for (uint32_t index = 0; index < memory_properties.memoryTypeCount; ++index) {
        if ((type_bits & (1u << index)) != 0 &&
            (memory_properties.memoryTypes[index].propertyFlags & properties) == properties) {
            return index;
        }
    }
    return std::numeric_limits<uint32_t>::max();
}

bool astc_vulkan_supports_sampled_transfer(VkPhysicalDevice physical_device,
                                            VkFormat format) {
    VkFormatProperties properties{};
    vkGetPhysicalDeviceFormatProperties(physical_device, format, &properties);
    constexpr VkFormatFeatureFlags required =
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    return (properties.optimalTilingFeatures & required) == required;
}

bool astc_vulkan_create_sampled_image(VkPhysicalDevice physical_device,
                                       VkDevice device, VkFormat format,
                                       uint32_t width, uint32_t height,
                                       astc_vulkan_image_resources & resources) {
    const VkImageCreateInfo image_info{
        VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, nullptr, 0, VK_IMAGE_TYPE_2D,
        format, { width, height, 1 }, 1, 1, VK_SAMPLE_COUNT_1_BIT,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_SHARING_MODE_EXCLUSIVE, 0, nullptr, VK_IMAGE_LAYOUT_UNDEFINED,
    };
    if (vkCreateImage(device, &image_info, nullptr, &resources.image) != VK_SUCCESS) return false;
    VkMemoryRequirements requirements{};
    vkGetImageMemoryRequirements(device, resources.image, &requirements);
    const uint32_t memory_type = astc_vulkan_find_memory_type(
        physical_device, requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (memory_type == std::numeric_limits<uint32_t>::max()) return false;
    const VkMemoryAllocateInfo allocate_info{
        VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, requirements.size, memory_type,
    };
    if (vkAllocateMemory(device, &allocate_info, nullptr, &resources.memory) != VK_SUCCESS ||
        vkBindImageMemory(device, resources.image, resources.memory, 0) != VK_SUCCESS) return false;
    const VkImageViewCreateInfo view_info{
        VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, nullptr, 0, resources.image,
        VK_IMAGE_VIEW_TYPE_2D, format,
        { VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
          VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY },
        { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
    };
    if (vkCreateImageView(device, &view_info, nullptr, &resources.view) != VK_SUCCESS) return false;
    const VkSamplerCreateInfo sampler_info{
        VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, nullptr, 0, VK_FILTER_NEAREST,
        VK_FILTER_NEAREST, VK_SAMPLER_MIPMAP_MODE_NEAREST,
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, 0.0f, VK_FALSE, 1.0f, VK_FALSE,
        VK_COMPARE_OP_ALWAYS, 0.0f, 0.0f, VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
        VK_FALSE,
    };
    return vkCreateSampler(device, &sampler_info, nullptr, &resources.sampler) == VK_SUCCESS;
}

void astc_vulkan_destroy_sampled_image(VkDevice device,
                                       astc_vulkan_image_resources & resources) {
    if (resources.sampler != VK_NULL_HANDLE) vkDestroySampler(device, resources.sampler, nullptr);
    if (resources.view != VK_NULL_HANDLE) vkDestroyImageView(device, resources.view, nullptr);
    if (resources.memory != VK_NULL_HANDLE) vkFreeMemory(device, resources.memory, nullptr);
    if (resources.image != VK_NULL_HANDLE) vkDestroyImage(device, resources.image, nullptr);
    resources = {};
}
