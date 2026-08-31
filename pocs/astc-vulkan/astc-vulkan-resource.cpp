#include "astc-vulkan-resource.h"

#include <limits>
#include <cstring>
#include "astc-vulkan-driver.h"

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
    if (memory_type == std::numeric_limits<uint32_t>::max()) {
        vkDestroyImage(device, resources.image, nullptr);
        resources.image = VK_NULL_HANDLE;
        return false;
    }
    const VkMemoryAllocateInfo allocate_info{
        VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, requirements.size, memory_type,
    };
    if (vkAllocateMemory(device, &allocate_info, nullptr, &resources.memory) != VK_SUCCESS ||
        vkBindImageMemory(device, resources.image, resources.memory, 0) != VK_SUCCESS) {
        if (resources.memory != VK_NULL_HANDLE) vkFreeMemory(device, resources.memory, nullptr);
        vkDestroyImage(device, resources.image, nullptr);
        resources = {};
        return false;
    }
    const VkImageViewCreateInfo view_info{
        VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, nullptr, 0, resources.image,
        VK_IMAGE_VIEW_TYPE_2D, format,
        { VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
          VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY },
        { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
    };
    if (vkCreateImageView(device, &view_info, nullptr, &resources.view) != VK_SUCCESS) {
        vkFreeMemory(device, resources.memory, nullptr);
        vkDestroyImage(device, resources.image, nullptr);
        resources = {};
        return false;
    }
    const VkSamplerCreateInfo sampler_info{
        VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, nullptr, 0, VK_FILTER_NEAREST,
        VK_FILTER_NEAREST, VK_SAMPLER_MIPMAP_MODE_NEAREST,
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, 0.0f, VK_FALSE, 1.0f, VK_FALSE,
        VK_COMPARE_OP_ALWAYS, 0.0f, 0.0f, VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
        VK_FALSE,
    };
    if (vkCreateSampler(device, &sampler_info, nullptr, &resources.sampler) != VK_SUCCESS) {
        vkDestroyImageView(device, resources.view, nullptr);
        vkFreeMemory(device, resources.memory, nullptr);
        vkDestroyImage(device, resources.image, nullptr);
        resources = {};
        return false;
    }
    return true;
}

void astc_vulkan_destroy_sampled_image(VkDevice device,
                                       astc_vulkan_image_resources & resources) {
    if (resources.sampler != VK_NULL_HANDLE) vkDestroySampler(device, resources.sampler, nullptr);
    if (resources.view != VK_NULL_HANDLE) vkDestroyImageView(device, resources.view, nullptr);
    if (resources.memory != VK_NULL_HANDLE) vkFreeMemory(device, resources.memory, nullptr);
    if (resources.image != VK_NULL_HANDLE) vkDestroyImage(device, resources.image, nullptr);
    resources = {};
}

VkFormat astc_vulkan_vk_format(uint8_t footprint) {
    switch (footprint) {
        case 0: return VK_FORMAT_ASTC_4x4_UNORM_BLOCK;
        case 1: return VK_FORMAT_ASTC_5x5_UNORM_BLOCK;
        case 2: return VK_FORMAT_ASTC_6x6_UNORM_BLOCK;
        default: return VK_FORMAT_UNDEFINED;
    }
}

astc_vulkan_texture::~astc_vulkan_texture() {
    reset();
}

void astc_vulkan_texture::reset() {
    if (device_ != VK_NULL_HANDLE) {
        astc_vulkan_destroy_sampled_image(device_, resources_);
    }
    device_ = VK_NULL_HANDLE;
    width_ = height_ = 0;
}

bool astc_vulkan_texture::upload(VkPhysicalDevice physical_device, VkDevice device,
                                 VkQueue queue, uint32_t queue_family, uint8_t footprint,
                                 uint32_t width, uint32_t height,
                                 const std::vector<uint8_t> & payload,
                                 std::string & error) {
    reset();
    const astc_vulkan_footprint fp = static_cast<astc_vulkan_footprint>(footprint);
    const VkFormat format = astc_vulkan_vk_format(footprint);
    const uint64_t expected = astc_vulkan_image_bytes(fp, width, height);
    if (format == VK_FORMAT_UNDEFINED || expected == 0 || payload.size() != expected) {
        error = "ASTC Vulkan texture payload dimensions do not match";
        return false;
    }
    if (!astc_vulkan_supports_sampled_transfer(physical_device, format) ||
        !astc_vulkan_create_sampled_image(physical_device, device, format, width, height, resources_)) {
        error = "ASTC Vulkan sampled image creation failed";
        astc_vulkan_destroy_sampled_image(device, resources_);
        return false;
    }
    VkBuffer staging_buffer = VK_NULL_HANDLE;
    VkDeviceMemory staging_memory = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool success = false;
    do {
        const VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0,
            static_cast<VkDeviceSize>(payload.size()), VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_SHARING_MODE_EXCLUSIVE, 0, nullptr};
        if (vkCreateBuffer(device, &buffer_info, nullptr, &staging_buffer) != VK_SUCCESS) break;
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(device, staging_buffer, &requirements);
        const uint32_t memory_type = astc_vulkan_find_memory_type(
            physical_device, requirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (memory_type == std::numeric_limits<uint32_t>::max()) break;
        const VkMemoryAllocateInfo allocate_info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr,
            requirements.size, memory_type};
        if (vkAllocateMemory(device, &allocate_info, nullptr, &staging_memory) != VK_SUCCESS ||
            vkBindBufferMemory(device, staging_buffer, staging_memory, 0) != VK_SUCCESS) break;
        void * mapped = nullptr;
        if (vkMapMemory(device, staging_memory, 0, payload.size(), 0, &mapped) != VK_SUCCESS) break;
        std::memcpy(mapped, payload.data(), payload.size());
        vkUnmapMemory(device, staging_memory);
        const VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr,
            VK_COMMAND_POOL_CREATE_TRANSIENT_BIT, queue_family};
        if (vkCreateCommandPool(device, &pool_info, nullptr, &command_pool) != VK_SUCCESS) break;
        const VkCommandBufferAllocateInfo command_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            nullptr, command_pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        if (vkAllocateCommandBuffers(device, &command_info, &command_buffer) != VK_SUCCESS) break;
        const VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
            VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
        if (vkBeginCommandBuffer(command_buffer, &begin_info) != VK_SUCCESS) break;
        const VkImageMemoryBarrier to_transfer{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr, 0,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, resources_.image,
            {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &to_transfer);
        const VkBufferImageCopy copy{0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            {0, 0, 0}, {width, height, 1}};
        vkCmdCopyBufferToImage(command_buffer, staging_buffer, resources_.image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        const VkImageMemoryBarrier to_shader{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            nullptr, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, resources_.image,
            {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &to_shader);
        if (vkEndCommandBuffer(command_buffer) != VK_SUCCESS) break;
        const VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr,
            1, &command_buffer, 0, nullptr};
        const VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, 0};
        if (vkCreateFence(device, &fence_info, nullptr, &fence) != VK_SUCCESS ||
            vkQueueSubmit(queue, 1, &submit, fence) != VK_SUCCESS ||
            vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) break;
        success = true;
    } while (false);
    if (fence != VK_NULL_HANDLE) vkDestroyFence(device, fence, nullptr);
    if (command_pool != VK_NULL_HANDLE) vkDestroyCommandPool(device, command_pool, nullptr);
    if (staging_memory != VK_NULL_HANDLE) vkFreeMemory(device, staging_memory, nullptr);
    if (staging_buffer != VK_NULL_HANDLE) vkDestroyBuffer(device, staging_buffer, nullptr);
    if (!success) {
        error = "ASTC Vulkan texture upload failed";
        astc_vulkan_destroy_sampled_image(device, resources_);
        return false;
    }
    device_ = device;
    width_ = width;
    height_ = height;
    error.clear();
    return true;
}
