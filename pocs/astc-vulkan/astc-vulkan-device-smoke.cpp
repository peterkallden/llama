#include "astc-vulkan-contract.h"
#include "astc-vulkan-driver.h"
#include "astc-vulkan-resource.h"
#include "astc-vulkan-paired-dispatch.h"
#include "astc-vulkan-paired-layout.h"

#include <vulkan/vulkan.h>

#include <cstdio>
#include <cstring>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

namespace {

constexpr VkDeviceSize kAstcBlockBytes =
    ggml_vk_astc_4x4_unorm_rgba.block_size_bytes;

struct astc_image_resources {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
};

uint32_t find_memory_type(VkPhysicalDevice physical_device, uint32_t type_bits,
                          VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memory_properties{};
    vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);
    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) != 0 &&
            (memory_properties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    return UINT32_MAX;
}

bool format_supports(VkPhysicalDevice physical_device, VkFormat format,
                     VkFormatFeatureFlags required) {
    VkFormatProperties properties{};
    vkGetPhysicalDeviceFormatProperties(physical_device, format, &properties);
    return (properties.optimalTilingFeatures & required) == required;
}

bool create_and_upload_image_legacy(VkPhysicalDevice physical_device, VkDevice device,
                             VkQueue queue, uint32_t queue_family,
                             VkFormat format, VkExtent3D extent) {
    const VkImageCreateInfo image_info{
        VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, nullptr, 0, VK_IMAGE_TYPE_2D,
        format, extent, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_SHARING_MODE_EXCLUSIVE, 0, nullptr, VK_IMAGE_LAYOUT_UNDEFINED,
    };

    astc_image_resources resources;
    if (vkCreateImage(device, &image_info, nullptr, &resources.image) != VK_SUCCESS) {
        return false;
    }

    VkMemoryRequirements image_requirements{};
    vkGetImageMemoryRequirements(device, resources.image, &image_requirements);
    const uint32_t image_memory_type = find_memory_type(
        physical_device, image_requirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (image_memory_type == UINT32_MAX) {
        vkDestroyImage(device, resources.image, nullptr);
        return false;
    }

    const VkMemoryAllocateInfo image_allocate_info{
        VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr,
        image_requirements.size, image_memory_type,
    };
    if (vkAllocateMemory(device, &image_allocate_info, nullptr, &resources.memory) != VK_SUCCESS ||
        vkBindImageMemory(device, resources.image, resources.memory, 0) != VK_SUCCESS) {
        vkDestroyImage(device, resources.image, nullptr);
        if (resources.memory != VK_NULL_HANDLE) {
            vkFreeMemory(device, resources.memory, nullptr);
        }
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
        vkDestroyImage(device, resources.image, nullptr);
        vkFreeMemory(device, resources.memory, nullptr);
        return false;
    }

    const VkSamplerCreateInfo sampler_info{
        VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, nullptr, 0,
        VK_FILTER_NEAREST, VK_FILTER_NEAREST, VK_SAMPLER_MIPMAP_MODE_NEAREST,
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, 0.0f, VK_FALSE, 1.0f, VK_FALSE,
        VK_COMPARE_OP_ALWAYS, 0.0f, 0.0f, VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
        VK_FALSE,
    };
    if (vkCreateSampler(device, &sampler_info, nullptr, &resources.sampler) != VK_SUCCESS) {
        vkDestroyImageView(device, resources.view, nullptr);
        vkDestroyImage(device, resources.image, nullptr);
        vkFreeMemory(device, resources.memory, nullptr);
        return false;
    }

    const VkBufferCreateInfo buffer_info{
        VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, kAstcBlockBytes,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, nullptr,
    };
    VkBuffer staging_buffer = VK_NULL_HANDLE;
    VkDeviceMemory staging_memory = VK_NULL_HANDLE;
    bool success = false;
    do {
        if (vkCreateBuffer(device, &buffer_info, nullptr, &staging_buffer) != VK_SUCCESS) {
            break;
        }
        VkMemoryRequirements staging_requirements{};
        vkGetBufferMemoryRequirements(device, staging_buffer, &staging_requirements);
        const uint32_t staging_memory_type = find_memory_type(
            physical_device, staging_requirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (staging_memory_type == UINT32_MAX) {
            break;
        }
        const VkMemoryAllocateInfo staging_allocate_info{
            VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr,
            staging_requirements.size, staging_memory_type,
        };
        if (vkAllocateMemory(device, &staging_allocate_info, nullptr, &staging_memory) != VK_SUCCESS ||
            vkBindBufferMemory(device, staging_buffer, staging_memory, 0) != VK_SUCCESS) {
            break;
        }
        void * mapped = nullptr;
        if (vkMapMemory(device, staging_memory, 0, kAstcBlockBytes, 0, &mapped) != VK_SUCCESS) {
            break;
        }
        std::memset(mapped, 0, kAstcBlockBytes);
        vkUnmapMemory(device, staging_memory);

        const VkCommandPoolCreateInfo pool_info{
            VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr,
            VK_COMMAND_POOL_CREATE_TRANSIENT_BIT, queue_family,
        };
        VkCommandPool command_pool = VK_NULL_HANDLE;
        if (vkCreateCommandPool(device, &pool_info, nullptr, &command_pool) != VK_SUCCESS) {
            break;
        }
        const VkCommandBufferAllocateInfo command_allocate_info{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, command_pool,
            VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1,
        };
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        if (vkAllocateCommandBuffers(device, &command_allocate_info, &command_buffer) != VK_SUCCESS) {
            vkDestroyCommandPool(device, command_pool, nullptr);
            break;
        }
        const VkCommandBufferBeginInfo begin_info{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
            VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr,
        };
        if (vkBeginCommandBuffer(command_buffer, &begin_info) != VK_SUCCESS) {
            vkDestroyCommandPool(device, command_pool, nullptr);
            break;
        }

        const VkImageMemoryBarrier to_transfer{
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr, 0,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED, resources.image,
            { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
        };
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
            &to_transfer);

        const VkBufferImageCopy copy_region{
            0, 0, 0, { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, { 0, 0, 0 }, extent,
        };
        vkCmdCopyBufferToImage(command_buffer, staging_buffer, resources.image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy_region);

        const VkImageMemoryBarrier to_shader{
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED, resources.image,
            { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
        };
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1,
            &to_shader);
        if (vkEndCommandBuffer(command_buffer) != VK_SUCCESS) {
            vkDestroyCommandPool(device, command_pool, nullptr);
            break;
        }

        const VkSubmitInfo submit_info{
            VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1,
            &command_buffer, 0, nullptr,
        };
        const VkFenceCreateInfo fence_info{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, 0 };
        VkFence fence = VK_NULL_HANDLE;
        if (vkCreateFence(device, &fence_info, nullptr, &fence) != VK_SUCCESS ||
            vkQueueSubmit(queue, 1, &submit_info, fence) != VK_SUCCESS ||
            vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
            if (fence != VK_NULL_HANDLE) {
                vkDestroyFence(device, fence, nullptr);
            }
            vkDestroyCommandPool(device, command_pool, nullptr);
            break;
        }
        vkDestroyFence(device, fence, nullptr);
        vkDestroyCommandPool(device, command_pool, nullptr);
        success = true;
    } while (false);

    if (staging_buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, staging_buffer, nullptr);
    }
    if (staging_memory != VK_NULL_HANDLE) {
        vkFreeMemory(device, staging_memory, nullptr);
    }
    vkDestroySampler(device, resources.sampler, nullptr);
    vkDestroyImageView(device, resources.view, nullptr);
    vkDestroyImage(device, resources.image, nullptr);
    vkFreeMemory(device, resources.memory, nullptr);
    return success;
}

bool create_and_upload_image(VkPhysicalDevice physical_device, VkDevice device,
                             VkQueue queue, uint32_t queue_family,
                             VkFormat format, VkExtent3D extent) {
    const uint8_t footprint = format == VK_FORMAT_ASTC_4x4_UNORM_BLOCK ? 0 :
                              format == VK_FORMAT_ASTC_5x5_UNORM_BLOCK ? 1 :
                              format == VK_FORMAT_ASTC_6x6_UNORM_BLOCK ? 2 :
                              format == VK_FORMAT_ASTC_8x6_UNORM_BLOCK ? 3 :
                              format == VK_FORMAT_ASTC_8x8_UNORM_BLOCK ? 4 :
                              format == VK_FORMAT_ASTC_10x6_UNORM_BLOCK ? 5 :
                              format == VK_FORMAT_ASTC_10x8_UNORM_BLOCK ? 6 : 7;
    const auto fp = static_cast<astc_vulkan_footprint>(footprint);
    const size_t bytes = static_cast<size_t>(astc_vulkan_image_bytes(
        fp, extent.width, extent.height));
    std::vector<uint8_t> payload(bytes, 0);
    astc_vulkan_tensor_record record{
        "device-smoke", extent.width, extent.height,
        fp, 0, static_cast<uint64_t>(bytes)};
    astc_vulkan_tensor_session session;
    const astc_vulkan_reconstruction reconstruction{};
    std::string error;
    return session.upload(physical_device, device, queue, queue_family, record,
                          reconstruction, payload, error);
}

bool repeat_create_and_upload_image(VkPhysicalDevice physical_device, VkDevice device,
                                    VkQueue queue, uint32_t queue_family,
                                    VkFormat format, VkExtent3D extent,
                                    uint32_t repeats) {
    for (uint32_t repeat = 0; repeat < repeats; ++repeat) {
        if (!create_and_upload_image(physical_device, device, queue, queue_family, format, extent)) {
            return false;
        }
    }
    return true;
}

bool stream_upload_bands(VkPhysicalDevice physical_device, VkDevice device, VkQueue queue,
                         uint32_t queue_family, astc_vulkan_footprint footprint,
                         uint32_t width, uint32_t logical_height, bool paired) {
    astc_vulkan_stream_geometry geometry;
    std::string error;
    if (!astc_vulkan_make_stream_geometry(footprint, width, logical_height, paired,
                                           geometry, error)) return false;
    std::vector<astc_vulkan_stream_band> bands;
    if (!astc_vulkan_plan_stream(geometry, geometry.blocks_x * 16u * 2u,
                                 bands, error)) return false;
    std::vector<uint8_t> resident(static_cast<size_t>(geometry.payload_bytes), 0);
    astc_vulkan_tensor_record record{
        "stream-smoke", width, logical_height, footprint, 0, geometry.payload_bytes};
    record.representation = paired ? astc_vulkan_representation::kPairedD2 :
                                     astc_vulkan_representation::kScalar;
    astc_vulkan_tensor_session session;
    for (const auto & band : bands) {
        std::vector<uint8_t> payload(
            resident.begin() + static_cast<size_t>(band.payload_offset),
            resident.begin() + static_cast<size_t>(band.payload_offset + band.payload_size));
        if (!session.upload_band(physical_device, device, queue, queue_family, record, {},
                                 geometry, band, payload, error)) return false;
    }
    session.reset();
    return true;
}

std::vector<uint32_t> read_spirv(const char * path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) return {};
    const std::streamsize size = input.tellg();
    if (size <= 0 || size % static_cast<std::streamsize>(sizeof(uint32_t)) != 0) return {};
    std::vector<uint32_t> code(static_cast<size_t>(size) / sizeof(uint32_t));
    input.seekg(0);
    input.read(reinterpret_cast<char *>(code.data()), size);
    return input ? code : std::vector<uint32_t>();
}

bool stream_dispatch_d2(VkPhysicalDevice physical_device, VkDevice device, VkQueue queue,
                        uint32_t queue_family, const char * shader_path) {
    const astc_vulkan_footprint footprint = astc_vulkan_footprint::k8x5;
    constexpr uint32_t width = 16;
    constexpr uint32_t logical_height = 21;
    constexpr uint32_t samples = 2;
    const std::vector<uint32_t> spirv = read_spirv(shader_path);
    if (spirv.empty()) return false;
    astc_vulkan_stream_geometry geometry;
    std::string error;
    if (!astc_vulkan_make_stream_geometry(footprint, width, logical_height, true,
                                           geometry, error)) return false;
    std::vector<astc_vulkan_stream_band> bands;
    if (!astc_vulkan_plan_stream(geometry, geometry.blocks_x * 16u, bands, error)) return false;
    constexpr uint8_t constant_half_block[16] = {
        0xfc, 0xfd, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80};
    std::vector<uint8_t> payload(static_cast<size_t>(geometry.payload_bytes));
    for (size_t offset = 0; offset < payload.size(); offset += sizeof(constant_half_block)) {
        std::memcpy(payload.data() + offset, constant_half_block, sizeof(constant_half_block));
    }
    std::vector<uint32_t> layout(static_cast<size_t>(astc_vulkan_paired_layout_bytes(
        footprint, width, logical_height) / sizeof(uint32_t)), 0u);
    astc_vulkan_tensor_record record{
        "stream-dispatch-d2", width, logical_height, footprint, 0, geometry.payload_bytes};
    record.representation = astc_vulkan_representation::kPairedD2;
    astc_vulkan_tensor_session tensor;
    const astc_vulkan_reconstruction reconstruction{};
    if (!tensor.upload(physical_device, device, queue, queue_family, record,
                       reconstruction, payload, error)) return false;
    astc_vulkan_paired_matvec_session dispatch;
    if (!dispatch.init(physical_device, device, queue, queue_family, tensor,
                       std::vector<uint8_t>(reinterpret_cast<uint8_t *>(layout.data()),
                                            reinterpret_cast<uint8_t *>(layout.data()) + layout.size() * sizeof(uint32_t)),
                       spirv, width, logical_height, samples, error,
                       astc_vulkan_paired_semantic::direct_rgb)) return false;
    std::vector<float> activations(static_cast<size_t>(samples) * width);
    for (size_t i = 0; i < activations.size(); ++i) activations[i] = 0.01f * static_cast<float>(i + 1);
    std::vector<float> full_output;
    if (!dispatch.run(activations, reconstruction, full_output, error)) return false;
    std::vector<float> streamed(static_cast<size_t>(samples) * logical_height, 0.0f);
    for (const auto & band : bands) {
        std::vector<uint8_t> band_payload(
            payload.begin() + static_cast<size_t>(band.payload_offset),
            payload.begin() + static_cast<size_t>(band.payload_offset + band.payload_size));
        if (!tensor.upload_band(physical_device, device, queue, queue_family, record,
                                reconstruction, geometry, band, band_payload, error) ||
            !dispatch.rebind_texture(tensor, band.physical_height, band.physical_y, error)) {
            return false;
        }
        std::vector<float> band_output;
        if (!dispatch.run_band(activations, reconstruction, band.logical_row_base,
                               band.logical_row_count, band_output, error) ||
            band_output.size() != static_cast<size_t>(samples) * band.logical_row_count) {
            return false;
        }
        for (uint32_t sample = 0; sample < samples; ++sample) {
            std::memcpy(streamed.data() + static_cast<size_t>(sample) * logical_height + band.logical_row_base,
                        band_output.data() + static_cast<size_t>(sample) * band.logical_row_count,
                        static_cast<size_t>(band.logical_row_count) * sizeof(float));
        }
    }
    if (full_output.size() != streamed.size()) return false;
    for (size_t i = 0; i < full_output.size(); ++i) {
        if (std::fabs(full_output[i] - streamed[i]) > 1.0e-5f) return false;
    }
    return true;
}

} // namespace

int main() {
    const VkApplicationInfo application_info{
        VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr, "astc-vulkan-device-smoke", 1,
        "llama.cpp ASTC Vulkan PoC", 1, VK_API_VERSION_1_0,
    };
    const VkInstanceCreateInfo instance_info{
        VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, nullptr, 0, &application_info,
        0, nullptr, 0, nullptr,
    };

    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&instance_info, nullptr, &instance) != VK_SUCCESS) {
        std::fprintf(stderr, "ASTC device smoke skipped: cannot create Vulkan instance\n");
        return 77;
    }
    uint32_t device_count = 0;
    if (vkEnumeratePhysicalDevices(instance, &device_count, nullptr) != VK_SUCCESS || device_count == 0) {
        vkDestroyInstance(instance, nullptr);
        std::fprintf(stderr, "ASTC device smoke skipped: no physical device\n");
        return 77;
    }
    std::vector<VkPhysicalDevice> devices(device_count);
    if (vkEnumeratePhysicalDevices(instance, &device_count, devices.data()) != VK_SUCCESS) {
        vkDestroyInstance(instance, nullptr);
        return 77;
    }

    VkPhysicalDevice selected_device = VK_NULL_HANDLE;
    uint32_t selected_queue_family = UINT32_MAX;
    bool supports_6x5 = false;
    bool supports_8x5 = false;
    bool supports_8x6 = false;
    bool supports_10x6 = false;
    bool supports_8x8 = false;
    bool supports_10x8 = false;
    for (VkPhysicalDevice device : devices) {
        const VkFormatFeatureFlags required =
            VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
        if (!format_supports(device, VK_FORMAT_ASTC_4x4_UNORM_BLOCK, required) ||
            !format_supports(device, VK_FORMAT_ASTC_5x5_UNORM_BLOCK, required) ||
            !format_supports(device, VK_FORMAT_ASTC_6x6_UNORM_BLOCK, required)) {
            continue;
        }
        supports_6x5 = format_supports(device, VK_FORMAT_ASTC_6x5_UNORM_BLOCK, required);
        supports_8x5 = format_supports(device, VK_FORMAT_ASTC_8x5_UNORM_BLOCK, required);
        supports_8x6 = format_supports(device, VK_FORMAT_ASTC_8x6_UNORM_BLOCK, required);
        supports_10x6 = format_supports(device, VK_FORMAT_ASTC_10x6_UNORM_BLOCK, required);
        supports_8x8 = format_supports(device, VK_FORMAT_ASTC_8x8_UNORM_BLOCK, required);
        supports_10x8 = format_supports(device, VK_FORMAT_ASTC_10x8_UNORM_BLOCK, required);
        uint32_t queue_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_count, nullptr);
        std::vector<VkQueueFamilyProperties> queues(queue_count);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_count, queues.data());
        for (uint32_t i = 0; i < queue_count; ++i) {
            if ((queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
                selected_device = device;
                selected_queue_family = i;
                break;
            }
        }
        if (selected_device != VK_NULL_HANDLE) {
            break;
        }
    }
    if (selected_device == VK_NULL_HANDLE) {
        vkDestroyInstance(instance, nullptr);
        std::fprintf(stderr, "ASTC device smoke skipped: no compatible device\n");
        return 77;
    }

    constexpr float queue_priority = 1.0f;
    const VkDeviceQueueCreateInfo queue_info{
        VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0,
        selected_queue_family, 1, &queue_priority,
    };
    const VkDeviceCreateInfo device_info{
        VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, nullptr, 0, 1, &queue_info,
        0, nullptr, 0, nullptr, nullptr,
    };
    VkDevice device = VK_NULL_HANDLE;
    if (vkCreateDevice(selected_device, &device_info, nullptr, &device) != VK_SUCCESS) {
        vkDestroyInstance(instance, nullptr);
        std::fprintf(stderr, "ASTC device smoke skipped: cannot create logical device\n");
        return 77;
    }
    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, selected_queue_family, 0, &queue);
    const bool success_4x4 = repeat_create_and_upload_image(
        selected_device, device, queue, selected_queue_family,
        VK_FORMAT_ASTC_4x4_UNORM_BLOCK, { 4, 4, 1 }, 3);
    const bool success_5x5 = create_and_upload_image(
        selected_device, device, queue, selected_queue_family,
        VK_FORMAT_ASTC_5x5_UNORM_BLOCK, { 5, 5, 1 });
    const bool success_6x6 = create_and_upload_image(
        selected_device, device, queue, selected_queue_family,
        VK_FORMAT_ASTC_6x6_UNORM_BLOCK, { 6, 6, 1 });
    const bool success_6x5 = !supports_6x5 || create_and_upload_image(
        selected_device, device, queue, selected_queue_family,
        VK_FORMAT_ASTC_6x5_UNORM_BLOCK, { 6, 5, 1 });
    const bool success_8x5 = !supports_8x5 || create_and_upload_image(
        selected_device, device, queue, selected_queue_family,
        VK_FORMAT_ASTC_8x5_UNORM_BLOCK, { 8, 5, 1 });
    const bool success_8x6 = !supports_8x6 || repeat_create_and_upload_image(
        selected_device, device, queue, selected_queue_family,
        VK_FORMAT_ASTC_8x6_UNORM_BLOCK, { 8, 6, 1 }, 3);
    const bool success_10x6 = !supports_10x6 || create_and_upload_image(
        selected_device, device, queue, selected_queue_family,
        VK_FORMAT_ASTC_10x6_UNORM_BLOCK, { 10, 6, 1 });
    const bool success_8x8 = !supports_8x8 || create_and_upload_image(
        selected_device, device, queue, selected_queue_family,
        VK_FORMAT_ASTC_8x8_UNORM_BLOCK, { 8, 8, 1 });
    const bool success_10x8 = !supports_10x8 || create_and_upload_image(
        selected_device, device, queue, selected_queue_family,
        VK_FORMAT_ASTC_10x8_UNORM_BLOCK, { 10, 8, 1 });
    const bool success_stream_d1 = stream_upload_bands(
        selected_device, device, queue, selected_queue_family,
        astc_vulkan_footprint::k8x6, 16, 13, false);
    const bool success_stream_d2 = !supports_8x5 || stream_upload_bands(
        selected_device, device, queue, selected_queue_family,
        astc_vulkan_footprint::k8x5, 16, 21, true);
    const bool success_dispatch_d2 = !supports_8x5 || stream_dispatch_d2(
        selected_device, device, queue, selected_queue_family,
#ifdef ASTC_VULKAN_PAIRED_SHADER_PATH
        ASTC_VULKAN_PAIRED_SHADER_PATH
#else
        "pocs/astc-vulkan/astc-paired-matvec.comp.spv"
#endif
    );
    vkDeviceWaitIdle(device);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    if (!success_4x4 || !success_5x5 || !success_6x6 || !success_6x5 || !success_8x5 || !success_8x6 || !success_10x6 || !success_8x8 || !success_10x8 || !success_stream_d1 || !success_stream_d2 || !success_dispatch_d2) {
        std::fprintf(stderr, "ASTC device smoke failed: image upload or layout transition failed\n");
        return 1;
    }
    std::printf("ASTC 4x4, 5x5, and 6x6 image resource smoke passed; streamed D1=passed, streamed D2=%s, paired D2 dispatch=%s; experimental 6x5=%s, 8x5=%s, 8x6=%s, 10x6=%s, 8x8=%s, 10x8=%s\n",
                supports_8x5 ? "passed" : "unsupported",
                supports_8x5 ? "passed" : "unsupported",
                supports_6x5 ? "passed" : "unsupported",
                supports_8x5 ? "passed" : "unsupported",
                supports_8x6 ? "passed" : "unsupported",
                supports_10x6 ? "passed" : "unsupported",
                supports_8x8 ? "passed" : "unsupported",
                supports_10x8 ? "passed" : "unsupported");
    return 0;
}
