#include <vulkan/vulkan.h>

#include "astc-vulkan-contract.h"

#include <cmath>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace {

constexpr VkDeviceSize kAstcBlockBytes = ggml_vk_astc_4x4_unorm_rgba.block_size_bytes;
constexpr float kExpectedChannel = 0.5f;
constexpr uint32_t kBenchmarkTexelExtent = 192;
constexpr uint32_t kBenchmarkDispatchRepeats = 20;

// A valid 2D LDR ASTC void-extent block.  Bits 10 and 11 are set as required;
// all-ones s/t extents make this a constant-color block.  Each RGBA component
// is UNORM16(0x8000), so texelFetch should return approximately 0.5.
constexpr unsigned char kConstantHalfBlock[kAstcBlockBytes] = {
    0xfc, 0xfd, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80,
};

struct image_resources {
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

bool supports_format(VkPhysicalDevice device, VkFormat format) {
    VkFormatProperties properties{};
    vkGetPhysicalDeviceFormatProperties(device, format, &properties);
    constexpr VkFormatFeatureFlags required =
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    return (properties.optimalTilingFeatures & required) == required;
}

std::vector<uint32_t> read_spirv(const char * path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        return {};
    }
    const std::streamsize size = input.tellg();
    if (size <= 0 || size % sizeof(uint32_t) != 0) {
        return {};
    }
    std::vector<uint32_t> code(static_cast<size_t>(size) / sizeof(uint32_t));
    input.seekg(0);
    input.read(reinterpret_cast<char *>(code.data()), size);
    return input ? code : std::vector<uint32_t>();
}

bool create_image(VkPhysicalDevice physical_device, VkDevice device, VkFormat format,
                  uint32_t width, uint32_t height, image_resources & resources) {
    const VkImageCreateInfo image_info{
        VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, nullptr, 0, VK_IMAGE_TYPE_2D,
        format, { width, height, 1 }, 1, 1, VK_SAMPLE_COUNT_1_BIT,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_SHARING_MODE_EXCLUSIVE, 0, nullptr, VK_IMAGE_LAYOUT_UNDEFINED,
    };
    if (vkCreateImage(device, &image_info, nullptr, &resources.image) != VK_SUCCESS) {
        return false;
    }
    VkMemoryRequirements requirements{};
    vkGetImageMemoryRequirements(device, resources.image, &requirements);
    const uint32_t memory_type = find_memory_type(
        physical_device, requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (memory_type == UINT32_MAX) {
        return false;
    }
    const VkMemoryAllocateInfo allocate_info{
        VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, requirements.size, memory_type,
    };
    if (vkAllocateMemory(device, &allocate_info, nullptr, &resources.memory) != VK_SUCCESS ||
        vkBindImageMemory(device, resources.image, resources.memory, 0) != VK_SUCCESS) {
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
    return vkCreateSampler(device, &sampler_info, nullptr, &resources.sampler) == VK_SUCCESS;
}

void destroy_image(VkDevice device, image_resources & resources) {
    if (resources.sampler != VK_NULL_HANDLE) {
        vkDestroySampler(device, resources.sampler, nullptr);
    }
    if (resources.view != VK_NULL_HANDLE) {
        vkDestroyImageView(device, resources.view, nullptr);
    }
    if (resources.memory != VK_NULL_HANDLE) {
        vkFreeMemory(device, resources.memory, nullptr);
    }
    if (resources.image != VK_NULL_HANDLE) {
        vkDestroyImage(device, resources.image, nullptr);
    }
    resources = {};
}

} // namespace

int main(int argc, char ** argv) {
    const std::string format_name = argc >= 3 ? argv[2] : "";
    const std::string pattern_name = argc == 4 ? argv[3] : "sequential";
    const bool benchmark = argc == 5 && std::string(argv[4]) == "--benchmark";
    if ((argc < 3 || argc > 5) ||
        (format_name != "4x4" && format_name != "5x5" && format_name != "6x6") ||
        (pattern_name != "sequential" && pattern_name != "nonlocal")) {
        std::fprintf(stderr,
                     "usage: %s <validation.spv> <4x4|5x5|6x6> "
                     "[sequential|nonlocal] [--benchmark]\n",
                     argv[0]);
        return 2;
    }
    if (argc == 5 && !benchmark) {
        std::fprintf(stderr, "unknown option: %s\n", argv[4]);
        return 2;
    }
    const uint32_t access_pattern = pattern_name == "nonlocal" ? 1u : 0u;
    const std::vector<uint32_t> spirv = read_spirv(argv[1]);
    if (spirv.empty()) {
        std::fprintf(stderr, "ASTC shader smoke skipped: SPIR-V file unavailable\n");
        return 77;
    }

    const VkApplicationInfo application_info{
        VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr, "astc-vulkan-shader-device-smoke", 1,
        "llama.cpp ASTC Vulkan PoC", 1, VK_API_VERSION_1_0,
    };
    const VkInstanceCreateInfo instance_info{
        VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, nullptr, 0, &application_info,
        0, nullptr, 0, nullptr,
    };
    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&instance_info, nullptr, &instance) != VK_SUCCESS) {
        return 77;
    }
    uint32_t device_count = 0;
    if (vkEnumeratePhysicalDevices(instance, &device_count, nullptr) != VK_SUCCESS || device_count == 0) {
        vkDestroyInstance(instance, nullptr);
        return 77;
    }
    std::vector<VkPhysicalDevice> devices(device_count);
    vkEnumeratePhysicalDevices(instance, &device_count, devices.data());
    const VkFormat format = format_name == "4x4" ? VK_FORMAT_ASTC_4x4_UNORM_BLOCK :
                            format_name == "5x5" ? VK_FORMAT_ASTC_5x5_UNORM_BLOCK :
                                                   VK_FORMAT_ASTC_6x6_UNORM_BLOCK;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    uint32_t queue_family = UINT32_MAX;
    uint32_t timestamp_valid_bits = 0;
    for (VkPhysicalDevice device : devices) {
        if (!supports_format(device, format)) {
            continue;
        }
        uint32_t queue_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_count, nullptr);
        std::vector<VkQueueFamilyProperties> queues(queue_count);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_count, queues.data());
        for (uint32_t i = 0; i < queue_count; ++i) {
            if ((queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
                physical_device = device;
                queue_family = i;
                timestamp_valid_bits = queues[i].timestampValidBits;
                break;
            }
        }
        if (physical_device != VK_NULL_HANDLE) {
            break;
        }
    }
    if (physical_device == VK_NULL_HANDLE) {
        vkDestroyInstance(instance, nullptr);
        return 77;
    }

    constexpr float queue_priority = 1.0f;
    const VkDeviceQueueCreateInfo queue_info{
        VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0, queue_family, 1,
        &queue_priority,
    };
    const VkDeviceCreateInfo device_info{
        VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, nullptr, 0, 1, &queue_info,
        0, nullptr, 0, nullptr, nullptr,
    };
    VkDevice device = VK_NULL_HANDLE;
    if (vkCreateDevice(physical_device, &device_info, nullptr, &device) != VK_SUCCESS) {
        vkDestroyInstance(instance, nullptr);
        return 77;
    }
    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, queue_family, 0, &queue);

    const uint32_t block_extent = format_name == "4x4" ? 4 : format_name == "5x5" ? 5 : 6;
    const uint32_t width = benchmark ? kBenchmarkTexelExtent : block_extent;
    const uint32_t height = width;
    const uint64_t block_count = ggml_vk_astc_image_block_count(
        format_name == "4x4" ? ggml_vk_astc_4x4_unorm_rgba :
        format_name == "5x5" ? ggml_vk_astc_5x5_unorm_rgba : ggml_vk_astc_6x6_unorm_rgba,
        width, height);
    const VkDeviceSize staging_bytes = block_count * kAstcBlockBytes;
    const uint32_t dispatch_repeats = benchmark ? kBenchmarkDispatchRepeats : 1;
    image_resources image;
    bool success = create_image(physical_device, device, format, width, height, image);
    VkBuffer staging_buffer = VK_NULL_HANDLE;
    VkDeviceMemory staging_memory = VK_NULL_HANDLE;
    VkBuffer output_buffer = VK_NULL_HANDLE;
    VkDeviceMemory output_memory = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkShaderModule shader_module = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkQueryPool query_pool = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    do {
        if (!success) break;
        const VkBufferCreateInfo staging_info{
            VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, staging_bytes,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, nullptr,
        };
        if (vkCreateBuffer(device, &staging_info, nullptr, &staging_buffer) != VK_SUCCESS) break;
        VkMemoryRequirements staging_requirements{};
        vkGetBufferMemoryRequirements(device, staging_buffer, &staging_requirements);
        const uint32_t staging_type = find_memory_type(
            physical_device, staging_requirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (staging_type == UINT32_MAX) break;
        const VkMemoryAllocateInfo staging_allocate{
            VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, staging_requirements.size, staging_type,
        };
        if (vkAllocateMemory(device, &staging_allocate, nullptr, &staging_memory) != VK_SUCCESS ||
            vkBindBufferMemory(device, staging_buffer, staging_memory, 0) != VK_SUCCESS) break;
        void * mapped = nullptr;
        if (vkMapMemory(device, staging_memory, 0, staging_bytes, 0, &mapped) != VK_SUCCESS) break;
        for (uint64_t block = 0; block < block_count; ++block) {
            std::memcpy(static_cast<unsigned char *>(mapped) + block * kAstcBlockBytes,
                        kConstantHalfBlock, kAstcBlockBytes);
        }
        vkUnmapMemory(device, staging_memory);

        const VkDeviceSize output_bytes = static_cast<VkDeviceSize>(width) * height * 4 * sizeof(float);
        const VkBufferCreateInfo output_info{
            VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, output_bytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, nullptr,
        };
        if (vkCreateBuffer(device, &output_info, nullptr, &output_buffer) != VK_SUCCESS) break;
        VkMemoryRequirements output_requirements{};
        vkGetBufferMemoryRequirements(device, output_buffer, &output_requirements);
        const uint32_t output_type = find_memory_type(
            physical_device, output_requirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (output_type == UINT32_MAX) break;
        const VkMemoryAllocateInfo output_allocate{
            VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, output_requirements.size, output_type,
        };
        if (vkAllocateMemory(device, &output_allocate, nullptr, &output_memory) != VK_SUCCESS ||
            vkBindBufferMemory(device, output_buffer, output_memory, 0) != VK_SUCCESS) break;

        const VkDescriptorSetLayoutBinding bindings[2] = {
            { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        };
        const VkDescriptorSetLayoutCreateInfo descriptor_info{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 2, bindings,
        };
        if (vkCreateDescriptorSetLayout(device, &descriptor_info, nullptr, &descriptor_layout) != VK_SUCCESS) break;
        const VkDescriptorPoolSize pool_sizes[2] = {
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 },
        };
        const VkDescriptorPoolCreateInfo pool_info{
            VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0, 1, 2, pool_sizes,
        };
        if (vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool) != VK_SUCCESS) break;
        const VkDescriptorSetAllocateInfo set_info{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, descriptor_pool, 1,
            &descriptor_layout,
        };
        if (vkAllocateDescriptorSets(device, &set_info, &descriptor_set) != VK_SUCCESS) break;
        const VkDescriptorImageInfo image_descriptor{ image.sampler, image.view,
                                                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        const VkDescriptorBufferInfo buffer_descriptor{ output_buffer, 0, output_bytes };
        const VkWriteDescriptorSet writes[2] = {
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor_set, 0, 0, 1,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &image_descriptor, nullptr, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor_set, 1, 0, 1,
              VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &buffer_descriptor, nullptr },
        };
        vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);

        const VkShaderModuleCreateInfo shader_info{
            VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
            spirv.size() * sizeof(uint32_t), spirv.data(),
        };
        if (vkCreateShaderModule(device, &shader_info, nullptr, &shader_module) != VK_SUCCESS) break;
        const VkPushConstantRange push_constants{
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t) * 3,
        };
        const VkPipelineLayoutCreateInfo layout_info{
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1,
            &descriptor_layout, 1, &push_constants,
        };
        if (vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) break;
        const VkPipelineShaderStageCreateInfo stage_info{
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
            VK_SHADER_STAGE_COMPUTE_BIT, shader_module, "main", nullptr,
        };
        const VkComputePipelineCreateInfo pipeline_info{
            VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, nullptr, 0, stage_info,
            pipeline_layout, VK_NULL_HANDLE, -1,
        };
        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info,
                                     nullptr, &pipeline) != VK_SUCCESS) break;

        const VkQueryPoolCreateInfo query_info{
            VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, nullptr, 0,
            VK_QUERY_TYPE_TIMESTAMP, 2, 0,
        };
        if (timestamp_valid_bits != 0 &&
            vkCreateQueryPool(device, &query_info, nullptr, &query_pool) != VK_SUCCESS) break;

        const VkCommandPoolCreateInfo command_pool_info{
            VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr,
            VK_COMMAND_POOL_CREATE_TRANSIENT_BIT, queue_family,
        };
        if (vkCreateCommandPool(device, &command_pool_info, nullptr, &command_pool) != VK_SUCCESS) break;
        const VkCommandBufferAllocateInfo command_allocate{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, command_pool,
            VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1,
        };
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        if (vkAllocateCommandBuffers(device, &command_allocate, &command_buffer) != VK_SUCCESS) break;
        const VkCommandBufferBeginInfo begin_info{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
            VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr,
        };
        if (vkBeginCommandBuffer(command_buffer, &begin_info) != VK_SUCCESS) break;
        if (query_pool != VK_NULL_HANDLE) vkCmdResetQueryPool(command_buffer, query_pool, 0, 2);
        const VkImageMemoryBarrier to_transfer{
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr, 0,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED, image.image, { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
        };
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &to_transfer);
        const VkBufferImageCopy copy_region{
            0, 0, 0, { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, { 0, 0, 0 }, { width, height, 1 },
        };
        vkCmdCopyBufferToImage(command_buffer, staging_buffer, image.image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy_region);
        const VkImageMemoryBarrier to_shader{
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, image.image,
            { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
        };
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &to_shader);
        if (query_pool != VK_NULL_HANDLE) {
            vkCmdWriteTimestamp(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                query_pool, 0);
        }
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);
        const uint32_t dimensions[3] = { width, height, access_pattern };
        vkCmdPushConstants(command_buffer, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(dimensions), dimensions);
        const uint32_t workgroup_count = (width * height + 63) / 64;
        for (uint32_t repeat = 0; repeat < dispatch_repeats; ++repeat) {
            vkCmdDispatch(command_buffer, workgroup_count, 1, 1);
        }
        if (query_pool != VK_NULL_HANDLE) {
            vkCmdWriteTimestamp(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                query_pool, 1);
        }
        const VkBufferMemoryBarrier output_barrier{
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, nullptr,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, output_buffer, 0,
            output_bytes,
        };
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &output_barrier, 0, nullptr);
        if (vkEndCommandBuffer(command_buffer) != VK_SUCCESS) break;
        const VkSubmitInfo submit_info{
            VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1,
            &command_buffer, 0, nullptr,
        };
        const VkFenceCreateInfo fence_info{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, 0 };
        if (vkCreateFence(device, &fence_info, nullptr, &fence) != VK_SUCCESS ||
            vkQueueSubmit(queue, 1, &submit_info, fence) != VK_SUCCESS ||
            vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) break;
        std::vector<float> values(static_cast<size_t>(width) * height * 4);
        void * output_mapped = nullptr;
        if (vkMapMemory(device, output_memory, 0, output_bytes, 0, &output_mapped) != VK_SUCCESS) break;
        std::memcpy(values.data(), output_mapped, static_cast<size_t>(output_bytes));
        vkUnmapMemory(device, output_memory);
        for (float value : values) {
            if (!std::isfinite(value) || std::fabs(value - kExpectedChannel) > 1e-3f) {
                std::fprintf(stderr,
                             "ASTC %s %s shader smoke failed: decoded value %.7f\n",
                             format_name.c_str(), pattern_name.c_str(), value);
                success = false;
                break;
            }
        }
        if (success && query_pool != VK_NULL_HANDLE) {
            uint64_t timestamps[2] = {};
            const VkResult query_result = vkGetQueryPoolResults(
                device, query_pool, 0, 2, sizeof(timestamps), timestamps,
                sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            if (query_result == VK_SUCCESS) {
                VkPhysicalDeviceProperties properties{};
                vkGetPhysicalDeviceProperties(physical_device, &properties);
                const double elapsed_ns = static_cast<double>(timestamps[1] - timestamps[0]) *
                                           properties.limits.timestampPeriod;
                std::printf("ASTC %s %s shader timestamp %.3f ns\n",
                            format_name.c_str(), pattern_name.c_str(), elapsed_ns);
            }
        }
    } while (false);

    if (device != VK_NULL_HANDLE) vkDeviceWaitIdle(device);
    if (fence != VK_NULL_HANDLE) vkDestroyFence(device, fence, nullptr);
    if (query_pool != VK_NULL_HANDLE) vkDestroyQueryPool(device, query_pool, nullptr);
    if (command_pool != VK_NULL_HANDLE) vkDestroyCommandPool(device, command_pool, nullptr);
    if (pipeline != VK_NULL_HANDLE) vkDestroyPipeline(device, pipeline, nullptr);
    if (pipeline_layout != VK_NULL_HANDLE) vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    if (shader_module != VK_NULL_HANDLE) vkDestroyShaderModule(device, shader_module, nullptr);
    if (descriptor_pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
    if (descriptor_layout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device, descriptor_layout, nullptr);
    if (output_memory != VK_NULL_HANDLE) vkFreeMemory(device, output_memory, nullptr);
    if (output_buffer != VK_NULL_HANDLE) vkDestroyBuffer(device, output_buffer, nullptr);
    if (staging_memory != VK_NULL_HANDLE) vkFreeMemory(device, staging_memory, nullptr);
    if (staging_buffer != VK_NULL_HANDLE) vkDestroyBuffer(device, staging_buffer, nullptr);
    destroy_image(device, image);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    if (!success) {
        std::fprintf(stderr, "ASTC %s %s shader smoke failed\n",
                     format_name.c_str(), pattern_name.c_str());
        return 1;
    }
    std::printf("ASTC %s %s shader fetch smoke passed\n",
                format_name.c_str(), pattern_name.c_str());
    return 0;
}
