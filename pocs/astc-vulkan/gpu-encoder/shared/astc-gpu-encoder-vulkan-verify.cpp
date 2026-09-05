#include "astc-gpu-encoder-vulkan-verify.h"

#include "astc-vulkan-resource.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>

namespace {

std::vector<uint32_t> read_spirv(const std::string & path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return {};
    const auto bytes = file.tellg();
    if (bytes <= 0 || bytes % static_cast<std::streamoff>(sizeof(uint32_t)) != 0) return {};
    std::vector<uint32_t> result(static_cast<size_t>(bytes) / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char *>(result.data()), bytes);
    return file ? result : std::vector<uint32_t>{};
}

bool select_sampled_compute_device(VkInstance instance, VkFormat format, uint32_t width, uint32_t height,
                                   VkPhysicalDevice & physical, uint32_t & family) {
    uint32_t count = 0;
    if (vkEnumeratePhysicalDevices(instance, &count, nullptr) != VK_SUCCESS || count == 0) return false;
    std::vector<VkPhysicalDevice> devices(count);
    if (vkEnumeratePhysicalDevices(instance, &count, devices.data()) != VK_SUCCESS) return false;
    for (uint32_t pass = 0; pass < 2; ++pass) for (const auto device : devices) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(device, &properties);
        if ((pass == 0) != (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)) continue;
        if (!astc_vulkan_supports_sampled_transfer_extent(device, format, width, height)) continue;
        uint32_t queue_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_count, nullptr);
        std::vector<VkQueueFamilyProperties> queues(queue_count);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_count, queues.data());
        for (uint32_t index = 0; index < queue_count; ++index) if (queues[index].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            physical = device;
            family = index;
            return true;
        }
    }
    return false;
}

bool create_host_output(VkPhysicalDevice physical, VkDevice device, VkDeviceSize bytes,
                        VkBuffer & buffer, VkDeviceMemory & memory) {
    const VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, nullptr};
    if (vkCreateBuffer(device, &info, nullptr, &buffer) != VK_SUCCESS) return false;
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device, buffer, &requirements);
    const uint32_t type = astc_vulkan_find_memory_type(
        physical, requirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (type == std::numeric_limits<uint32_t>::max()) return false;
    const VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr,
        requirements.size, type};
    return vkAllocateMemory(device, &allocation, nullptr, &memory) == VK_SUCCESS &&
           vkBindBufferMemory(device, buffer, memory, 0) == VK_SUCCESS;
}

} // namespace

bool astc_gpu_encoder_verify_d1_vulkan_decode_default(
    const std::string & validation_spirv_path,
    const std::vector<astc_gpu_encoder_finished_block> & finished,
    uint32_t blocks_x, double & mse, float & max_abs, std::string & error) {
    mse = 0.0;
    max_abs = 0.0f;
    if (finished.empty() || blocks_x == 0 || finished.size() % blocks_x != 0) {
        error = "invalid D1 4x4 finished payload raster";
        return false;
    }
    const astc_vulkan_footprint footprint = finished.front().footprint;
    const auto format_info = astc_vulkan_format(footprint);
    const VkFormat vk_format = astc_vulkan_vk_format(static_cast<uint8_t>(footprint));
    if (format_info.block_width == 0 || format_info.block_height == 0 || vk_format == VK_FORMAT_UNDEFINED) {
        error = "unsupported D1 finished footprint";
        return false;
    }
    const auto spirv = read_spirv(validation_spirv_path);
    if (spirv.empty()) { error = "invalid ASTC validation SPIR-V"; return false; }
    const uint32_t blocks_y = static_cast<uint32_t>(finished.size()) / blocks_x;
    const uint32_t width = blocks_x * format_info.block_width;
    const uint32_t height = blocks_y * format_info.block_height;
    std::vector<uint8_t> payload(finished.size() * 16);
    std::vector<float> expected(size_t(width) * height * 4);
    for (uint32_t by = 0; by < blocks_y; ++by) for (uint32_t bx = 0; bx < blocks_x; ++bx) {
        const auto & block = finished[size_t(by) * blocks_x + bx];
        if (block.footprint != footprint ||
            block.decoded_rgba.size() != size_t(format_info.block_width) * format_info.block_height * 4) {
            error = "D1 Vulkan verifier received incompatible finished block";
            return false;
        }
        std::memcpy(payload.data() + (size_t(by) * blocks_x + bx) * 16, block.payload.data(), 16);
        for (uint32_t y = 0; y < format_info.block_height; ++y)
            for (uint32_t x = 0; x < format_info.block_width; ++x)
                for (uint32_t channel = 0; channel < 4; ++channel) {
            expected[((size_t(by * format_info.block_height + y) * width +
                       bx * format_info.block_width + x) * 4) + channel] =
                block.decoded_rgba[(y * format_info.block_width + x) * 4 + channel];
        }
    }
    const VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr,
        "astc-gpu-encoder-vulkan-verify", 1, "llama.cpp", 1, VK_API_VERSION_1_1};
    const VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, nullptr,
        0, &app, 0, nullptr, 0, nullptr};
    VkInstance instance = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    uint32_t family = UINT32_MAX;
    VkQueue queue = VK_NULL_HANDLE;
    VkBuffer output_buffer = VK_NULL_HANDLE;
    VkDeviceMemory output_memory = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkShaderModule shader_module = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    astc_vulkan_texture texture;
    bool ok = vkCreateInstance(&instance_info, nullptr, &instance) == VK_SUCCESS &&
              select_sampled_compute_device(instance, vk_format, width, height, physical, family);
    constexpr float priority = 1.0f;
    const VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr,
        0, family, 1, &priority};
    const VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, nullptr,
        0, 1, &queue_info, 0, nullptr, 0, nullptr, nullptr};
    if (ok) ok = vkCreateDevice(physical, &device_info, nullptr, &device) == VK_SUCCESS;
    if (ok) vkGetDeviceQueue(device, family, 0, &queue);
    if (ok) ok = texture.upload(physical, device, queue, family,
                                static_cast<uint8_t>(footprint),
                                width, height, payload, error);
    const VkDeviceSize output_bytes = VkDeviceSize(expected.size()) * sizeof(float);
    if (ok) ok = create_host_output(physical, device, output_bytes, output_buffer, output_memory);
    const VkDescriptorSetLayoutBinding bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
    const VkDescriptorSetLayoutCreateInfo descriptor_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        nullptr, 0, 2, bindings};
    const VkDescriptorPoolSize pool_sizes[] = {
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}};
    const VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        nullptr, 0, 1, 2, pool_sizes};
    if (ok) ok = vkCreateDescriptorSetLayout(device, &descriptor_info, nullptr, &descriptor_layout) == VK_SUCCESS &&
                 vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool) == VK_SUCCESS;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    const VkDescriptorSetAllocateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        nullptr, descriptor_pool, 1, &descriptor_layout};
    if (ok) ok = vkAllocateDescriptorSets(device, &set_info, &descriptor_set) == VK_SUCCESS;
    const VkDescriptorImageInfo image_info{texture.sampler(), texture.view(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    const VkDescriptorBufferInfo buffer_info{output_buffer, 0, output_bytes};
    VkWriteDescriptorSet writes[2]{};
    writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor_set, 0, 0, 1,
                 VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &image_info, nullptr, nullptr};
    writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor_set, 1, 0, 1,
                 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &buffer_info, nullptr};
    if (ok) vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
    const VkShaderModuleCreateInfo shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr,
        0, spirv.size() * sizeof(uint32_t), spirv.data()};
    const VkPushConstantRange push_range{VK_SHADER_STAGE_COMPUTE_BIT, 0, 12};
    const VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        nullptr, 0, 1, &descriptor_layout, 1, &push_range};
    if (ok) ok = vkCreateShaderModule(device, &shader_info, nullptr, &shader_module) == VK_SUCCESS &&
                 vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout) == VK_SUCCESS;
    const VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, shader_module, "main", nullptr};
    const VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        nullptr, 0, stage, pipeline_layout, VK_NULL_HANDLE, -1};
    if (ok) ok = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) == VK_SUCCESS;
    const VkCommandPoolCreateInfo command_pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        nullptr, 0, family};
    const VkCommandBufferAllocateInfo command_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        nullptr, command_pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    const VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, 0};
    if (ok) ok = vkCreateCommandPool(device, &command_pool_info, nullptr, &command_pool) == VK_SUCCESS;
    if (ok) {
        const VkCommandBufferAllocateInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            nullptr, command_pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
        ok = vkAllocateCommandBuffers(device, &info, &command) == VK_SUCCESS &&
             vkCreateFence(device, &fence_info, nullptr, &fence) == VK_SUCCESS;
    }
    const VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
    const uint32_t push[] = {width, height, 0};
    if (ok) ok = vkBeginCommandBuffer(command, &begin) == VK_SUCCESS;
    if (ok) {
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout,
                                0, 1, &descriptor_set, 0, nullptr);
        vkCmdPushConstants(command, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);
        vkCmdDispatch(command, (width * height + 63) / 64, 1, 1);
        ok = vkEndCommandBuffer(command) == VK_SUCCESS;
    }
    const VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &command, 0, nullptr};
    if (ok) ok = vkQueueSubmit(queue, 1, &submit, fence) == VK_SUCCESS &&
                 vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX) == VK_SUCCESS;
    std::vector<float> actual(expected.size());
    if (ok) {
        void * mapped = nullptr;
        ok = vkMapMemory(device, output_memory, 0, output_bytes, 0, &mapped) == VK_SUCCESS;
        if (ok) { std::memcpy(actual.data(), mapped, actual.size() * sizeof(float)); vkUnmapMemory(device, output_memory); }
    }
    if (ok) {
        double total = 0.0;
        for (size_t index = 0; index < actual.size(); ++index) {
            const float difference = std::fabs(actual[index] - expected[index]);
            total += static_cast<double>(difference) * difference;
            max_abs = std::max(max_abs, difference);
        }
        mse = total / actual.size();
        ok = std::isfinite(mse) && mse <= 1e-8 && max_abs <= 1e-3f;
        if (!ok) error = "CPU/Vulkan ASTC decode mismatch";
    }
    if (device != VK_NULL_HANDLE) vkDeviceWaitIdle(device);
    if (fence) vkDestroyFence(device, fence, nullptr);
    if (command_pool) vkDestroyCommandPool(device, command_pool, nullptr);
    if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
    if (shader_module) vkDestroyShaderModule(device, shader_module, nullptr);
    if (pipeline_layout) vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    if (descriptor_pool) vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
    if (descriptor_layout) vkDestroyDescriptorSetLayout(device, descriptor_layout, nullptr);
    if (output_buffer) vkDestroyBuffer(device, output_buffer, nullptr);
    if (output_memory) vkFreeMemory(device, output_memory, nullptr);
    texture.reset();
    if (device) vkDestroyDevice(device, nullptr);
    if (instance) vkDestroyInstance(instance, nullptr);
    if (!ok && error.empty()) error = "failed to run Vulkan ASTC decode verification";
    return ok;
}
