#include "astc-gpu-encoder-exact-dispatch.h"

#include "astc-vulkan-resource.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <limits>

namespace {

struct gpu_texel { float rgba[4]; };
static_assert(sizeof(gpu_texel) == 16, "GPU exact source texels must match vec4");

std::vector<uint32_t> read_spirv(const std::string & path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() <= 0 || (input.tellg() % std::streamoff(4)) != 0) return {};
    const size_t words = static_cast<size_t>(input.tellg() / std::streamoff(4));
    std::vector<uint32_t> result(words);
    input.seekg(0);
    if (!input.read(reinterpret_cast<char *>(result.data()), static_cast<std::streamsize>(words * 4))) return {};
    return result;
}

bool select_compute_device(VkInstance instance, VkPhysicalDevice & physical, uint32_t & family) {
    uint32_t count = 0;
    if (vkEnumeratePhysicalDevices(instance, &count, nullptr) != VK_SUCCESS || count == 0) return false;
    std::vector<VkPhysicalDevice> devices(count);
    if (vkEnumeratePhysicalDevices(instance, &count, devices.data()) != VK_SUCCESS) return false;
    for (uint32_t pass = 0; pass < 2; ++pass) for (const auto device : devices) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(device, &properties);
        if ((pass == 0) != (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)) continue;
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

bool create_host_buffer(VkPhysicalDevice physical, VkDevice device, VkDeviceSize size,
                        VkBuffer & buffer, VkDeviceMemory & memory) {
    const VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, nullptr};
    if (vkCreateBuffer(device, &info, nullptr, &buffer) != VK_SUCCESS) return false;
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device, buffer, &requirements);
    const uint32_t type = astc_vulkan_find_memory_type(physical, requirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (type == std::numeric_limits<uint32_t>::max()) return false;
    const VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr,
        requirements.size, type};
    return vkAllocateMemory(device, &allocation, nullptr, &memory) == VK_SUCCESS &&
        vkBindBufferMemory(device, buffer, memory, 0) == VK_SUCCESS;
}

bool map_write(VkDevice device, VkDeviceMemory memory, const void * values, VkDeviceSize bytes) {
    void * mapped = nullptr;
    if (vkMapMemory(device, memory, 0, bytes, 0, &mapped) != VK_SUCCESS) return false;
    std::memcpy(mapped, values, static_cast<size_t>(bytes));
    vkUnmapMemory(device, memory);
    return true;
}

} // namespace

bool astc_gpu_exact_subset_encode_gpu_default(
    const std::string & spirv_path,
    const astc_gpu_encoder_request & request,
    std::vector<astc_gpu_exact_subset_block> & blocks,
    std::string & error) {
    if (request.mode != astc_gpu_encode_mode::exact_subset || request.blocks.empty()) {
        error = "GPU exact subset requires non-empty exact-subset input";
        return false;
    }
    std::vector<astc_gpu_encoder_batch> batches;
    if (!astc_gpu_encoder_plan_batches(request, batches)) {
        error = "GPU exact subset requires valid source batches";
        return false;
    }
    std::vector<astc_gpu_exact_subset_block> cpu_check;
    if (!astc_gpu_exact_subset_encode_cpu(request, cpu_check)) {
        error = "GPU exact subset requires finite UNORM source texels";
        return false;
    }
    const auto format = astc_vulkan_format(request.footprint);
    const auto spirv = read_spirv(spirv_path);
    if (format.block_width == 0 || format.block_height == 0 || spirv.empty()) {
        error = "GPU exact subset has invalid footprint or SPIR-V";
        return false;
    }

    const VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr,
        "astc-gpu-exact-subset", 1, "llama.cpp", 1, VK_API_VERSION_1_1};
    const VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, nullptr, 0,
        &app, 0, nullptr, 0, nullptr};
    VkInstance instance = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkBuffer source_buffer = VK_NULL_HANDLE, payload_buffer = VK_NULL_HANDLE;
    VkDeviceMemory source_memory = VK_NULL_HANDLE, payload_memory = VK_NULL_HANDLE;
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    const auto cleanup = [&]() {
        if (device != VK_NULL_HANDLE) vkDeviceWaitIdle(device);
        if (fence) vkDestroyFence(device, fence, nullptr);
        if (command_pool) vkDestroyCommandPool(device, command_pool, nullptr);
        if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
        if (shader) vkDestroyShaderModule(device, shader, nullptr);
        if (pipeline_layout) vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
        if (pool) vkDestroyDescriptorPool(device, pool, nullptr);
        if (layout) vkDestroyDescriptorSetLayout(device, layout, nullptr);
        if (source_buffer) vkDestroyBuffer(device, source_buffer, nullptr);
        if (source_memory) vkFreeMemory(device, source_memory, nullptr);
        if (payload_buffer) vkDestroyBuffer(device, payload_buffer, nullptr);
        if (payload_memory) vkFreeMemory(device, payload_memory, nullptr);
        if (device != VK_NULL_HANDLE) vkDestroyDevice(device, nullptr);
        if (instance != VK_NULL_HANDLE) vkDestroyInstance(instance, nullptr);
    };
    if (vkCreateInstance(&instance_info, nullptr, &instance) != VK_SUCCESS) {
        error = "GPU exact subset failed to create Vulkan instance";
        cleanup();
        return false;
    }
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    uint32_t family = UINT32_MAX;
    if (!select_compute_device(instance, physical, family)) {
        error = "GPU exact subset found no Vulkan compute device";
        cleanup();
        return false;
    }
    constexpr float priority = 1.0f;
    const VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr,
        0, family, 1, &priority};
    const VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, nullptr,
        0, 1, &queue_info, 0, nullptr, 0, nullptr, nullptr};
    if (vkCreateDevice(physical, &device_info, nullptr, &device) != VK_SUCCESS) {
        error = "GPU exact subset failed to create Vulkan device";
        cleanup();
        return false;
    }
    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, family, 0, &queue);
    uint32_t capacity = 0;
    for (const auto & batch : batches) capacity = std::max(capacity, batch.block_count);
    const VkDeviceSize source_bytes = VkDeviceSize(capacity) * format.block_width * format.block_height * sizeof(gpu_texel);
    const VkDeviceSize payload_bytes = VkDeviceSize(capacity) * 16u;
    bool ok = create_host_buffer(physical, device, source_bytes, source_buffer, source_memory) &&
        create_host_buffer(physical, device, payload_bytes, payload_buffer, payload_memory);
    const VkDescriptorSetLayoutBinding bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
    const VkDescriptorSetLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        nullptr, 0, 2, bindings};
    const VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
    const VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        nullptr, 0, 1, 1, &pool_size};
    if (ok) ok = vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &layout) == VK_SUCCESS &&
        vkCreateDescriptorPool(device, &pool_info, nullptr, &pool) == VK_SUCCESS;
    VkDescriptorSet set = VK_NULL_HANDLE;
    const VkDescriptorSetAllocateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        nullptr, pool, 1, &layout};
    if (ok) ok = vkAllocateDescriptorSets(device, &set_info, &set) == VK_SUCCESS;
    const VkDescriptorBufferInfo source_info{source_buffer, 0, source_bytes};
    const VkDescriptorBufferInfo payload_info{payload_buffer, 0, payload_bytes};
    VkWriteDescriptorSet writes[2]{};
    for (uint32_t index = 0; index < 2; ++index) {
        writes[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[index].dstSet = set;
        writes[index].dstBinding = index;
        writes[index].descriptorCount = 1;
        writes[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[index].pBufferInfo = index == 0 ? &source_info : &payload_info;
    }
    if (ok) vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
    const VkShaderModuleCreateInfo shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
        spirv.size() * sizeof(uint32_t), spirv.data()};
    const VkPushConstantRange push_range{VK_SHADER_STAGE_COMPUTE_BIT, 0, 8};
    const VkPipelineLayoutCreateInfo pipeline_layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        nullptr, 0, 1, &layout, 1, &push_range};
    if (ok) ok = vkCreateShaderModule(device, &shader_info, nullptr, &shader) == VK_SUCCESS &&
        vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &pipeline_layout) == VK_SUCCESS;
    const VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, shader, "main", nullptr};
    const VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        nullptr, 0, stage, pipeline_layout, VK_NULL_HANDLE, -1};
    if (ok) ok = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) == VK_SUCCESS;
    const VkCommandPoolCreateInfo command_pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        nullptr, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, family};
    VkCommandBuffer command = VK_NULL_HANDLE;
    const VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, 0};
    if (ok) ok = vkCreateCommandPool(device, &command_pool_info, nullptr, &command_pool) == VK_SUCCESS;
    const VkCommandBufferAllocateInfo command_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        nullptr, command_pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    if (ok) ok = vkAllocateCommandBuffers(device, &command_info, &command) == VK_SUCCESS &&
        vkCreateFence(device, &fence_info, nullptr, &fence) == VK_SUCCESS;
    if (!ok) {
        error = "GPU exact subset failed to initialize Vulkan resources";
        cleanup();
        return false;
    }
    std::vector<uint8_t> raw(request.blocks.size() * 16u);
    for (const auto & batch : batches) {
        std::vector<gpu_texel> source;
        source.reserve(size_t(batch.block_count) * format.block_width * format.block_height);
        for (uint32_t index = 0; index < batch.block_count; ++index) {
            for (const auto & texel : request.blocks[batch.first_block + index].texels) {
                source.push_back({{texel.rgba[0], texel.rgba[1], texel.rgba[2], texel.rgba[3]}});
            }
        }
        if (!map_write(device, source_memory, source.data(), source.size() * sizeof(gpu_texel)) ||
            vkResetFences(device, 1, &fence) != VK_SUCCESS ||
            vkResetCommandBuffer(command, 0) != VK_SUCCESS) {
            error = "GPU exact subset failed to prepare a source batch";
            cleanup();
            return false;
        }
        const uint32_t constants[2] = {batch.block_count, format.block_width * format.block_height};
        const VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
            VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
        ok = vkBeginCommandBuffer(command, &begin) == VK_SUCCESS;
        if (ok) {
            vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout,
                0, 1, &set, 0, nullptr);
            vkCmdPushConstants(command, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, constants);
            vkCmdDispatch(command, batch.block_count, 1, 1);
            ok = vkEndCommandBuffer(command) == VK_SUCCESS;
        }
        const VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr,
            1, &command, 0, nullptr};
        if (ok) ok = vkQueueSubmit(queue, 1, &submit, fence) == VK_SUCCESS &&
            vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX) == VK_SUCCESS;
        void * mapped = nullptr;
        if (!ok || vkMapMemory(device, payload_memory, 0, batch.block_count * 16u, 0, &mapped) != VK_SUCCESS) {
            error = "GPU exact subset dispatch failed";
            cleanup();
            return false;
        }
        std::memcpy(raw.data() + size_t(batch.first_block) * 16u, mapped, batch.block_count * 16u);
        vkUnmapMemory(device, payload_memory);
    }
    blocks = std::move(cpu_check);
    for (size_t index = 0; index < blocks.size(); ++index) {
        std::memcpy(blocks[index].payload.data(), raw.data() + index * 16u, 16u);
    }
    error.clear();
    cleanup();
    return true;
}
