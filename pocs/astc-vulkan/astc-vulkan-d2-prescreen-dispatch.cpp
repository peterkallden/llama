#include "astc-vulkan-d2-prescreen-dispatch.h"

#include "astc-vulkan-resource.h"

#include <vulkan/vulkan.h>

#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <vector>

namespace {

std::vector<uint32_t> read_spirv(const std::string & path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return {};
    const auto size = file.tellg();
    if (size <= 0 || size % static_cast<std::streamoff>(sizeof(uint32_t)) != 0) return {};
    std::vector<uint32_t> code(static_cast<size_t>(size) / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char *>(code.data()), size);
    return file ? code : std::vector<uint32_t>{};
}

bool make_buffer(VkPhysicalDevice physical, VkDevice device, VkDeviceSize size,
                 VkBufferUsageFlags usage, VkBuffer & buffer, VkDeviceMemory & memory) {
    const VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, size,
        usage, VK_SHARING_MODE_EXCLUSIVE, 0, nullptr};
    if (vkCreateBuffer(device, &info, nullptr, &buffer) != VK_SUCCESS) return false;
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device, buffer, &requirements);
    uint32_t type = astc_vulkan_find_memory_type(physical, requirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (type == std::numeric_limits<uint32_t>::max()) return false;
    const VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr,
        requirements.size, type};
    if (vkAllocateMemory(device, &allocation, nullptr, &memory) != VK_SUCCESS ||
        vkBindBufferMemory(device, buffer, memory, 0) != VK_SUCCESS) return false;
    return true;
}

bool write_buffer(VkDevice device, VkDeviceMemory memory, const void * data, VkDeviceSize size) {
    void * mapped = nullptr;
    if (vkMapMemory(device, memory, 0, size, 0, &mapped) != VK_SUCCESS) return false;
    std::memcpy(mapped, data, static_cast<size_t>(size));
    vkUnmapMemory(device, memory);
    return true;
}

bool choose_compute(VkInstance instance, VkPhysicalDevice & physical, uint32_t & family) {
    uint32_t count = 0;
    if (vkEnumeratePhysicalDevices(instance, &count, nullptr) != VK_SUCCESS || count == 0) return false;
    std::vector<VkPhysicalDevice> devices(count);
    if (vkEnumeratePhysicalDevices(instance, &count, devices.data()) != VK_SUCCESS) return false;
    // Prefer a discrete arithmetic device, but do not require sampled ASTC.
    for (int pass = 0; pass < 2; ++pass) for (const auto device : devices) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(device, &properties);
        if ((pass == 0) != (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)) continue;
        uint32_t queue_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_count, nullptr);
        std::vector<VkQueueFamilyProperties> queues(queue_count);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_count, queues.data());
        for (uint32_t index = 0; index < queue_count; ++index) {
            if ((queues[index].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) continue;
            physical = device;
            family = index;
            return true;
        }
    }
    return false;
}

} // namespace

bool astc_vulkan_score_d2_prescreen_gpu_default(
        const std::string & spirv_path, const std::vector<float> & weights,
        uint32_t rows, uint32_t columns, const std::vector<float> & column_energy,
        const std::vector<astc_vulkan_d2_prescreen_candidate> & candidates,
        std::vector<astc_vulkan_d2_prescreen_score> & scores, std::string & error) {
    std::vector<astc_vulkan_d2_prescreen_score> cpu_scores;
    if (!astc_vulkan_score_d2_prescreen_cpu(weights, rows, columns, column_energy,
                                            candidates, cpu_scores)) {
        error = "invalid D2 pre-screen inputs";
        return false;
    }
    const auto code = read_spirv(spirv_path);
    if (code.empty()) { error = "invalid D2 pre-screen SPIR-V"; return false; }

    struct gpu_candidate { uint32_t width, height, levels, scaled; };
    std::vector<gpu_candidate> gpu_candidates;
    gpu_candidates.reserve(candidates.size());
    for (const auto & candidate : candidates) {
        const auto fp = astc_vulkan_format(candidate.footprint);
        gpu_candidates.push_back({fp.block_width, fp.block_height, candidate.levels,
            candidate.normalization == astc_vulkan_d2_prescreen_normalization::row_absmax ? 1u : 0u});
    }

    const VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr,
        "astc-d2-prescreen", 1, "llama.cpp", 1, VK_API_VERSION_1_1};
    const VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, nullptr, 0,
        &app, 0, nullptr, 0, nullptr};
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    uint32_t family = UINT32_MAX;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    if (vkCreateInstance(&instance_info, nullptr, &instance) != VK_SUCCESS ||
        !choose_compute(instance, physical, family)) {
        if (instance) vkDestroyInstance(instance, nullptr);
        error = "D2 pre-screen has no Vulkan compute device";
        return false;
    }
    constexpr float priority = 1.0f;
    const VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr,
        0, family, 1, &priority};
    const VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, nullptr, 0,
        1, &queue_info, 0, nullptr, 0, nullptr, nullptr};
    if (vkCreateDevice(physical, &device_info, nullptr, &device) != VK_SUCCESS) {
        vkDestroyInstance(instance, nullptr);
        error = "D2 pre-screen cannot create Vulkan compute device";
        return false;
    }
    vkGetDeviceQueue(device, family, 0, &queue);
    VkBuffer buffers[4]{};
    VkDeviceMemory memory[4]{};
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkShaderModule module = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool ok = true;
    const VkDeviceSize sizes[] = {
        weights.size() * sizeof(float), column_energy.size() * sizeof(float),
        gpu_candidates.size() * sizeof(gpu_candidate), candidates.size() * sizeof(float)};
    for (uint32_t i = 0; ok && i < 4; ++i)
        ok = make_buffer(physical, device, sizes[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                         buffers[i], memory[i]);
    if (ok) ok = write_buffer(device, memory[0], weights.data(), sizes[0]) &&
                     write_buffer(device, memory[1], column_energy.data(), sizes[1]) &&
                     write_buffer(device, memory[2], gpu_candidates.data(), sizes[2]);

    const VkDescriptorSetLayoutBinding bindings[4] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
    const VkDescriptorSetLayoutCreateInfo layout_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 4, bindings};
    const VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4};
    const VkDescriptorPoolCreateInfo pool_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0, 1, 1, &pool_size};
    if (ok) ok = vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &layout) == VK_SUCCESS &&
                     vkCreateDescriptorPool(device, &pool_info, nullptr, &pool) == VK_SUCCESS;
    const VkDescriptorSetAllocateInfo allocate_set{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        nullptr, pool, 1, &layout};
    if (ok) ok = vkAllocateDescriptorSets(device, &allocate_set, &set) == VK_SUCCESS;
    VkDescriptorBufferInfo infos[4]{};
    for (uint32_t i = 0; i < 4; ++i) infos[i] = {buffers[i], 0, sizes[i]};
    VkWriteDescriptorSet writes[4]{};
    for (uint32_t i = 0; i < 4; ++i) {
        writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, i, 0, 1,
                     VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &infos[i], nullptr};
    }
    if (ok) vkUpdateDescriptorSets(device, 4, writes, 0, nullptr);
    const VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        nullptr, 0, code.size() * sizeof(uint32_t), code.data()};
    const VkPushConstantRange push_range{VK_SHADER_STAGE_COMPUTE_BIT, 0, 12};
    const VkPipelineLayoutCreateInfo pipeline_layout_info{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &layout, 1, &push_range};
    if (ok) ok = vkCreateShaderModule(device, &module_info, nullptr, &module) == VK_SUCCESS &&
                     vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr,
                                            &pipeline_layout) == VK_SUCCESS;
    const VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, module, "main", nullptr};
    const VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        nullptr, 0, stage, pipeline_layout, VK_NULL_HANDLE, -1};
    if (ok) ok = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info,
                                          nullptr, &pipeline) == VK_SUCCESS;
    const VkCommandPoolCreateInfo command_pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        nullptr, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT, family};
    if (ok) ok = vkCreateCommandPool(device, &command_pool_info, nullptr, &command_pool) == VK_SUCCESS;
    const VkCommandBufferAllocateInfo command_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        nullptr, command_pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    if (ok) ok = vkAllocateCommandBuffers(device, &command_info, &command) == VK_SUCCESS;
    const VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, 0};
    if (ok) ok = vkCreateFence(device, &fence_info, nullptr, &fence) == VK_SUCCESS;
    const VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
    const uint32_t push[] = {rows, columns, static_cast<uint32_t>(candidates.size())};
    if (ok) ok = vkBeginCommandBuffer(command, &begin) == VK_SUCCESS;
    if (ok) {
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1,
                                &set, 0, nullptr);
        vkCmdPushConstants(command, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, push);
        vkCmdDispatch(command, static_cast<uint32_t>(candidates.size()), 1, 1);
        ok = vkEndCommandBuffer(command) == VK_SUCCESS;
    }
    const VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr,
        1, &command, 0, nullptr};
    if (ok) ok = vkQueueSubmit(queue, 1, &submit, fence) == VK_SUCCESS &&
                     vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX) == VK_SUCCESS;
    std::vector<float> gpu_scores(candidates.size());
    if (ok) {
        void * mapped = nullptr;
        ok = vkMapMemory(device, memory[3], 0, sizes[3], 0, &mapped) == VK_SUCCESS;
        if (ok) { std::memcpy(gpu_scores.data(), mapped, sizes[3]); vkUnmapMemory(device, memory[3]); }
    }
    if (fence) vkDestroyFence(device, fence, nullptr);
    if (command_pool) vkDestroyCommandPool(device, command_pool, nullptr);
    if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
    if (module) vkDestroyShaderModule(device, module, nullptr);
    if (pipeline_layout) vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    if (pool) vkDestroyDescriptorPool(device, pool, nullptr);
    if (layout) vkDestroyDescriptorSetLayout(device, layout, nullptr);
    for (uint32_t i = 0; i < 4; ++i) {
        if (buffers[i]) vkDestroyBuffer(device, buffers[i], nullptr);
        if (memory[i]) vkFreeMemory(device, memory[i], nullptr);
    }
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    if (!ok) { error = "D2 pre-screen Vulkan dispatch failed"; return false; }
    for (size_t i = 0; i < gpu_scores.size(); ++i) {
        if (!std::isfinite(gpu_scores[i]) || std::abs(gpu_scores[i] - cpu_scores[i].weighted_error) > 1e-4f) {
            error = "D2 pre-screen Vulkan score mismatch";
            return false;
        }
        cpu_scores[i].weighted_error = gpu_scores[i];
        cpu_scores[i].proxy_cost = gpu_scores[i];
    }
    scores = std::move(cpu_scores);
    error.clear();
    return true;
}
