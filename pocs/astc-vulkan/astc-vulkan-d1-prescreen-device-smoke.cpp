#include "astc-vulkan-d1-prescreen.h"
#include "astc-vulkan-resource.h"

#include <vulkan/vulkan.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <vector>

namespace {
std::vector<uint32_t> read_spirv(const char * path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return {};
    const auto size = file.tellg();
    if (size <= 0 || size % static_cast<std::streamoff>(sizeof(uint32_t)) != 0) return {};
    std::vector<uint32_t> code(static_cast<size_t>(size) / sizeof(uint32_t));
    file.seekg(0); file.read(reinterpret_cast<char *>(code.data()), size);
    return file ? code : std::vector<uint32_t>{};
}

bool choose_device(VkInstance instance, VkPhysicalDevice & physical, uint32_t & family) {
    uint32_t count = 0;
    if (vkEnumeratePhysicalDevices(instance, &count, nullptr) != VK_SUCCESS || count == 0) return false;
    std::vector<VkPhysicalDevice> devices(count);
    if (vkEnumeratePhysicalDevices(instance, &count, devices.data()) != VK_SUCCESS) return false;
    for (int pass = 0; pass < 2; ++pass) for (auto device : devices) {
        VkPhysicalDeviceProperties properties{}; vkGetPhysicalDeviceProperties(device, &properties);
        if ((pass == 0) != (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)) continue;
        uint32_t queues = 0; vkGetPhysicalDeviceQueueFamilyProperties(device, &queues, nullptr);
        std::vector<VkQueueFamilyProperties> families(queues);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queues, families.data());
        for (uint32_t i = 0; i < queues; ++i) if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            physical = device; family = i; return true;
        }
    }
    return false;
}

bool make_buffer(VkPhysicalDevice physical, VkDevice device, VkDeviceSize size,
                 VkBufferUsageFlags usage, VkBuffer & buffer, VkDeviceMemory & memory) {
    const VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, size,
        usage, VK_SHARING_MODE_EXCLUSIVE, 0, nullptr};
    if (vkCreateBuffer(device, &info, nullptr, &buffer) != VK_SUCCESS) return false;
    VkMemoryRequirements requirements{}; vkGetBufferMemoryRequirements(device, buffer, &requirements);
    const uint32_t type = astc_vulkan_find_memory_type(physical, requirements.memoryTypeBits,
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

} // namespace

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    const auto spirv = read_spirv(argv[1]); if (spirv.empty()) return 77;
    const uint32_t rows = 4, columns = 8;
    const std::vector<float> weights{
        0.0f, 0.2f, 0.4f, 0.6f, 10.0f, 10.2f, 10.4f, 10.6f,
        0.1f, 0.3f, 0.5f, 0.7f, 10.1f, 10.3f, 10.5f, 10.7f,
        0.0f, 0.2f, 0.4f, 0.6f, 10.0f, 10.2f, 10.4f, 10.6f,
        0.1f, 0.3f, 0.5f, 0.7f, 10.1f, 10.3f, 10.5f, 10.7f};
    const std::vector<float> energy{1.0f, 2.0f, 3.0f, 4.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    const std::vector<astc_vulkan_d1_prescreen_candidate> candidates{
        {astc_vulkan_footprint::k4x4, 16}, {astc_vulkan_footprint::k8x6, 16},
        {astc_vulkan_footprint::k10x8, 8}};
    std::vector<astc_vulkan_d1_prescreen_score> expected;
    if (!astc_vulkan_score_d1_prescreen_cpu(weights, rows, columns, energy, candidates, expected)) return 1;
    struct candidate_gpu { uint32_t width, height, levels, reserved; };
    std::vector<candidate_gpu> gpu_candidates;
    for (const auto & candidate : candidates) {
        const auto format = astc_vulkan_format(candidate.footprint);
        gpu_candidates.push_back({format.block_width, format.block_height, candidate.levels, 0});
    }

    const VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr, "astc-d1-prescreen", 1,
        "llama.cpp", 1, VK_API_VERSION_1_1};
    const VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, nullptr, 0,
        &app, 0, nullptr, 0, nullptr};
    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&instance_info, nullptr, &instance) != VK_SUCCESS) return 77;
    VkPhysicalDevice physical = VK_NULL_HANDLE; uint32_t family = UINT32_MAX;
    if (!choose_device(instance, physical, family)) { vkDestroyInstance(instance, nullptr); return 77; }
    constexpr float priority = 1.0f;
    const VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0,
        family, 1, &priority};
    const VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, nullptr, 0, 1,
        &queue_info, 0, nullptr, 0, nullptr, nullptr};
    VkDevice device = VK_NULL_HANDLE;
    if (vkCreateDevice(physical, &device_info, nullptr, &device) != VK_SUCCESS) {
        vkDestroyInstance(instance, nullptr); return 77;
    }
    VkQueue queue = VK_NULL_HANDLE; vkGetDeviceQueue(device, family, 0, &queue);
    VkBuffer buffers[4]{}; VkDeviceMemory memory[4]{};
    const VkDeviceSize sizes[] = {weights.size() * sizeof(float), energy.size() * sizeof(float),
        gpu_candidates.size() * sizeof(candidate_gpu), candidates.size() * sizeof(float)};
    bool ok = true;
    for (uint32_t i = 0; i < 4; ++i) ok &= make_buffer(physical, device, sizes[i],
        i == 3 ? VK_BUFFER_USAGE_STORAGE_BUFFER_BIT : VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        buffers[i], memory[i]);
    ok &= write_buffer(device, memory[0], weights.data(), sizes[0]);
    ok &= write_buffer(device, memory[1], energy.data(), sizes[1]);
    ok &= write_buffer(device, memory[2], gpu_candidates.data(), sizes[2]);
    const VkDescriptorSetLayoutBinding bindings[4] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
    const VkDescriptorSetLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        nullptr, 0, 4, bindings};
    VkDescriptorSetLayout layout = VK_NULL_HANDLE; VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE; VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkShaderModule module = VK_NULL_HANDLE; VkPipeline pipeline = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE; VkCommandBuffer command = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    const VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4};
    const VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0, 1, 1, &pool_size};
    ok &= vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &layout) == VK_SUCCESS;
    ok &= vkCreateDescriptorPool(device, &pool_info, nullptr, &pool) == VK_SUCCESS;
    const VkDescriptorSetAllocateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, pool, 1, &layout};
    ok &= vkAllocateDescriptorSets(device, &set_info, &set) == VK_SUCCESS;
    VkDescriptorBufferInfo infos[4] = {{buffers[0], 0, sizes[0]}, {buffers[1], 0, sizes[1]},
        {buffers[2], 0, sizes[2]}, {buffers[3], 0, sizes[3]}};
    VkWriteDescriptorSet writes[4]{};
    for (uint32_t i = 0; i < 4; ++i) { writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[i].dstSet = set;
        writes[i].dstBinding = i; writes[i].descriptorCount = 1; writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[i].pBufferInfo = &infos[i]; }
    if (ok) vkUpdateDescriptorSets(device, 4, writes, 0, nullptr);
    const VkShaderModuleCreateInfo shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
        spirv.size() * sizeof(uint32_t), spirv.data()};
    const VkPushConstantRange push_range{VK_SHADER_STAGE_COMPUTE_BIT, 0, 12};
    const VkPipelineLayoutCreateInfo pipeline_layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        nullptr, 0, 1, &layout, 1, &push_range};
    ok &= vkCreateShaderModule(device, &shader_info, nullptr, &module) == VK_SUCCESS;
    ok &= vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &pipeline_layout) == VK_SUCCESS;
    const VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
        VK_SHADER_STAGE_COMPUTE_BIT, module, "main", nullptr};
    const VkComputePipelineCreateInfo compute_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, nullptr, 0,
        stage, pipeline_layout, VK_NULL_HANDLE, -1};
    ok &= vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &compute_info, nullptr, &pipeline) == VK_SUCCESS;
    const VkCommandPoolCreateInfo command_pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr,
        VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, family};
    ok &= vkCreateCommandPool(device, &command_pool_info, nullptr, &command_pool) == VK_SUCCESS;
    const VkCommandBufferAllocateInfo command_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr,
        command_pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    ok &= vkAllocateCommandBuffers(device, &command_info, &command) == VK_SUCCESS;
    const VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, 0};
    ok &= vkCreateFence(device, &fence_info, nullptr, &fence) == VK_SUCCESS;
    const VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
    const uint32_t push[] = {rows, columns, static_cast<uint32_t>(candidates.size())};
    if (ok) ok &= vkBeginCommandBuffer(command, &begin) == VK_SUCCESS;
    if (ok) { vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(command, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);
        vkCmdDispatch(command, static_cast<uint32_t>(candidates.size()), 1, 1);
        ok &= vkEndCommandBuffer(command) == VK_SUCCESS; }
    const VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &command, 0, nullptr};
    if (ok) ok &= vkQueueSubmit(queue, 1, &submit, fence) == VK_SUCCESS &&
        vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX) == VK_SUCCESS;
    std::vector<float> actual(candidates.size(), NAN);
    if (ok) { void * mapped = nullptr; ok &= vkMapMemory(device, memory[3], 0, sizes[3], 0, &mapped) == VK_SUCCESS;
        if (ok) std::memcpy(actual.data(), mapped, static_cast<size_t>(sizes[3])); if (mapped) vkUnmapMemory(device, memory[3]); }
    if (fence) vkDestroyFence(device, fence, nullptr); if (command_pool) vkDestroyCommandPool(device, command_pool, nullptr);
    if (pipeline) vkDestroyPipeline(device, pipeline, nullptr); if (module) vkDestroyShaderModule(device, module, nullptr);
    if (pipeline_layout) vkDestroyPipelineLayout(device, pipeline_layout, nullptr); if (pool) vkDestroyDescriptorPool(device, pool, nullptr);
    if (layout) vkDestroyDescriptorSetLayout(device, layout, nullptr);
    for (uint32_t i = 0; i < 4; ++i) { if (buffers[i]) vkDestroyBuffer(device, buffers[i], nullptr); if (memory[i]) vkFreeMemory(device, memory[i], nullptr); }
    vkDestroyDevice(device, nullptr); vkDestroyInstance(instance, nullptr);
    if (!ok) return 1;
    for (size_t i = 0; i < actual.size(); ++i) if (std::abs(actual[i] - expected[i].weighted_error) > 1e-4f) {
        std::fprintf(stderr, "D1 pre-screen GPU mismatch candidate %zu: %.8g vs %.8g\n", i, actual[i], expected[i].weighted_error);
        return 1;
    }
    std::printf("D1 GPU pre-screen smoke passed (%zu candidates)\n", actual.size());
    return 0;
}
