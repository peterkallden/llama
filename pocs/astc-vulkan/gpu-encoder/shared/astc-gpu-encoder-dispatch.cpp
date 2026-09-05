#include "astc-gpu-encoder-dispatch.h"

#include "astc-vulkan-resource.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>

namespace {

struct gpu_texel { float rgba[4]; };
struct gpu_proposal {
    float endpoint_low[4];
    float endpoint_high[4];
    float approximate_error;
    uint32_t source_block_id;
    uint32_t weight_grid_x;
    uint32_t weight_grid_y;
    uint32_t mode_family;
    // std430 rounds an array-of-struct element to the maximum member
    // alignment (vec4 = 16 bytes), therefore the 52-byte logical record has
    // a 64-byte array stride.
    float padding[3];
};
static_assert(sizeof(gpu_texel) == 16, "GPU source texels must match vec4");
static_assert(sizeof(gpu_proposal) == 64, "GPU proposal must match std430 layout");

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

bool create_host_buffer(VkPhysicalDevice physical, VkDevice device, VkDeviceSize bytes,
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

bool map_write(VkDevice device, VkDeviceMemory memory, const void * values, VkDeviceSize bytes) {
    void * mapped = nullptr;
    if (vkMapMemory(device, memory, 0, bytes, 0, &mapped) != VK_SUCCESS) return false;
    std::memcpy(mapped, values, static_cast<size_t>(bytes));
    vkUnmapMemory(device, memory);
    return true;
}

bool select_compute_device(VkInstance instance, VkPhysicalDevice & physical, uint32_t & family) {
    uint32_t count = 0;
    if (vkEnumeratePhysicalDevices(instance, &count, nullptr) != VK_SUCCESS || count == 0) return false;
    std::vector<VkPhysicalDevice> devices(count);
    if (vkEnumeratePhysicalDevices(instance, &count, devices.data()) != VK_SUCCESS) return false;
    // Prefer discrete for offline work but keep the path vendor-neutral.
    for (uint32_t pass = 0; pass < 2; ++pass) for (const auto device : devices) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(device, &properties);
        if ((pass == 0) != (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)) continue;
        uint32_t count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
        std::vector<VkQueueFamilyProperties> queues(count);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &count, queues.data());
        for (uint32_t index = 0; index < count; ++index) if (queues[index].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            physical = device;
            family = index;
            return true;
        }
    }
    return false;
}

} // namespace

astc_gpu_encoder_session::~astc_gpu_encoder_session() { reset(); }

bool astc_gpu_encoder_session::init(VkPhysicalDevice physical_device, VkDevice device, VkQueue queue,
                                    uint32_t queue_family, astc_vulkan_footprint footprint,
                                    uint32_t max_blocks,
                                    const std::vector<uint32_t> & spirv, std::string & error) {
    reset();
    const auto format = astc_vulkan_format(footprint);
    if (physical_device == VK_NULL_HANDLE || device == VK_NULL_HANDLE || queue == VK_NULL_HANDLE ||
        format.block_width == 0 || format.block_height == 0 || max_blocks == 0 || spirv.empty()) {
        error = "invalid GPU proposer session input";
        return false;
    }
    physical_device_ = physical_device;
    device_ = device;
    queue_ = queue;
    queue_family_ = queue_family;
    footprint_ = footprint;
    texels_per_block_ = format.block_width * format.block_height;
    capacity_ = max_blocks;
    bool ok = create_host_buffer(physical_device_, device_,
        VkDeviceSize(capacity_) * texels_per_block_ * sizeof(gpu_texel), source_buffer_, source_memory_) &&
        create_host_buffer(physical_device_, device_,
        VkDeviceSize(capacity_) * sizeof(uint32_t), source_id_buffer_, source_id_memory_) &&
        create_host_buffer(physical_device_, device_,
        VkDeviceSize(capacity_) * sizeof(gpu_proposal), proposal_buffer_, proposal_memory_);
    const VkDescriptorSetLayoutBinding bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
    const VkDescriptorSetLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        nullptr, 0, 3, bindings};
    const VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
    const VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        nullptr, 0, 1, 1, &pool_size};
    if (ok) ok = vkCreateDescriptorSetLayout(device_, &layout_info, nullptr, &descriptor_layout_) == VK_SUCCESS &&
                 vkCreateDescriptorPool(device_, &pool_info, nullptr, &descriptor_pool_) == VK_SUCCESS;
    const VkDescriptorSetAllocateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        nullptr, descriptor_pool_, 1, &descriptor_layout_};
    if (ok) ok = vkAllocateDescriptorSets(device_, &set_info, &descriptor_set_) == VK_SUCCESS;
    const VkDescriptorBufferInfo source_info{source_buffer_, 0, VkDeviceSize(capacity_) * texels_per_block_ * sizeof(gpu_texel)};
    const VkDescriptorBufferInfo proposal_info{proposal_buffer_, 0, VkDeviceSize(capacity_) * sizeof(gpu_proposal)};
    const VkDescriptorBufferInfo source_id_info{source_id_buffer_, 0, VkDeviceSize(capacity_) * sizeof(uint32_t)};
    VkWriteDescriptorSet writes[3]{};
    for (uint32_t index = 0; index < 3; ++index) {
        writes[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[index].dstSet = descriptor_set_;
        writes[index].dstBinding = index;
        writes[index].descriptorCount = 1;
        writes[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[index].pBufferInfo = index == 0 ? &source_info : index == 1 ? &proposal_info : &source_id_info;
    }
    if (ok) vkUpdateDescriptorSets(device_, 3, writes, 0, nullptr);
    const VkShaderModuleCreateInfo shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
        spirv.size() * sizeof(uint32_t), spirv.data()};
    const VkPushConstantRange push_range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t)};
    const VkPipelineLayoutCreateInfo pipeline_layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        nullptr, 0, 1, &descriptor_layout_, 1, &push_range};
    if (ok) ok = vkCreateShaderModule(device_, &shader_info, nullptr, &shader_module_) == VK_SUCCESS &&
                 vkCreatePipelineLayout(device_, &pipeline_layout_info, nullptr, &pipeline_layout_) == VK_SUCCESS;
    const VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, shader_module_, "main", nullptr};
    const VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        nullptr, 0, stage, pipeline_layout_, VK_NULL_HANDLE, -1};
    if (ok) ok = vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline_) == VK_SUCCESS;
    const VkCommandPoolCreateInfo command_pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        nullptr, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, queue_family_};
    if (ok) ok = vkCreateCommandPool(device_, &command_pool_info, nullptr, &command_pool_) == VK_SUCCESS;
    const VkCommandBufferAllocateInfo command_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        nullptr, command_pool_, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    const VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, 0};
    if (ok) ok = vkAllocateCommandBuffers(device_, &command_info, &command_buffer_) == VK_SUCCESS &&
                 vkCreateFence(device_, &fence_info, nullptr, &fence_) == VK_SUCCESS;
    if (!ok) {
        error = "failed to initialize reusable GPU proposer session";
        reset();
        return false;
    }
    error.clear();
    return true;
}

bool astc_gpu_encoder_session::run(const astc_gpu_encoder_request & request,
                                   std::vector<astc_gpu_encoder_proposal> & proposals,
                                   std::string & error) {
    std::vector<astc_gpu_encoder_batch> batches;
    if (!ready() || request.footprint != footprint_ ||
        !astc_gpu_encoder_plan_batches(request, batches)) {
        error = "GPU proposer requires valid batches matching its footprint and capacity";
        return false;
    }
    for (const auto & batch : batches) {
        if (batch.block_count == 0 || batch.block_count > capacity_) {
            error = "GPU proposer batch exceeds its resident capacity";
            return false;
        }
    }
    proposals.clear();
    proposals.reserve(request.blocks.size());
    for (const auto & batch : batches) {
        const uint32_t count = batch.block_count;
        std::vector<gpu_texel> source;
        std::vector<uint32_t> source_ids;
        source.reserve(size_t(count) * texels_per_block_);
        source_ids.reserve(count);
        for (uint32_t index = 0; index < count; ++index) {
            const auto & block = request.blocks[batch.first_block + index];
            source_ids.push_back(block.source_block_id);
            for (const auto & texel : block.texels) {
                source.push_back({{texel.rgba[0], texel.rgba[1], texel.rgba[2], texel.rgba[3]}});
            }
        }
        if (!map_write(device_, source_memory_, source.data(), source.size() * sizeof(gpu_texel)) ||
            !map_write(device_, source_id_memory_, source_ids.data(), source_ids.size() * sizeof(uint32_t))) {
            error = "failed to upload GPU proposer source batch";
            return false;
        }
        if (vkResetFences(device_, 1, &fence_) != VK_SUCCESS ||
            vkResetCommandBuffer(command_buffer_, 0) != VK_SUCCESS) {
            error = "failed to reset GPU proposer command state";
            return false;
        }
        const VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
        if (vkBeginCommandBuffer(command_buffer_, &begin) != VK_SUCCESS) { error = "failed to begin GPU proposer command"; return false; }
        vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        vkCmdBindDescriptorSets(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_,
            0, 1, &descriptor_set_, 0, nullptr);
        vkCmdPushConstants(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(count), &count);
        vkCmdDispatch(command_buffer_, count, 1, 1);
        if (vkEndCommandBuffer(command_buffer_) != VK_SUCCESS) { error = "failed to end GPU proposer command"; return false; }
        const VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr,
            1, &command_buffer_, 0, nullptr};
        if (vkQueueSubmit(queue_, 1, &submit, fence_) != VK_SUCCESS ||
            vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
            error = "GPU proposer dispatch failed";
            return false;
        }
        std::vector<gpu_proposal> result(count);
        void * mapped = nullptr;
        if (vkMapMemory(device_, proposal_memory_, 0, result.size() * sizeof(gpu_proposal), 0, &mapped) != VK_SUCCESS) {
            error = "failed to read GPU proposer results";
            return false;
        }
        std::memcpy(result.data(), mapped, result.size() * sizeof(gpu_proposal));
        vkUnmapMemory(device_, proposal_memory_);
        for (const auto & value : result) {
            astc_gpu_encoder_proposal proposal;
            proposal.source_block_id = value.source_block_id;
            proposal.weight_grid_x = value.weight_grid_x;
            proposal.weight_grid_y = value.weight_grid_y;
            proposal.mode_family = value.mode_family;
            std::memcpy(proposal.endpoint_low.data(), value.endpoint_low, sizeof(value.endpoint_low));
            std::memcpy(proposal.endpoint_high.data(), value.endpoint_high, sizeof(value.endpoint_high));
            proposal.approximate_error = value.approximate_error;
            proposals.push_back(proposal);
        }
    }
    error.clear();
    return true;
}

void astc_gpu_encoder_session::reset() {
    if (device_ != VK_NULL_HANDLE) vkDeviceWaitIdle(device_);
    if (fence_) vkDestroyFence(device_, fence_, nullptr);
    if (command_pool_) vkDestroyCommandPool(device_, command_pool_, nullptr);
    if (pipeline_) vkDestroyPipeline(device_, pipeline_, nullptr);
    if (shader_module_) vkDestroyShaderModule(device_, shader_module_, nullptr);
    if (pipeline_layout_) vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
    if (descriptor_pool_) vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
    if (descriptor_layout_) vkDestroyDescriptorSetLayout(device_, descriptor_layout_, nullptr);
    if (source_buffer_) vkDestroyBuffer(device_, source_buffer_, nullptr);
    if (source_memory_) vkFreeMemory(device_, source_memory_, nullptr);
    if (source_id_buffer_) vkDestroyBuffer(device_, source_id_buffer_, nullptr);
    if (source_id_memory_) vkFreeMemory(device_, source_id_memory_, nullptr);
    if (proposal_buffer_) vkDestroyBuffer(device_, proposal_buffer_, nullptr);
    if (proposal_memory_) vkFreeMemory(device_, proposal_memory_, nullptr);
    physical_device_ = VK_NULL_HANDLE; device_ = VK_NULL_HANDLE; queue_ = VK_NULL_HANDLE;
    queue_family_ = UINT32_MAX; footprint_ = astc_vulkan_footprint::k4x4; texels_per_block_ = 0;
    capacity_ = 0; source_buffer_ = VK_NULL_HANDLE; source_memory_ = VK_NULL_HANDLE;
    source_id_buffer_ = VK_NULL_HANDLE; source_id_memory_ = VK_NULL_HANDLE;
    proposal_buffer_ = VK_NULL_HANDLE; proposal_memory_ = VK_NULL_HANDLE; descriptor_layout_ = VK_NULL_HANDLE;
    descriptor_pool_ = VK_NULL_HANDLE; descriptor_set_ = VK_NULL_HANDLE; pipeline_layout_ = VK_NULL_HANDLE;
    shader_module_ = VK_NULL_HANDLE; pipeline_ = VK_NULL_HANDLE; command_pool_ = VK_NULL_HANDLE;
    command_buffer_ = VK_NULL_HANDLE; fence_ = VK_NULL_HANDLE;
}

bool astc_gpu_encoder_propose_gpu_default(
    const std::string & spirv_path, const astc_gpu_encoder_request & request,
    std::vector<astc_gpu_encoder_proposal> & proposals, std::string & error) {
    return astc_gpu_encoder_propose_gpu_for_footprint_default(
        spirv_path, astc_vulkan_footprint::k4x4, request, proposals, error);
}

bool astc_gpu_encoder_propose_gpu_for_footprint_default(
    const std::string & spirv_path, astc_vulkan_footprint footprint,
    const astc_gpu_encoder_request & request,
    std::vector<astc_gpu_encoder_proposal> & proposals, std::string & error) {
    const auto spirv = read_spirv(spirv_path);
    if (request.footprint != footprint || spirv.empty() || request.blocks.empty()) {
        error = "invalid GPU proposer footprint, SPIR-V, or request";
        return false;
    }
    const VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr,
        "astc-gpu-encoder-proposer", 1, "llama.cpp", 1, VK_API_VERSION_1_1};
    const VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, nullptr, 0,
        &app, 0, nullptr, 0, nullptr};
    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&instance_info, nullptr, &instance) != VK_SUCCESS) { error = "failed to create Vulkan instance"; return false; }
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    uint32_t family = UINT32_MAX;
    if (!select_compute_device(instance, physical, family)) { vkDestroyInstance(instance, nullptr); error = "no Vulkan compute device"; return false; }
    constexpr float priority = 1.0f;
    const VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr,
        0, family, 1, &priority};
    const VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, nullptr,
        0, 1, &queue_info, 0, nullptr, 0, nullptr, nullptr};
    VkDevice device = VK_NULL_HANDLE;
    if (vkCreateDevice(physical, &device_info, nullptr, &device) != VK_SUCCESS) {
        vkDestroyInstance(instance, nullptr); error = "failed to create Vulkan device"; return false;
    }
    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, family, 0, &queue);
    astc_gpu_encoder_session session;
    const uint32_t resident_capacity = std::min<uint32_t>(
        request.max_blocks_per_batch, static_cast<uint32_t>(request.blocks.size()));
    const bool initialized = session.init(physical, device, queue, family, footprint,
                                          resident_capacity, spirv, error);
    const bool ran = initialized && session.run(request, proposals, error);
    session.reset();
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    return ran;
}

bool astc_gpu_encoder_benchmark_gpu_default(
    const std::string & spirv_path, astc_vulkan_footprint footprint,
    const astc_gpu_encoder_request & request, uint32_t warmup_iterations,
    uint32_t measured_iterations, astc_gpu_encoder_benchmark_result & result,
    std::string & error) {
    result = {};
    const auto spirv = read_spirv(spirv_path);
    if (request.footprint != footprint || request.blocks.empty() || spirv.empty() ||
        measured_iterations == 0) {
        error = "invalid GPU proposer benchmark input";
        return false;
    }
    const VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr,
        "astc-gpu-encoder-benchmark", 1, "llama.cpp", 1, VK_API_VERSION_1_1};
    const VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, nullptr,
        0, &app, 0, nullptr, 0, nullptr};
    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&instance_info, nullptr, &instance) != VK_SUCCESS) {
        error = "failed to create Vulkan instance for GPU proposer benchmark";
        return false;
    }
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    uint32_t family = UINT32_MAX;
    if (!select_compute_device(instance, physical, family)) {
        vkDestroyInstance(instance, nullptr);
        error = "no Vulkan compute device for GPU proposer benchmark";
        return false;
    }
    constexpr float priority = 1.0f;
    const VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr,
        0, family, 1, &priority};
    const VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, nullptr,
        0, 1, &queue_info, 0, nullptr, 0, nullptr, nullptr};
    VkDevice device = VK_NULL_HANDLE;
    if (vkCreateDevice(physical, &device_info, nullptr, &device) != VK_SUCCESS) {
        vkDestroyInstance(instance, nullptr);
        error = "failed to create Vulkan device for GPU proposer benchmark";
        return false;
    }
    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, family, 0, &queue);
    astc_gpu_encoder_session session;
    const uint32_t resident_capacity = std::min<uint32_t>(
        request.max_blocks_per_batch, static_cast<uint32_t>(request.blocks.size()));
    bool ok = session.init(physical, device, queue, family, footprint,
                           resident_capacity, spirv, error);
    std::vector<astc_gpu_encoder_proposal> proposals;
    for (uint32_t iteration = 0; ok && iteration < warmup_iterations; ++iteration) {
        ok = session.run(request, proposals, error);
    }
    const auto started = std::chrono::steady_clock::now();
    for (uint32_t iteration = 0; ok && iteration < measured_iterations; ++iteration) {
        ok = session.run(request, proposals, error);
    }
    const auto stopped = std::chrono::steady_clock::now();
    session.reset();
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    if (!ok) return false;
    const double elapsed_ms = std::chrono::duration<double, std::milli>(stopped - started).count();
    result.block_count = static_cast<uint32_t>(request.blocks.size());
    result.warmup_iterations = warmup_iterations;
    result.measured_iterations = measured_iterations;
    result.mean_roundtrip_ms = elapsed_ms / measured_iterations;
    result.blocks_per_second = result.mean_roundtrip_ms > 0.0
        ? result.block_count * 1000.0 / result.mean_roundtrip_ms : 0.0;
    if (!std::isfinite(result.blocks_per_second) || result.blocks_per_second <= 0.0) {
        error = "invalid GPU proposer benchmark result";
        return false;
    }
    error.clear();
    return true;
}
