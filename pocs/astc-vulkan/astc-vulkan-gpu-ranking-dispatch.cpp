#include "astc-vulkan-gpu-ranking-dispatch.h"

#include <cstring>
#include <limits>

namespace {

struct gpu_record {
    uint32_t atlas_block_x;
    uint32_t atlas_block_y;
    uint32_t source_block;
    uint32_t layout;
};

bool create_host_buffer(VkPhysicalDevice physical_device, VkDevice device,
                        VkDeviceSize size, VkBufferUsageFlags usage,
                        VkBuffer & buffer, VkDeviceMemory & memory) {
    const VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, size,
        usage, VK_SHARING_MODE_EXCLUSIVE, 0, nullptr};
    if (vkCreateBuffer(device, &info, nullptr, &buffer) != VK_SUCCESS) return false;
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device, buffer, &requirements);
    uint32_t type = astc_vulkan_find_memory_type(physical_device, requirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (type == std::numeric_limits<uint32_t>::max()) {
        type = astc_vulkan_find_memory_type(physical_device, requirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    }
    if (type == std::numeric_limits<uint32_t>::max()) {
        vkDestroyBuffer(device, buffer, nullptr);
        buffer = VK_NULL_HANDLE;
        return false;
    }
    const VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr,
        requirements.size, type};
    if (vkAllocateMemory(device, &allocation, nullptr, &memory) != VK_SUCCESS ||
        vkBindBufferMemory(device, buffer, memory, 0) != VK_SUCCESS) {
        vkDestroyBuffer(device, buffer, nullptr);
        if (memory != VK_NULL_HANDLE) vkFreeMemory(device, memory, nullptr);
        buffer = VK_NULL_HANDLE;
        memory = VK_NULL_HANDLE;
        return false;
    }
    return true;
}

void destroy_buffer(VkDevice device, VkBuffer & buffer, VkDeviceMemory & memory) {
    if (buffer != VK_NULL_HANDLE) vkDestroyBuffer(device, buffer, nullptr);
    if (memory != VK_NULL_HANDLE) vkFreeMemory(device, memory, nullptr);
    buffer = VK_NULL_HANDLE;
    memory = VK_NULL_HANDLE;
}

bool map_write(VkDevice device, VkDeviceMemory memory, const void * data, VkDeviceSize bytes) {
    void * mapped = nullptr;
    if (vkMapMemory(device, memory, 0, bytes, 0, &mapped) != VK_SUCCESS) return false;
    std::memcpy(mapped, data, static_cast<size_t>(bytes));
    const VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, nullptr,
        memory, 0, VK_WHOLE_SIZE};
    const VkResult result = vkFlushMappedMemoryRanges(device, 1, &range);
    vkUnmapMemory(device, memory);
    return result == VK_SUCCESS;
}

} // namespace

astc_vulkan_gpu_ranking_session::~astc_vulkan_gpu_ranking_session() { reset(); }

void astc_vulkan_gpu_ranking_session::reset() {
    if (device_ != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device_);
        if (fence_ != VK_NULL_HANDLE) vkDestroyFence(device_, fence_, nullptr);
        if (command_pool_ != VK_NULL_HANDLE) vkDestroyCommandPool(device_, command_pool_, nullptr);
        if (pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(device_, pipeline_, nullptr);
        if (shader_module_ != VK_NULL_HANDLE) vkDestroyShaderModule(device_, shader_module_, nullptr);
        if (pipeline_layout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
        if (descriptor_pool_ != VK_NULL_HANDLE) vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
        if (descriptor_layout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device_, descriptor_layout_, nullptr);
        destroy_buffer(device_, delta_buffer_, delta_memory_);
        destroy_buffer(device_, activation_buffer_, activation_memory_);
        destroy_buffer(device_, baseline_buffer_, baseline_memory_);
        destroy_buffer(device_, records_buffer_, records_memory_);
    }
    atlas_texture_.reset();
    physical_device_ = VK_NULL_HANDLE;
    device_ = VK_NULL_HANDLE;
    queue_ = VK_NULL_HANDLE;
    queue_family_ = UINT32_MAX;
    candidate_count_ = source_blocks_x_ = tensor_width_ = tensor_logical_height_ = 0;
    calibration_samples_ = block_width_ = block_height_ = 0;
    descriptor_set_ = VK_NULL_HANDLE;
    command_buffer_ = VK_NULL_HANDLE;
}

bool astc_vulkan_gpu_ranking_session::init(
        VkPhysicalDevice physical_device, VkDevice device, VkQueue queue, uint32_t queue_family,
        const astc_vulkan_gpu_ranking_atlas & atlas, uint32_t source_blocks_x,
        uint32_t tensor_width, uint32_t tensor_logical_height, uint32_t calibration_samples,
        const std::vector<uint32_t> & spirv, std::string & error) {
    reset();
    const auto format = astc_vulkan_format(atlas.footprint);
    if (physical_device == VK_NULL_HANDLE || device == VK_NULL_HANDLE || queue == VK_NULL_HANDLE ||
        queue_family == UINT32_MAX || atlas.records.empty() || spirv.empty() ||
        source_blocks_x == 0 || tensor_width == 0 || tensor_logical_height == 0 ||
        calibration_samples == 0 || (atlas.footprint != astc_vulkan_footprint::k8x5 &&
        atlas.footprint != astc_vulkan_footprint::k10x5)) {
        error = "invalid GPU ranking session configuration";
        return false;
    }
    physical_device_ = physical_device; device_ = device; queue_ = queue; queue_family_ = queue_family;
    candidate_count_ = static_cast<uint32_t>(atlas.records.size());
    source_blocks_x_ = source_blocks_x; tensor_width_ = tensor_width;
    tensor_logical_height_ = tensor_logical_height; calibration_samples_ = calibration_samples;
    block_width_ = format.block_width; block_height_ = format.block_height;
    if (!atlas_texture_.upload(physical_device_, device_, queue_, queue_family_,
            static_cast<uint8_t>(atlas.footprint), atlas.width, atlas.height, atlas.payload, error)) {
        reset(); return false;
    }
    std::vector<gpu_record> records; records.reserve(atlas.records.size());
    std::vector<uint32_t> baselines; baselines.reserve(atlas.records.size());
    for (const auto & record : atlas.records) {
        records.push_back({record.atlas_block_x, record.atlas_block_y, record.source_block,
            static_cast<uint32_t>(record.layout)});
        baselines.push_back(record.baseline_record);
    }
    const VkDeviceSize records_bytes = records.size() * sizeof(gpu_record);
    const VkDeviceSize baselines_bytes = baselines.size() * sizeof(uint32_t);
    const VkDeviceSize activation_bytes = static_cast<VkDeviceSize>(calibration_samples_) * tensor_width_ * sizeof(float);
    const VkDeviceSize delta_bytes = static_cast<VkDeviceSize>(candidate_count_) * calibration_samples_ *
        block_height_ * 2u * sizeof(float);
    if (!create_host_buffer(physical_device_, device_, records_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, records_buffer_, records_memory_) ||
        !create_host_buffer(physical_device_, device_, baselines_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, baseline_buffer_, baseline_memory_) ||
        !create_host_buffer(physical_device_, device_, activation_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, activation_buffer_, activation_memory_) ||
        !create_host_buffer(physical_device_, device_, delta_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, delta_buffer_, delta_memory_) ||
        !map_write(device_, records_memory_, records.data(), records_bytes) ||
        !map_write(device_, baseline_memory_, baselines.data(), baselines_bytes)) {
        error = "failed to allocate or upload GPU ranking buffers"; reset(); return false;
    }
    const VkDescriptorSetLayoutBinding bindings[5] = {
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    const VkDescriptorSetLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        nullptr, 0, 5, bindings};
    if (vkCreateDescriptorSetLayout(device_, &layout_info, nullptr, &descriptor_layout_) != VK_SUCCESS) {
        error = "failed to create GPU ranking descriptor layout"; reset(); return false;
    }
    const VkDescriptorPoolSize pool_sizes[2] = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4}};
    const VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr,
        0, 1, 2, pool_sizes};
    if (vkCreateDescriptorPool(device_, &pool_info, nullptr, &descriptor_pool_) != VK_SUCCESS) {
        error = "failed to allocate GPU ranking descriptor set"; reset(); return false;
    }
    const VkDescriptorSetAllocateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr,
        descriptor_pool_, 1, &descriptor_layout_};
    if (vkAllocateDescriptorSets(device_, &set_info, &descriptor_set_) != VK_SUCCESS) {
        error = "failed to allocate GPU ranking descriptor set"; reset(); return false;
    }
    const VkDescriptorImageInfo image{atlas_texture_.sampler(), atlas_texture_.view(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    const VkDescriptorBufferInfo buffers[4] = {{records_buffer_, 0, records_bytes}, {baseline_buffer_, 0, baselines_bytes},
        {activation_buffer_, 0, activation_bytes}, {delta_buffer_, 0, delta_bytes}};
    VkWriteDescriptorSet writes[5]{};
    for (uint32_t i = 0; i < 5; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[i].dstSet = descriptor_set_;
        writes[i].dstBinding = i; writes[i].descriptorCount = 1;
        writes[i].descriptorType = i == 0 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        if (i == 0) writes[i].pImageInfo = &image; else writes[i].pBufferInfo = &buffers[i - 1];
    }
    vkUpdateDescriptorSets(device_, 5, writes, 0, nullptr);
    const VkShaderModuleCreateInfo shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
        spirv.size() * sizeof(uint32_t), spirv.data()};
    if (vkCreateShaderModule(device_, &shader_info, nullptr, &shader_module_) != VK_SUCCESS) {
        error = "failed to create GPU ranking shader module"; reset(); return false;
    }
    const VkPushConstantRange push_range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(astc_vulkan_gpu_ranking_push_constants)};
    const VkPipelineLayoutCreateInfo pipeline_layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        nullptr, 0, 1, &descriptor_layout_, 1, &push_range};
    if (vkCreatePipelineLayout(device_, &pipeline_layout_info, nullptr, &pipeline_layout_) != VK_SUCCESS) {
        error = "failed to create GPU ranking compute pipeline"; reset(); return false;
    }
    const VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
        VK_SHADER_STAGE_COMPUTE_BIT, shader_module_, "main", nullptr};
    const VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, nullptr,
        0, stage, pipeline_layout_, VK_NULL_HANDLE, -1};
    if (vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline_) != VK_SUCCESS) {
        error = "failed to create GPU ranking compute pipeline"; reset(); return false;
    }
    const VkCommandPoolCreateInfo command_pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr,
        VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, queue_family_};
    const VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, 0};
    if (vkCreateCommandPool(device_, &command_pool_info, nullptr, &command_pool_) != VK_SUCCESS) {
        error = "failed to create GPU ranking command resources"; reset(); return false;
    }
    const VkCommandBufferAllocateInfo command_buffer_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        nullptr, command_pool_, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    if (vkAllocateCommandBuffers(device_, &command_buffer_info, &command_buffer_) != VK_SUCCESS ||
        vkCreateFence(device_, &fence_info, nullptr, &fence_) != VK_SUCCESS) {
        error = "failed to create GPU ranking command resources"; reset(); return false;
    }
    error.clear(); return true;
}

bool astc_vulkan_gpu_ranking_session::run(const std::vector<float> & activations,
        float reconstruction_scale, std::vector<float> & deltas, std::string & error) {
    if (!ready() || activations.size() != static_cast<size_t>(calibration_samples_) * tensor_width_) {
        error = "invalid GPU ranking run inputs"; return false;
    }
    const VkDeviceSize activation_bytes = activations.size() * sizeof(float);
    const VkDeviceSize delta_bytes = static_cast<VkDeviceSize>(candidate_count_) * calibration_samples_ * block_height_ * 2u * sizeof(float);
    if (!map_write(device_, activation_memory_, activations.data(), activation_bytes) ||
        vkResetFences(device_, 1, &fence_) != VK_SUCCESS || vkResetCommandBuffer(command_buffer_, 0) != VK_SUCCESS) {
        error = "failed to prepare GPU ranking dispatch"; return false;
    }
    const VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
    if (vkBeginCommandBuffer(command_buffer_, &begin) != VK_SUCCESS) { error = "failed to begin GPU ranking dispatch"; return false; }
    const astc_vulkan_gpu_ranking_push_constants constants{candidate_count_, calibration_samples_, tensor_width_,
        tensor_logical_height_, source_blocks_x_, block_width_, block_height_, reconstruction_scale};
    vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    vkCmdBindDescriptorSets(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1, &descriptor_set_, 0, nullptr);
    vkCmdPushConstants(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants), &constants);
    const VkBufferMemoryBarrier activation_barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, nullptr,
        VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_QUEUE_FAMILY_IGNORED,
        VK_QUEUE_FAMILY_IGNORED, activation_buffer_, 0, activation_bytes};
    vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_HOST_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1,
        &activation_barrier, 0, nullptr);
    vkCmdDispatch(command_buffer_, candidate_count_, calibration_samples_ * block_height_ * 2u, 1);
    const VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_HOST_READ_BIT, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, delta_buffer_, 0, delta_bytes};
    vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
        0, 0, nullptr, 1, &barrier, 0, nullptr);
    if (vkEndCommandBuffer(command_buffer_) != VK_SUCCESS) { error = "failed to end GPU ranking dispatch"; return false; }
    const VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &command_buffer_, 0, nullptr};
    if (vkQueueSubmit(queue_, 1, &submit, fence_) != VK_SUCCESS ||
        vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX) != VK_SUCCESS) { error = "GPU ranking dispatch failed"; return false; }
    void * mapped = nullptr;
    if (vkMapMemory(device_, delta_memory_, 0, delta_bytes, 0, &mapped) != VK_SUCCESS) { error = "failed to map GPU ranking deltas"; return false; }
    const VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, nullptr, delta_memory_, 0, VK_WHOLE_SIZE};
    const VkResult invalidate = vkInvalidateMappedMemoryRanges(device_, 1, &range);
    if (invalidate == VK_SUCCESS) {
        deltas.resize(static_cast<size_t>(delta_bytes / sizeof(float)));
        std::memcpy(deltas.data(), mapped, static_cast<size_t>(delta_bytes));
    }
    vkUnmapMemory(device_, delta_memory_);
    if (invalidate != VK_SUCCESS) { error = "failed to invalidate GPU ranking deltas"; return false; }
    error.clear(); return true;
}
