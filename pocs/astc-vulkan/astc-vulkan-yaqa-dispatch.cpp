#include "astc-vulkan-yaqa-dispatch.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace {

bool create_host_buffer(VkPhysicalDevice physical_device, VkDevice device,
                        VkDeviceSize size, VkBuffer & buffer,
                        VkDeviceMemory & memory) {
    if (size == 0) return false;
    const VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0,
        size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, nullptr};
    if (vkCreateBuffer(device, &info, nullptr, &buffer) != VK_SUCCESS) return false;
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device, buffer, &requirements);
    uint32_t type = UINT32_MAX;
    const VkMemoryPropertyFlags property_candidates[] = {
        static_cast<VkMemoryPropertyFlags>(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
        static_cast<VkMemoryPropertyFlags>(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT),
    };
    for (VkMemoryPropertyFlags properties : property_candidates) {
        type = UINT32_MAX;
        VkPhysicalDeviceMemoryProperties memory_properties{};
        vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);
        for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
            if ((requirements.memoryTypeBits & (1u << i)) != 0 &&
                (memory_properties.memoryTypes[i].propertyFlags & properties) == properties) {
                type = i;
                break;
            }
        }
        if (type != UINT32_MAX) break;
    }
    if (type == UINT32_MAX) {
        vkDestroyBuffer(device, buffer, nullptr);
        buffer = VK_NULL_HANDLE;
        return false;
    }
    const VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr,
        requirements.size, type};
    if (vkAllocateMemory(device, &allocation, nullptr, &memory) != VK_SUCCESS ||
        vkBindBufferMemory(device, buffer, memory, 0) != VK_SUCCESS) {
        if (memory != VK_NULL_HANDLE) vkFreeMemory(device, memory, nullptr);
        vkDestroyBuffer(device, buffer, nullptr);
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
    const VkResult flush = vkFlushMappedMemoryRanges(device, 1, &range);
    vkUnmapMemory(device, memory);
    return flush == VK_SUCCESS;
}

bool map_read(VkDevice device, VkDeviceMemory memory, void * data, VkDeviceSize bytes) {
    void * mapped = nullptr;
    if (vkMapMemory(device, memory, 0, bytes, 0, &mapped) != VK_SUCCESS) return false;
    const VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, nullptr,
        memory, 0, VK_WHOLE_SIZE};
    const VkResult invalidate = vkInvalidateMappedMemoryRanges(device, 1, &range);
    if (invalidate != VK_SUCCESS) {
        vkUnmapMemory(device, memory);
        return false;
    }
    std::memcpy(data, mapped, static_cast<size_t>(bytes));
    vkUnmapMemory(device, memory);
    return true;
}

bool checked_product(uint64_t a, uint64_t b, uint64_t & result) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) return false;
    result = a * b;
    return true;
}

} // namespace

astc_vulkan_yaqa_session::~astc_vulkan_yaqa_session() { reset(); }

void astc_vulkan_yaqa_session::reset() {
    if (device_ != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device_);
        if (fence_ != VK_NULL_HANDLE) vkDestroyFence(device_, fence_, nullptr);
        if (command_pool_ != VK_NULL_HANDLE) vkDestroyCommandPool(device_, command_pool_, nullptr);
        if (reduce_pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(device_, reduce_pipeline_, nullptr);
        if (partial_pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(device_, partial_pipeline_, nullptr);
        if (reduce_shader_module_ != VK_NULL_HANDLE) vkDestroyShaderModule(device_, reduce_shader_module_, nullptr);
        if (partial_shader_module_ != VK_NULL_HANDLE) vkDestroyShaderModule(device_, partial_shader_module_, nullptr);
        if (reduce_pipeline_layout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(device_, reduce_pipeline_layout_, nullptr);
        if (partial_pipeline_layout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(device_, partial_pipeline_layout_, nullptr);
        if (descriptor_pool_ != VK_NULL_HANDLE) vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
        if (reduce_descriptor_layout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device_, reduce_descriptor_layout_, nullptr);
        if (descriptor_layout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device_, descriptor_layout_, nullptr);
        destroy_buffer(device_, scores_buffer_, scores_memory_);
        destroy_buffer(device_, partials_buffer_, partials_memory_);
        destroy_buffer(device_, output_trace_buffer_, output_trace_memory_);
        destroy_buffer(device_, input_trace_buffer_, input_trace_memory_);
        destroy_buffer(device_, errors_buffer_, errors_memory_);
    }
    physical_device_ = VK_NULL_HANDLE;
    device_ = VK_NULL_HANDLE;
    queue_ = VK_NULL_HANDLE;
    queue_family_ = UINT32_MAX;
    rows_ = columns_ = samples_ = pairs_per_candidate_ = candidate_capacity_ = 0;
    traces_resident_ = false;
    descriptor_layout_ = VK_NULL_HANDLE;
    reduce_descriptor_layout_ = VK_NULL_HANDLE;
    descriptor_pool_ = VK_NULL_HANDLE;
    descriptor_set_ = VK_NULL_HANDLE;
    reduce_descriptor_set_ = VK_NULL_HANDLE;
    partial_pipeline_layout_ = VK_NULL_HANDLE;
    reduce_pipeline_layout_ = VK_NULL_HANDLE;
    partial_shader_module_ = VK_NULL_HANDLE;
    reduce_shader_module_ = VK_NULL_HANDLE;
    partial_pipeline_ = VK_NULL_HANDLE;
    reduce_pipeline_ = VK_NULL_HANDLE;
    command_pool_ = VK_NULL_HANDLE;
    command_buffer_ = VK_NULL_HANDLE;
    fence_ = VK_NULL_HANDLE;
}

bool astc_vulkan_yaqa_session::init(
        VkPhysicalDevice physical_device, VkDevice device, VkQueue queue,
        uint32_t queue_family, uint32_t rows, uint32_t columns, uint32_t samples,
        const std::vector<uint32_t> & partial_spirv,
        const std::vector<uint32_t> & reduce_spirv,
        uint32_t candidate_capacity, std::string & error) {
    reset();
    uint64_t pairs = 0, matrix_elements = 0, trace_input_elements = 0,
        trace_output_elements = 0, partial_elements = 0;
    if (physical_device == VK_NULL_HANDLE || device == VK_NULL_HANDLE ||
        queue == VK_NULL_HANDLE || queue_family == UINT32_MAX || rows == 0 ||
        columns == 0 || samples == 0 || candidate_capacity == 0 ||
        partial_spirv.empty() || reduce_spirv.empty() ||
        !checked_product(samples, samples, pairs) ||
        !checked_product(rows, columns, matrix_elements) ||
        !checked_product(samples, columns, trace_input_elements) ||
        !checked_product(samples, rows, trace_output_elements) ||
        !checked_product(candidate_capacity, pairs, partial_elements) ||
        pairs > std::numeric_limits<uint32_t>::max()) {
        error = "invalid YAQA GPU session configuration";
        return false;
    }
    physical_device_ = physical_device;
    device_ = device;
    queue_ = queue;
    queue_family_ = queue_family;
    rows_ = rows;
    columns_ = columns;
    samples_ = samples;
    pairs_per_candidate_ = static_cast<uint32_t>(pairs);
    candidate_capacity_ = candidate_capacity;

    const VkDeviceSize matrix_bytes = static_cast<VkDeviceSize>(candidate_capacity) *
        static_cast<VkDeviceSize>(matrix_elements) * sizeof(float);
    const VkDeviceSize input_bytes = static_cast<VkDeviceSize>(trace_input_elements) * sizeof(float);
    const VkDeviceSize output_bytes = static_cast<VkDeviceSize>(trace_output_elements) * sizeof(float);
    const VkDeviceSize partial_bytes = static_cast<VkDeviceSize>(partial_elements) * sizeof(float);
    const VkDeviceSize score_bytes = static_cast<VkDeviceSize>(candidate_capacity) * sizeof(float);
    if (!create_host_buffer(physical_device_, device_,
            matrix_bytes, errors_buffer_, errors_memory_) ||
        !create_host_buffer(physical_device_, device_, input_bytes, input_trace_buffer_, input_trace_memory_) ||
        !create_host_buffer(physical_device_, device_, output_bytes, output_trace_buffer_, output_trace_memory_) ||
        !create_host_buffer(physical_device_, device_, partial_bytes, partials_buffer_, partials_memory_) ||
        !create_host_buffer(physical_device_, device_, score_bytes, scores_buffer_, scores_memory_)) {
        error = "failed to allocate YAQA GPU buffers";
        reset();
        return false;
    }

    const VkDescriptorSetLayoutBinding bindings[5] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    const VkDescriptorSetLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        nullptr, 0, 5, bindings};
    if (vkCreateDescriptorSetLayout(device_, &layout_info, nullptr, &descriptor_layout_) != VK_SUCCESS) {
        error = "failed to create YAQA descriptor layout"; reset(); return false;
    }
    const VkDescriptorSetLayoutBinding reduce_bindings[2] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    const VkDescriptorSetLayoutCreateInfo reduce_descriptor_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        nullptr, 0, 2, reduce_bindings};
    if (vkCreateDescriptorSetLayout(device_, &reduce_descriptor_info, nullptr,
            &reduce_descriptor_layout_) != VK_SUCCESS) {
        error = "failed to create YAQA reduce descriptor layout"; reset(); return false;
    }
    const VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 7};
    const VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        nullptr, 0, 2, 1, &pool_size};
    if (vkCreateDescriptorPool(device_, &pool_info, nullptr, &descriptor_pool_) != VK_SUCCESS) {
        error = "failed to create YAQA descriptor pool"; reset(); return false;
    }
    const VkDescriptorSetLayout set_layouts[2] = {descriptor_layout_, reduce_descriptor_layout_};
    const VkDescriptorSetAllocateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        nullptr, descriptor_pool_, 2, set_layouts};
    VkDescriptorSet sets[2]{};
    if (vkAllocateDescriptorSets(device_, &set_info, sets) != VK_SUCCESS) {
        error = "failed to allocate YAQA descriptor set"; reset(); return false;
    }
    descriptor_set_ = sets[0];
    reduce_descriptor_set_ = sets[1];
    const VkDescriptorBufferInfo buffer_infos[5] = {
        {errors_buffer_, 0, matrix_bytes},
        {input_trace_buffer_, 0, input_bytes},
        {output_trace_buffer_, 0, output_bytes},
        {partials_buffer_, 0, partial_bytes},
        {scores_buffer_, 0, score_bytes},
    };
    VkWriteDescriptorSet writes[5]{};
    for (uint32_t i = 0; i < 5; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = descriptor_set_;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buffer_infos[i];
    }
    vkUpdateDescriptorSets(device_, 5, writes, 0, nullptr);
    const VkDescriptorBufferInfo reduce_infos[2] = {
        {partials_buffer_, 0, partial_bytes},
        {scores_buffer_, 0, score_bytes},
    };
    VkWriteDescriptorSet reduce_writes[2]{};
    for (uint32_t i = 0; i < 2; ++i) {
        reduce_writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        reduce_writes[i].dstSet = reduce_descriptor_set_;
        reduce_writes[i].dstBinding = i;
        reduce_writes[i].descriptorCount = 1;
        reduce_writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        reduce_writes[i].pBufferInfo = &reduce_infos[i];
    }
    vkUpdateDescriptorSets(device_, 2, reduce_writes, 0, nullptr);

    const VkShaderModuleCreateInfo partial_module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        nullptr, 0, partial_spirv.size() * sizeof(uint32_t), partial_spirv.data()};
    const VkShaderModuleCreateInfo reduce_module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        nullptr, 0, reduce_spirv.size() * sizeof(uint32_t), reduce_spirv.data()};
    if (vkCreateShaderModule(device_, &partial_module_info, nullptr, &partial_shader_module_) != VK_SUCCESS ||
        vkCreateShaderModule(device_, &reduce_module_info, nullptr, &reduce_shader_module_) != VK_SUCCESS) {
        error = "failed to create YAQA shader modules"; reset(); return false;
    }
    const VkPushConstantRange partial_range{VK_SHADER_STAGE_COMPUTE_BIT, 0, 20};
    const VkPushConstantRange reduce_range{VK_SHADER_STAGE_COMPUTE_BIT, 0, 8};
    const VkPipelineLayoutCreateInfo partial_layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        nullptr, 0, 1, &descriptor_layout_, 1, &partial_range};
    const VkPipelineLayoutCreateInfo reduce_layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        nullptr, 0, 1, &reduce_descriptor_layout_, 1, &reduce_range};
    if (vkCreatePipelineLayout(device_, &partial_layout_info, nullptr, &partial_pipeline_layout_) != VK_SUCCESS ||
        vkCreatePipelineLayout(device_, &reduce_layout_info, nullptr, &reduce_pipeline_layout_) != VK_SUCCESS) {
        error = "failed to create YAQA pipeline layouts"; reset(); return false;
    }
    const VkPipelineShaderStageCreateInfo partial_stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, partial_shader_module_, "main", nullptr};
    const VkPipelineShaderStageCreateInfo reduce_stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, reduce_shader_module_, "main", nullptr};
    const VkComputePipelineCreateInfo partial_pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        nullptr, 0, partial_stage, partial_pipeline_layout_, VK_NULL_HANDLE, -1};
    const VkComputePipelineCreateInfo reduce_pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        nullptr, 0, reduce_stage, reduce_pipeline_layout_, VK_NULL_HANDLE, -1};
    if (vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &partial_pipeline_info,
            nullptr, &partial_pipeline_) != VK_SUCCESS ||
        vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &reduce_pipeline_info,
            nullptr, &reduce_pipeline_) != VK_SUCCESS) {
        error = "failed to create YAQA compute pipelines"; reset(); return false;
    }
    const VkCommandPoolCreateInfo command_pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        nullptr, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, queue_family_};
    if (vkCreateCommandPool(device_, &command_pool_info, nullptr, &command_pool_) != VK_SUCCESS) {
        error = "failed to create YAQA command pool"; reset(); return false;
    }
    const VkCommandBufferAllocateInfo command_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        nullptr, command_pool_, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    if (vkAllocateCommandBuffers(device_, &command_info, &command_buffer_) != VK_SUCCESS) {
        error = "failed to allocate YAQA command buffer"; reset(); return false;
    }
    const VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, 0};
    if (vkCreateFence(device_, &fence_info, nullptr, &fence_) != VK_SUCCESS) {
        error = "failed to create YAQA fence"; reset(); return false;
    }
    return true;
}

bool astc_vulkan_yaqa_session::upload_traces(
        const std::vector<float> & input_trace,
        const std::vector<float> & output_trace, std::string & error) {
    if (!ready() || input_trace.size() != static_cast<size_t>(samples_) * columns_ ||
        output_trace.size() != static_cast<size_t>(samples_) * rows_) {
        error = "invalid YAQA trace dimensions";
        return false;
    }
    const bool input_ok = map_write(device_, input_trace_memory_, input_trace.data(),
        static_cast<VkDeviceSize>(input_trace.size()) * sizeof(float));
    const bool output_ok = map_write(device_, output_trace_memory_, output_trace.data(),
        static_cast<VkDeviceSize>(output_trace.size()) * sizeof(float));
    if (!input_ok || !output_ok) {
        error = "failed to upload YAQA traces";
        return false;
    }
    traces_resident_ = true;
    return true;
}

bool astc_vulkan_yaqa_session::run(
        const std::vector<float> & errors, uint32_t candidate_count,
        std::vector<float> & scores, std::string & error) {
    const size_t matrix_size = static_cast<size_t>(rows_) * columns_;
    if (!ready() || !traces_resident_ || candidate_count == 0 ||
        candidate_count > candidate_capacity_ || errors.size() !=
            static_cast<size_t>(candidate_count) * matrix_size) {
        error = "invalid YAQA candidate batch";
        return false;
    }
    if (scores.size() != candidate_count) scores.resize(candidate_count);
    if (!map_write(device_, errors_memory_, errors.data(),
            static_cast<VkDeviceSize>(errors.size()) * sizeof(float))) {
        error = "failed to upload YAQA candidate errors";
        return false;
    }
    if (vkResetFences(device_, 1, &fence_) != VK_SUCCESS ||
        vkResetCommandBuffer(command_buffer_, 0) != VK_SUCCESS) {
        error = "failed to reset YAQA command state";
        return false;
    }
    const VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
    if (vkBeginCommandBuffer(command_buffer_, &begin) != VK_SUCCESS) {
        error = "failed to begin YAQA command buffer";
        return false;
    }
    const uint32_t partial_push[5] = {candidate_count, rows_, columns_, samples_, pairs_per_candidate_};
    vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, partial_pipeline_);
    vkCmdBindDescriptorSets(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
        partial_pipeline_layout_, 0, 1, &descriptor_set_, 0, nullptr);
    vkCmdPushConstants(command_buffer_, partial_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
        0, sizeof(partial_push), partial_push);
    vkCmdDispatch(command_buffer_, candidate_count, (pairs_per_candidate_ + 63u) / 64u, 1);
    const VkBufferMemoryBarrier partial_barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, partials_buffer_, 0,
        static_cast<VkDeviceSize>(candidate_capacity_) * pairs_per_candidate_ * sizeof(float)};
    vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &partial_barrier, 0, nullptr);
    const uint32_t reduce_push[2] = {candidate_count, pairs_per_candidate_};
    vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, reduce_pipeline_);
    vkCmdBindDescriptorSets(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
        reduce_pipeline_layout_, 0, 1, &reduce_descriptor_set_, 0, nullptr);
    vkCmdPushConstants(command_buffer_, reduce_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
        0, sizeof(reduce_push), reduce_push);
    vkCmdDispatch(command_buffer_, candidate_count, 1, 1);
    if (vkEndCommandBuffer(command_buffer_) != VK_SUCCESS) {
        error = "failed to record YAQA command buffer";
        return false;
    }
    const VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr,
        nullptr, 1, &command_buffer_, 0, nullptr};
    if (vkQueueSubmit(queue_, 1, &submit, fence_) != VK_SUCCESS ||
        vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX) != VK_SUCCESS ||
        !map_read(device_, scores_memory_, scores.data(),
            static_cast<VkDeviceSize>(candidate_count) * sizeof(float))) {
        error = "YAQA GPU execution failed";
        return false;
    }
    return true;
}
