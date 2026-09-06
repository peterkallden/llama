#include "astc-vulkan-dispatch.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace {

bool create_host_buffer(VkPhysicalDevice physical_device, VkDevice device,
                        VkDeviceSize size, VkBufferUsageFlags usage,
                        VkBuffer & buffer, VkDeviceMemory & memory) {
    const VkBufferCreateInfo buffer_info{
        VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, size, usage,
        VK_SHARING_MODE_EXCLUSIVE, 0, nullptr};
    if (vkCreateBuffer(device, &buffer_info, nullptr, &buffer) != VK_SUCCESS) return false;
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device, buffer, &requirements);
    uint32_t memory_type = astc_vulkan_find_memory_type(
        physical_device, requirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (memory_type == std::numeric_limits<uint32_t>::max()) {
        memory_type = astc_vulkan_find_memory_type(
            physical_device, requirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    }
    if (memory_type == std::numeric_limits<uint32_t>::max()) {
        vkDestroyBuffer(device, buffer, nullptr);
        buffer = VK_NULL_HANDLE;
        return false;
    }
    const VkMemoryAllocateInfo allocation{
        VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, requirements.size, memory_type};
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

bool map_write(VkDevice device, VkDeviceMemory memory, const void * data, VkDeviceSize size) {
    void * mapped = nullptr;
    if (vkMapMemory(device, memory, 0, size, 0, &mapped) != VK_SUCCESS) return false;
    std::memcpy(mapped, data, static_cast<size_t>(size));
    const VkMappedMemoryRange range{
        VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, nullptr, memory, 0, VK_WHOLE_SIZE};
    const VkResult flush = vkFlushMappedMemoryRanges(device, 1, &range);
    vkUnmapMemory(device, memory);
    return flush == VK_SUCCESS;
}

} // namespace

astc_vulkan_matvec_session::~astc_vulkan_matvec_session() {
    reset();
}

void astc_vulkan_matvec_session::reset() {
    if (device_ == VK_NULL_HANDLE) return;
    // Native sessions borrow the backend device and are reset as part of the
    // graph/resource owner lifecycle. The owner, not this session, provides
    // the final device synchronization in that mode.
    if (!native_mode_) vkDeviceWaitIdle(device_);
    if (fence_ != VK_NULL_HANDLE) vkDestroyFence(device_, fence_, nullptr);
    if (command_pool_ != VK_NULL_HANDLE) vkDestroyCommandPool(device_, command_pool_, nullptr);
    if (pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(device_, pipeline_, nullptr);
    if (shader_module_ != VK_NULL_HANDLE) vkDestroyShaderModule(device_, shader_module_, nullptr);
    if (pipeline_layout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
    if (descriptor_pool_ != VK_NULL_HANDLE) vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
    if (descriptor_layout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device_, descriptor_layout_, nullptr);
    destroy_buffer(device_, output_buffer_, output_memory_);
    destroy_buffer(device_, activation_buffer_, activation_memory_);
    physical_device_ = VK_NULL_HANDLE;
    device_ = VK_NULL_HANDLE;
    queue_ = VK_NULL_HANDLE;
    queue_family_ = UINT32_MAX;
    width_ = height_ = texture_height_ = samples_ = 0;
    descriptor_set_ = VK_NULL_HANDLE;
    command_buffer_ = VK_NULL_HANDLE;
    native_mode_ = false;
}

bool astc_vulkan_matvec_session::init(
        VkPhysicalDevice physical_device, VkDevice device, VkQueue queue,
        uint32_t queue_family, const astc_vulkan_tensor_session & tensor,
        const std::vector<uint32_t> & spirv, uint32_t width, uint32_t height,
        uint32_t samples, std::string & error) {
    return init_impl(physical_device, device, queue, queue_family, tensor, spirv,
                     width, height, samples, false, error);
}

bool astc_vulkan_matvec_session::init_native(
        VkPhysicalDevice physical_device, VkDevice device, VkQueue queue,
        uint32_t queue_family, const astc_vulkan_tensor_session & tensor,
        const std::vector<uint32_t> & spirv, uint32_t width, uint32_t height,
        uint32_t samples, std::string & error) {
    return init_impl(physical_device, device, queue, queue_family, tensor, spirv,
                     width, height, samples, true, error);
}

bool astc_vulkan_matvec_session::init_impl(
        VkPhysicalDevice physical_device, VkDevice device, VkQueue queue,
        uint32_t queue_family, const astc_vulkan_tensor_session & tensor,
        const std::vector<uint32_t> & spirv, uint32_t width, uint32_t height,
        uint32_t samples, bool native_mode, std::string & error) {
    reset();
    if (physical_device == VK_NULL_HANDLE || device == VK_NULL_HANDLE ||
        queue == VK_NULL_HANDLE || queue_family == UINT32_MAX || spirv.empty() ||
        width == 0 || height == 0 || samples == 0 ||
        tensor.texture().view() == VK_NULL_HANDLE || tensor.texture().sampler() == VK_NULL_HANDLE ||
        tensor.texture().width() != width || tensor.texture().height() == 0 ||
        tensor.texture().height() > height) {
        error = "invalid ASTC matvec session configuration";
        return false;
    }

    physical_device_ = physical_device;
    device_ = device;
    queue_ = queue;
    queue_family_ = queue_family;
    width_ = width;
    height_ = height;
    texture_height_ = tensor.texture().height();
    samples_ = samples;
    native_mode_ = native_mode;
    const VkDeviceSize activation_bytes = static_cast<VkDeviceSize>(samples) * width * sizeof(float);
    const VkDeviceSize output_bytes = static_cast<VkDeviceSize>(samples) * height * sizeof(float);
    if (!native_mode_) {
        if (!create_host_buffer(physical_device_, device_, activation_bytes,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                activation_buffer_, activation_memory_) ||
            !create_host_buffer(physical_device_, device_, output_bytes,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                output_buffer_, output_memory_)) {
            error = "failed to allocate ASTC matvec buffers";
            reset();
            return false;
        }
    }

    const VkDescriptorSetLayoutBinding bindings[3] = {
        {static_cast<uint32_t>(astc_vulkan_descriptor_binding::kWeights),
         VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {static_cast<uint32_t>(astc_vulkan_descriptor_binding::kActivations),
         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {static_cast<uint32_t>(astc_vulkan_descriptor_binding::kOutput),
         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    const VkDescriptorSetLayoutCreateInfo layout_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 3, bindings};
    if (vkCreateDescriptorSetLayout(device_, &layout_info, nullptr, &descriptor_layout_) != VK_SUCCESS) {
        error = "failed to create ASTC matvec descriptor layout";
        reset();
        return false;
    }
    const VkDescriptorPoolSize pool_sizes[2] = {
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2},
    };
    const VkDescriptorPoolCreateInfo pool_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0, 1, 2, pool_sizes};
    if (vkCreateDescriptorPool(device_, &pool_info, nullptr, &descriptor_pool_) != VK_SUCCESS) {
        error = "failed to create ASTC matvec descriptor pool";
        reset();
        return false;
    }
    const VkDescriptorSetAllocateInfo set_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, descriptor_pool_, 1,
        &descriptor_layout_};
    if (vkAllocateDescriptorSets(device_, &set_info, &descriptor_set_) != VK_SUCCESS) {
        error = "failed to allocate ASTC matvec descriptor set";
        reset();
        return false;
    }
    const VkDescriptorImageInfo image_info{
        tensor.texture().sampler(), tensor.texture().view(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    const VkDescriptorBufferInfo activation_info{activation_buffer_, 0, activation_bytes};
    const VkDescriptorBufferInfo output_info{output_buffer_, 0, output_bytes};
    const VkWriteDescriptorSet writes[3] = {
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor_set_,
         static_cast<uint32_t>(astc_vulkan_descriptor_binding::kWeights), 0, 1,
         VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &image_info, nullptr, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor_set_,
         static_cast<uint32_t>(astc_vulkan_descriptor_binding::kActivations), 0, 1,
         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &activation_info, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor_set_,
         static_cast<uint32_t>(astc_vulkan_descriptor_binding::kOutput), 0, 1,
         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &output_info, nullptr},
    };
    vkUpdateDescriptorSets(device_, native_mode_ ? 1 : 3, writes, 0, nullptr);

    const VkShaderModuleCreateInfo shader_info{
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
        spirv.size() * sizeof(uint32_t), spirv.data()};
    if (vkCreateShaderModule(device_, &shader_info, nullptr, &shader_module_) != VK_SUCCESS) {
        error = "failed to create ASTC matvec shader module";
        reset();
        return false;
    }
    const VkPushConstantRange push_range{
        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(astc_vulkan_matvec_push_constants)};
    const VkPipelineLayoutCreateInfo pipeline_layout_info{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &descriptor_layout_,
        1, &push_range};
    if (vkCreatePipelineLayout(device_, &pipeline_layout_info, nullptr, &pipeline_layout_) != VK_SUCCESS) {
        error = "failed to create ASTC matvec pipeline layout";
        reset();
        return false;
    }
    const VkPipelineShaderStageCreateInfo stage{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
        VK_SHADER_STAGE_COMPUTE_BIT, shader_module_, "main", nullptr};
    const VkComputePipelineCreateInfo pipeline_info{
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, nullptr, 0, stage,
        pipeline_layout_, VK_NULL_HANDLE, -1};
    if (vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipeline_info,
                                 nullptr, &pipeline_) != VK_SUCCESS) {
        error = "failed to create ASTC matvec pipeline";
        reset();
        return false;
    }
    if (native_mode_) {
        error.clear();
        return true;
    }
    const VkCommandPoolCreateInfo command_pool_info{
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr,
        VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        queue_family_};
    if (vkCreateCommandPool(device_, &command_pool_info, nullptr, &command_pool_) != VK_SUCCESS) {
        error = "failed to create ASTC matvec command pool";
        reset();
        return false;
    }
    const VkCommandBufferAllocateInfo command_buffer_info{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, command_pool_,
        VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    if (vkAllocateCommandBuffers(device_, &command_buffer_info, &command_buffer_) != VK_SUCCESS) {
        error = "failed to allocate ASTC matvec command buffer";
        reset();
        return false;
    }
    const VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, 0};
    if (vkCreateFence(device_, &fence_info, nullptr, &fence_) != VK_SUCCESS) {
        error = "failed to create ASTC matvec fence";
        reset();
        return false;
    }
    error.clear();
    return true;
}

bool astc_vulkan_matvec_session::run(
        const std::vector<float> & activations,
        const astc_vulkan_reconstruction & reconstruction,
        std::vector<float> & output, std::string & error) {
    return run_band(activations, reconstruction, 0, height_, output, error);
}

bool astc_vulkan_matvec_session::rebind_texture(
        const astc_vulkan_tensor_session & tensor, std::string & error) {
    if (device_ == VK_NULL_HANDLE || tensor.texture().view() == VK_NULL_HANDLE ||
        tensor.texture().sampler() == VK_NULL_HANDLE || tensor.texture().width() != width_ ||
        tensor.texture().height() == 0 || tensor.texture().height() > height_) {
        error = "invalid ASTC matvec texture rebind";
        return false;
    }
    if (vkDeviceWaitIdle(device_) != VK_SUCCESS) {
        error = "ASTC matvec texture rebind could not quiesce the device";
        return false;
    }
    const VkDescriptorImageInfo image_info{
        tensor.texture().sampler(), tensor.texture().view(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    const VkWriteDescriptorSet write{
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor_set_,
        static_cast<uint32_t>(astc_vulkan_descriptor_binding::kWeights), 0, 1,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &image_info, nullptr, nullptr};
    vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
    texture_height_ = tensor.texture().height();
    error.clear();
    return true;
}

bool astc_vulkan_matvec_session::run_band(
        const std::vector<float> & activations,
        const astc_vulkan_reconstruction & reconstruction,
        uint32_t row_base, uint32_t band_height,
        std::vector<float> & output, std::string & error) {
    if (device_ == VK_NULL_HANDLE || activations.size() !=
            static_cast<size_t>(samples_) * width_ || band_height == 0 ||
        row_base > height_ || band_height > height_ - row_base ||
        texture_height_ < band_height) {
        error = "invalid ASTC matvec band run inputs";
        return false;
    }
    const VkDeviceSize activation_bytes = static_cast<VkDeviceSize>(activations.size()) * sizeof(float);
    const VkDeviceSize output_bytes = static_cast<VkDeviceSize>(samples_) * height_ * sizeof(float);
    if (!map_write(device_, activation_memory_, activations.data(), activation_bytes)) {
        error = "failed to upload ASTC matvec band activations";
        return false;
    }
    if (vkResetFences(device_, 1, &fence_) != VK_SUCCESS ||
        vkResetCommandBuffer(command_buffer_, 0) != VK_SUCCESS) {
        error = "failed to reset ASTC matvec band synchronization";
        return false;
    }
    const VkCommandBufferBeginInfo begin_info{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
    if (vkBeginCommandBuffer(command_buffer_, &begin_info) != VK_SUCCESS) {
        error = "failed to begin ASTC matvec band command buffer";
        return false;
    }
    vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    vkCmdBindDescriptorSets(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline_layout_, 0, 1, &descriptor_set_, 0, nullptr);
    for (uint32_t sample = 0; sample < samples_; ++sample) {
        const astc_vulkan_matvec_push_constants constants{
            width_, band_height, sample, row_base, height_, reconstruction.scale_l,
            reconstruction.scale_a, reconstruction.offset};
        vkCmdPushConstants(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(constants), &constants);
        vkCmdDispatch(command_buffer_, band_height, 1, 1);
    }
    const VkBufferMemoryBarrier output_barrier{
        VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_HOST_READ_BIT, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        output_buffer_, 0, output_bytes};
    vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1,
                         &output_barrier, 0, nullptr);
    if (vkEndCommandBuffer(command_buffer_) != VK_SUCCESS) {
        error = "failed to end ASTC matvec band command buffer";
        return false;
    }
    const VkSubmitInfo submit_info{
        VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1,
        &command_buffer_, 0, nullptr};
    if (vkQueueSubmit(queue_, 1, &submit_info, fence_) != VK_SUCCESS ||
        vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
        error = "failed to submit ASTC matvec band dispatch";
        return false;
    }
    void * mapped = nullptr;
    if (vkMapMemory(device_, output_memory_, 0, output_bytes, 0, &mapped) != VK_SUCCESS) {
        error = "failed to map ASTC matvec band output";
        return false;
    }
    const VkMappedMemoryRange invalidate_range{
        VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, nullptr, output_memory_, 0, VK_WHOLE_SIZE};
    const VkResult invalidate = vkInvalidateMappedMemoryRanges(device_, 1, &invalidate_range);
    if (invalidate == VK_SUCCESS) {
        output.assign(static_cast<size_t>(samples_) * band_height, 0.0f);
        const float * source = static_cast<const float *>(mapped);
        for (uint32_t sample = 0; sample < samples_; ++sample) {
            std::memcpy(output.data() + static_cast<size_t>(sample) * band_height,
                        source + static_cast<size_t>(sample) * height_ + row_base,
                        static_cast<size_t>(band_height) * sizeof(float));
        }
    }
    vkUnmapMemory(device_, output_memory_);
    if (invalidate != VK_SUCCESS) {
        error = "failed to invalidate ASTC matvec band output";
        return false;
    }
    error.clear();
    return true;
}

bool astc_vulkan_matvec_session::record_external(
        VkCommandBuffer command_buffer, VkBuffer activation_buffer,
        VkDeviceSize activation_offset, VkDeviceSize activation_size,
        VkBuffer output_buffer, VkDeviceSize output_offset,
        VkDeviceSize output_size, const astc_vulkan_reconstruction & reconstruction,
        uint32_t row_base, uint32_t band_height, std::string & error) {
    if (device_ == VK_NULL_HANDLE || pipeline_ == VK_NULL_HANDLE ||
        descriptor_set_ == VK_NULL_HANDLE || command_buffer == VK_NULL_HANDLE ||
        activation_buffer == VK_NULL_HANDLE || output_buffer == VK_NULL_HANDLE ||
        activation_size == 0 || output_size == 0 || band_height == 0 ||
        row_base > height_ || band_height > height_ - row_base ||
        texture_height_ < band_height) {
        error = "invalid ASTC external matvec recording inputs";
        return false;
    }
    if (activation_size < static_cast<VkDeviceSize>(samples_) * width_ * sizeof(float) ||
        output_size < static_cast<VkDeviceSize>(samples_) * height_ * sizeof(float)) {
        error = "ASTC external matvec buffers are smaller than the dispatch shape";
        return false;
    }
    const VkDescriptorBufferInfo activation_info{activation_buffer, activation_offset,
                                                  static_cast<VkDeviceSize>(samples_) * width_ * sizeof(float)};
    const VkDescriptorBufferInfo output_info{output_buffer, output_offset,
                                              static_cast<VkDeviceSize>(samples_) * height_ * sizeof(float)};
    const VkWriteDescriptorSet writes[2] = {
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor_set_,
         static_cast<uint32_t>(astc_vulkan_descriptor_binding::kActivations), 0, 1,
         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &activation_info, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor_set_,
         static_cast<uint32_t>(astc_vulkan_descriptor_binding::kOutput), 0, 1,
         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &output_info, nullptr},
    };
    vkUpdateDescriptorSets(device_, 2, writes, 0, nullptr);

    const VkBufferMemoryBarrier input_barrier{
        VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, nullptr,
        VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        activation_buffer, activation_offset, activation_info.range};
    const VkBufferMemoryBarrier output_before{
        VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, nullptr,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_SHADER_WRITE_BIT, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        output_buffer, output_offset, output_info.range};
    const VkBufferMemoryBarrier before_barriers[2] = { input_barrier, output_before };
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 2,
                         before_barriers, 0, nullptr);
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline_layout_, 0, 1, &descriptor_set_, 0, nullptr);
    for (uint32_t sample = 0; sample < samples_; ++sample) {
        const astc_vulkan_matvec_push_constants constants{
            width_, band_height, sample, row_base, height_, reconstruction.scale_l,
            reconstruction.scale_a, reconstruction.offset};
        vkCmdPushConstants(command_buffer, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(constants), &constants);
        vkCmdDispatch(command_buffer, band_height, 1, 1);
    }
    const VkBufferMemoryBarrier output_after{
        VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, nullptr,
        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        output_buffer, output_offset, output_info.range};
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 1,
                         &output_after, 0, nullptr);
    error.clear();
    return true;
}
