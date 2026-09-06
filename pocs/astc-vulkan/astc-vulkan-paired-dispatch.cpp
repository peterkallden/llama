#include "astc-vulkan-paired-dispatch.h"

#include "astc-vulkan-paired-layout.h"

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

astc_vulkan_paired_matvec_session::~astc_vulkan_paired_matvec_session() {
    reset();
}

void astc_vulkan_paired_matvec_session::reset() {
    if (device_ != VK_NULL_HANDLE) {
        if (!native_mode_) vkDeviceWaitIdle(device_);
        if (fence_ != VK_NULL_HANDLE) vkDestroyFence(device_, fence_, nullptr);
        if (command_pool_ != VK_NULL_HANDLE) vkDestroyCommandPool(device_, command_pool_, nullptr);
        if (pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(device_, pipeline_, nullptr);
        if (shader_module_ != VK_NULL_HANDLE) vkDestroyShaderModule(device_, shader_module_, nullptr);
        if (pipeline_layout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
        if (descriptor_pool_ != VK_NULL_HANDLE) vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
        if (descriptor_layout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device_, descriptor_layout_, nullptr);
        destroy_buffer(device_, row_scale_buffer_, row_scale_memory_);
        destroy_buffer(device_, layout_buffer_, layout_memory_);
        destroy_buffer(device_, output_buffer_, output_memory_);
        destroy_buffer(device_, activation_buffer_, activation_memory_);
    }
    physical_device_ = VK_NULL_HANDLE;
    device_ = VK_NULL_HANDLE;
    queue_ = VK_NULL_HANDLE;
    queue_family_ = UINT32_MAX;
    width_ = logical_height_ = block_width_ = block_height_ = storage_height_ =
        texture_row_base_ = layout_map_words_ = samples_ = 0;
    paired_semantic_ = row_scale_count_ = 0;
    descriptor_set_ = VK_NULL_HANDLE;
    command_buffer_ = VK_NULL_HANDLE;
    native_mode_ = false;
}

bool astc_vulkan_paired_matvec_session::init(
        VkPhysicalDevice physical_device, VkDevice device, VkQueue queue,
        uint32_t queue_family, const astc_vulkan_tensor_session & tensor,
        const std::vector<uint8_t> & layout_map, const std::vector<uint32_t> & spirv,
        uint32_t width, uint32_t logical_height, uint32_t samples, std::string & error,
        astc_vulkan_paired_semantic semantic, const std::vector<float> & row_scales,
        uint32_t storage_height, uint32_t texture_row_base) {
    return init_impl(physical_device, device, queue, queue_family, tensor, layout_map, spirv,
                     width, logical_height, samples, error, semantic, row_scales,
                     storage_height, texture_row_base, false);
}

bool astc_vulkan_paired_matvec_session::init_native(
        VkPhysicalDevice physical_device, VkDevice device, VkQueue queue,
        uint32_t queue_family, const astc_vulkan_tensor_session & tensor,
        const std::vector<uint8_t> & layout_map, const std::vector<uint32_t> & spirv,
        uint32_t width, uint32_t logical_height, uint32_t samples, std::string & error,
        astc_vulkan_paired_semantic semantic, const std::vector<float> & row_scales,
        uint32_t storage_height, uint32_t texture_row_base) {
    return init_impl(physical_device, device, queue, queue_family, tensor, layout_map, spirv,
                     width, logical_height, samples, error, semantic, row_scales,
                     storage_height, texture_row_base, true);
}

bool astc_vulkan_paired_matvec_session::init_impl(
        VkPhysicalDevice physical_device, VkDevice device, VkQueue queue,
        uint32_t queue_family, const astc_vulkan_tensor_session & tensor,
        const std::vector<uint8_t> & layout_map, const std::vector<uint32_t> & spirv,
        uint32_t width, uint32_t logical_height, uint32_t samples, std::string & error,
        astc_vulkan_paired_semantic semantic, const std::vector<float> & row_scales,
        uint32_t storage_height, uint32_t texture_row_base, bool native_mode) {
    reset();
    const astc_vulkan_tensor_record & record = tensor.record();
    const astc_vulkan_format_info format = astc_vulkan_format(record.footprint);
    const uint32_t expected_storage_height = astc_vulkan_paired_storage_height(logical_height);
    if (storage_height == 0) storage_height = expected_storage_height;
    if (physical_device == VK_NULL_HANDLE || device == VK_NULL_HANDLE || queue == VK_NULL_HANDLE ||
        queue_family == UINT32_MAX || spirv.empty() || width == 0 || logical_height == 0 || samples == 0 ||
        record.representation != astc_vulkan_representation::kPairedD2 ||
        record.width != width || record.height != logical_height || format.block_width == 0 ||
        layout_map.size() != astc_vulkan_paired_layout_bytes(record.footprint, width, logical_height) ||
        tensor.texture().view() == VK_NULL_HANDLE || tensor.texture().sampler() == VK_NULL_HANDLE ||
        tensor.texture().width() != width || storage_height == 0 ||
        storage_height > expected_storage_height ||
        texture_row_base > expected_storage_height - storage_height ||
        tensor.texture().height() != storage_height) {
        error = "invalid paired-D2 ASTC matvec session configuration";
        return false;
    }
    physical_device_ = physical_device;
    device_ = device;
    queue_ = queue;
    queue_family_ = queue_family;
    width_ = width;
    logical_height_ = logical_height;
    block_width_ = format.block_width;
    block_height_ = format.block_height;
    storage_height_ = storage_height;
    texture_row_base_ = texture_row_base;
    layout_map_words_ = static_cast<uint32_t>(layout_map.size() / sizeof(uint32_t));
    paired_semantic_ = static_cast<uint32_t>(semantic);
    if (!row_scales.empty() && row_scales.size() != logical_height) {
        error = "paired-D2 row-scale count does not match logical height";
        reset();
        return false;
    }
    row_scale_count_ = logical_height;
    samples_ = samples;
    native_mode_ = native_mode;
    const VkDeviceSize activation_bytes = static_cast<VkDeviceSize>(samples) * width * sizeof(float);
    const VkDeviceSize output_bytes = static_cast<VkDeviceSize>(samples) * logical_height * sizeof(float);
    const VkDeviceSize layout_bytes = static_cast<VkDeviceSize>(layout_map.size());
    std::vector<float> effective_scales = row_scales;
    if (effective_scales.empty()) effective_scales.assign(logical_height, 1.0f);
    const VkDeviceSize row_scale_bytes = static_cast<VkDeviceSize>(effective_scales.size() * sizeof(float));
    if ((!native_mode_ &&
         (!create_host_buffer(physical_device_, device_, activation_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              activation_buffer_, activation_memory_) ||
          !create_host_buffer(physical_device_, device_, output_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              output_buffer_, output_memory_))) ||
        !create_host_buffer(physical_device_, device_, layout_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                            layout_buffer_, layout_memory_) ||
        !create_host_buffer(physical_device_, device_, row_scale_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                            row_scale_buffer_, row_scale_memory_) ||
        !map_write(device_, layout_memory_, layout_map.data(), layout_bytes)) {
        error = "failed to allocate paired-D2 ASTC matvec buffers";
        reset();
        return false;
    }
    if (!map_write(device_, row_scale_memory_, effective_scales.data(), row_scale_bytes)) {
        error = "failed to upload paired-D2 row scales";
        reset();
        return false;
    }
    const VkDescriptorSetLayoutBinding bindings[5] = {
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    const VkDescriptorSetLayoutCreateInfo layout_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 5, bindings};
    if (vkCreateDescriptorSetLayout(device_, &layout_info, nullptr, &descriptor_layout_) != VK_SUCCESS) {
        error = "failed to create paired-D2 ASTC descriptor layout";
        reset();
        return false;
    }
    const VkDescriptorPoolSize pool_sizes[2] = {
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4}};
    const VkDescriptorPoolCreateInfo pool_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0, 1, 2, pool_sizes};
    if (vkCreateDescriptorPool(device_, &pool_info, nullptr, &descriptor_pool_) != VK_SUCCESS) {
        error = "failed to create paired-D2 ASTC descriptor pool";
        reset();
        return false;
    }
    const VkDescriptorSetAllocateInfo set_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, descriptor_pool_, 1, &descriptor_layout_};
    if (vkAllocateDescriptorSets(device_, &set_info, &descriptor_set_) != VK_SUCCESS) {
        error = "failed to allocate paired-D2 ASTC descriptor set";
        reset();
        return false;
    }
    const VkDescriptorImageInfo image_info{
        tensor.texture().sampler(), tensor.texture().view(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    const VkDescriptorBufferInfo activation_info{activation_buffer_, 0, activation_bytes};
    const VkDescriptorBufferInfo output_info{output_buffer_, 0, output_bytes};
    const VkDescriptorBufferInfo paired_layout_info{layout_buffer_, 0, layout_bytes};
    const VkDescriptorBufferInfo row_scale_info{row_scale_buffer_, 0, row_scale_bytes};
    const VkWriteDescriptorSet writes[5] = {
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor_set_, 0, 0, 1,
         VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &image_info, nullptr, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor_set_, 1, 0, 1,
         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &activation_info, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor_set_, 2, 0, 1,
         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &output_info, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor_set_, 3, 0, 1,
         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &paired_layout_info, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor_set_, 4, 0, 1,
         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &row_scale_info, nullptr},
    };
    if (native_mode_) {
        // Activation/output are intentionally left unbound until
        // record_external() supplies the caller-owned buffer views. Keep the
        // image, layout map and row-scale metadata descriptors initialized.
        const VkWriteDescriptorSet native_writes[3] = {writes[0], writes[3], writes[4]};
        vkUpdateDescriptorSets(device_, 3, native_writes, 0, nullptr);
    } else {
        vkUpdateDescriptorSets(device_, 5, writes, 0, nullptr);
    }
    const VkShaderModuleCreateInfo shader_info{
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
        spirv.size() * sizeof(uint32_t), spirv.data()};
    if (vkCreateShaderModule(device_, &shader_info, nullptr, &shader_module_) != VK_SUCCESS) {
        error = "failed to create paired-D2 ASTC shader module";
        reset();
        return false;
    }
    const VkPushConstantRange push_range{VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                          sizeof(astc_vulkan_paired_matvec_push_constants)};
    const VkPipelineLayoutCreateInfo pipeline_layout_info{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &descriptor_layout_, 1, &push_range};
    if (vkCreatePipelineLayout(device_, &pipeline_layout_info, nullptr, &pipeline_layout_) != VK_SUCCESS) {
        error = "failed to create paired-D2 ASTC pipeline layout";
        reset();
        return false;
    }
    const VkPipelineShaderStageCreateInfo stage{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
        VK_SHADER_STAGE_COMPUTE_BIT, shader_module_, "main", nullptr};
    const VkComputePipelineCreateInfo pipeline_info{
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, nullptr, 0, stage, pipeline_layout_, VK_NULL_HANDLE, -1};
    if (vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline_) != VK_SUCCESS) {
        error = "failed to create paired-D2 ASTC matvec pipeline";
        reset();
        return false;
    }
    if (native_mode_) {
        error.clear();
        return true;
    }
    const VkCommandPoolCreateInfo command_pool_info{
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr,
        VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, queue_family_};
    if (vkCreateCommandPool(device_, &command_pool_info, nullptr, &command_pool_) != VK_SUCCESS) {
        error = "failed to create paired-D2 command pool";
        reset();
        return false;
    }
    const VkCommandBufferAllocateInfo command_buffer_info{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, command_pool_, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    if (vkAllocateCommandBuffers(device_, &command_buffer_info, &command_buffer_) != VK_SUCCESS) {
        error = "failed to allocate paired-D2 command buffer";
        reset();
        return false;
    }
    const VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, 0};
    if (vkCreateFence(device_, &fence_info, nullptr, &fence_) != VK_SUCCESS) {
        error = "failed to create paired-D2 fence";
        reset();
        return false;
    }
    error.clear();
    return true;
}

bool astc_vulkan_paired_matvec_session::run(
        const std::vector<float> & activations,
        const astc_vulkan_reconstruction & reconstruction,
        std::vector<float> & output, std::string & error) {
    if (!ready() || activations.size() != static_cast<size_t>(samples_) * width_) {
        error = "invalid paired-D2 ASTC matvec run inputs";
        return false;
    }
    return run_band(activations, reconstruction, 0, logical_height_, output, error);
}

bool astc_vulkan_paired_matvec_session::rebind_texture(
        const astc_vulkan_tensor_session & tensor, uint32_t storage_height,
        uint32_t texture_row_base, std::string & error) {
    const uint32_t expected_storage_height = astc_vulkan_paired_storage_height(logical_height_);
    if (!ready() || tensor.texture().view() == VK_NULL_HANDLE ||
        tensor.texture().sampler() == VK_NULL_HANDLE || tensor.texture().width() != width_ ||
        storage_height == 0 || storage_height > expected_storage_height ||
        texture_row_base > expected_storage_height - storage_height ||
        tensor.texture().height() != storage_height) {
        error = "invalid paired-D2 ASTC matvec texture rebind";
        return false;
    }
    if (vkDeviceWaitIdle(device_) != VK_SUCCESS) {
        error = "paired-D2 ASTC texture rebind could not quiesce the device";
        return false;
    }
    const VkDescriptorImageInfo image_info{
        tensor.texture().sampler(), tensor.texture().view(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    const VkWriteDescriptorSet write{
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor_set_, 0, 0, 1,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &image_info, nullptr, nullptr};
    vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
    storage_height_ = storage_height;
    texture_row_base_ = texture_row_base;
    error.clear();
    return true;
}

bool astc_vulkan_paired_matvec_session::run_band(
        const std::vector<float> & activations,
        const astc_vulkan_reconstruction & reconstruction,
        uint32_t row_base, uint32_t band_height,
        std::vector<float> & output, std::string & error) {
    const uint32_t required_storage_height = (band_height + 1u) / 2u;
    const uint32_t expected_storage_height = astc_vulkan_paired_storage_height(logical_height_);
    if (!ready() || activations.size() != static_cast<size_t>(samples_) * width_ ||
        band_height == 0 || row_base > logical_height_ ||
        band_height > logical_height_ - row_base ||
        storage_height_ < required_storage_height ||
        texture_row_base_ > expected_storage_height - required_storage_height) {
        error = "invalid paired-D2 ASTC matvec band run inputs";
        return false;
    }
    const VkDeviceSize activation_bytes = static_cast<VkDeviceSize>(activations.size()) * sizeof(float);
    const VkDeviceSize output_bytes = static_cast<VkDeviceSize>(samples_) * logical_height_ * sizeof(float);
    if (!map_write(device_, activation_memory_, activations.data(), activation_bytes) ||
        vkResetFences(device_, 1, &fence_) != VK_SUCCESS || vkResetCommandBuffer(command_buffer_, 0) != VK_SUCCESS) {
        error = "failed to prepare paired-D2 ASTC dispatch";
        return false;
    }
    const VkCommandBufferBeginInfo begin_info{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
    if (vkBeginCommandBuffer(command_buffer_, &begin_info) != VK_SUCCESS) {
        error = "failed to begin paired-D2 command buffer";
        return false;
    }
    vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    vkCmdBindDescriptorSets(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline_layout_, 0, 1, &descriptor_set_, 0, nullptr);
    for (uint32_t sample = 0; sample < samples_; ++sample) {
        const astc_vulkan_paired_matvec_push_constants constants{
            width_, logical_height_, sample, block_width_, block_height_, layout_map_words_,
            paired_semantic_, row_scale_count_, band_height, row_base, texture_row_base_,
            reconstruction.scale_l, reconstruction.offset};
        vkCmdPushConstants(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(constants), &constants);
        vkCmdDispatch(command_buffer_, band_height, 1, 1);
    }
    const VkBufferMemoryBarrier output_barrier{
        VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_HOST_READ_BIT, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        output_buffer_, 0, output_bytes};
    vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &output_barrier, 0, nullptr);
    if (vkEndCommandBuffer(command_buffer_) != VK_SUCCESS) {
        error = "failed to end paired-D2 command buffer";
        return false;
    }
    const VkSubmitInfo submit_info{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1,
                                   &command_buffer_, 0, nullptr};
    if (vkQueueSubmit(queue_, 1, &submit_info, fence_) != VK_SUCCESS ||
        vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
        error = "failed to submit paired-D2 ASTC dispatch";
        return false;
    }
    void * mapped = nullptr;
    if (vkMapMemory(device_, output_memory_, 0, output_bytes, 0, &mapped) != VK_SUCCESS) {
        error = "failed to map paired-D2 output";
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
                        source + static_cast<size_t>(sample) * logical_height_ + row_base,
                        static_cast<size_t>(band_height) * sizeof(float));
        }
    }
    vkUnmapMemory(device_, output_memory_);
    if (invalidate != VK_SUCCESS) {
        error = "failed to invalidate paired-D2 output";
        return false;
    }
    error.clear();
    return true;
}

bool astc_vulkan_paired_matvec_session::record_external(
        VkCommandBuffer command_buffer, VkBuffer activation_buffer,
        VkDeviceSize activation_offset, VkDeviceSize activation_size,
        VkBuffer output_buffer, VkDeviceSize output_offset,
        VkDeviceSize output_size, const astc_vulkan_reconstruction & reconstruction,
        uint32_t row_base, uint32_t band_height, std::string & error) {
    const uint32_t required_storage_height = (band_height + 1u) / 2u;
    const uint32_t expected_storage_height = astc_vulkan_paired_storage_height(logical_height_);
    if (device_ == VK_NULL_HANDLE || pipeline_ == VK_NULL_HANDLE ||
        descriptor_set_ == VK_NULL_HANDLE || command_buffer == VK_NULL_HANDLE ||
        activation_buffer == VK_NULL_HANDLE || output_buffer == VK_NULL_HANDLE ||
        activation_size < static_cast<VkDeviceSize>(samples_) * width_ * sizeof(float) ||
        output_size < static_cast<VkDeviceSize>(samples_) * logical_height_ * sizeof(float) ||
        band_height == 0 || row_base > logical_height_ ||
        band_height > logical_height_ - row_base || storage_height_ < required_storage_height ||
        texture_row_base_ > expected_storage_height - required_storage_height) {
        error = "invalid paired-D2 external matvec recording inputs";
        return false;
    }
    const VkDescriptorBufferInfo activation_info{
        activation_buffer, activation_offset,
        static_cast<VkDeviceSize>(samples_) * width_ * sizeof(float)};
    const VkDescriptorBufferInfo output_info{
        output_buffer, output_offset,
        static_cast<VkDeviceSize>(samples_) * logical_height_ * sizeof(float)};
    const VkWriteDescriptorSet writes[2] = {
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor_set_, 1, 0, 1,
         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &activation_info, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor_set_, 2, 0, 1,
         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &output_info, nullptr},
    };
    vkUpdateDescriptorSets(device_, 2, writes, 0, nullptr);
    const VkBufferMemoryBarrier input_barrier{
        VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, nullptr,
        VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, activation_buffer,
        activation_offset, activation_info.range};
    const VkBufferMemoryBarrier output_before{
        VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, nullptr,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_WRITE_BIT,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, output_buffer,
        output_offset, output_info.range};
    const VkBufferMemoryBarrier before_barriers[2] = {input_barrier, output_before};
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 2,
                         before_barriers, 0, nullptr);
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline_layout_, 0, 1, &descriptor_set_, 0, nullptr);
    for (uint32_t sample = 0; sample < samples_; ++sample) {
        const astc_vulkan_paired_matvec_push_constants constants{
            width_, logical_height_, sample, block_width_, block_height_, layout_map_words_,
            paired_semantic_, row_scale_count_, band_height, row_base, texture_row_base_,
            reconstruction.scale_l, reconstruction.offset};
        vkCmdPushConstants(command_buffer, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(constants), &constants);
        vkCmdDispatch(command_buffer, band_height, 1, 1);
    }
    const VkBufferMemoryBarrier output_after{
        VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, nullptr,
        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, output_buffer,
        output_offset, output_info.range};
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 1,
                         &output_after, 0, nullptr);
    error.clear();
    return true;
}
