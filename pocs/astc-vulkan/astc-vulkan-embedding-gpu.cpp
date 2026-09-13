#include "astc-vulkan-embedding-gpu.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace {

struct push_constants {
    uint32_t dimensions;
    uint32_t vocabulary;
    uint32_t tiles_per_token;
    uint32_t token_columns;
    uint32_t token_count;
};
static_assert(sizeof(push_constants) == 20, "embedding push constants must match GLSL");

bool create_upload_buffer(VkPhysicalDevice physical_device, VkDevice device, VkDeviceSize size,
                          VkBuffer & buffer, VkDeviceMemory & memory) {
    const VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, nullptr};
    if (vkCreateBuffer(device, &info, nullptr, &buffer) != VK_SUCCESS) return false;
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device, buffer, &requirements);
    const uint32_t type = astc_vulkan_find_memory_type(physical_device, requirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (type == std::numeric_limits<uint32_t>::max()) return false;
    const VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr,
        requirements.size, type};
    if (vkAllocateMemory(device, &allocation, nullptr, &memory) != VK_SUCCESS ||
        vkBindBufferMemory(device, buffer, memory, 0) != VK_SUCCESS) return false;
    return true;
}

bool upload(VkDevice device, VkDeviceMemory memory, const void * source, size_t bytes) {
    void * mapped = nullptr;
    if (vkMapMemory(device, memory, 0, bytes, 0, &mapped) != VK_SUCCESS) return false;
    std::memcpy(mapped, source, bytes);
    const VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, nullptr, memory, 0, VK_WHOLE_SIZE};
    const VkResult status = vkFlushMappedMemoryRanges(device, 1, &range);
    vkUnmapMemory(device, memory);
    return status == VK_SUCCESS;
}

void destroy_buffer(VkDevice device, VkBuffer & buffer, VkDeviceMemory & memory) {
    if (buffer != VK_NULL_HANDLE) vkDestroyBuffer(device, buffer, nullptr);
    if (memory != VK_NULL_HANDLE) vkFreeMemory(device, memory, nullptr);
    buffer = VK_NULL_HANDLE;
    memory = VK_NULL_HANDLE;
}

} // namespace

astc_vulkan_embedding_gpu_session::~astc_vulkan_embedding_gpu_session() { reset(); }

void astc_vulkan_embedding_gpu_session::reset() {
    if (device_ != VK_NULL_HANDLE) {
        // This session is only materialized between graph evaluations. Its
        // external dispatch borrows the command buffer and never submits it.
        vkDeviceWaitIdle(device_);
        if (pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(device_, pipeline_, nullptr);
        if (shader_module_ != VK_NULL_HANDLE) vkDestroyShaderModule(device_, shader_module_, nullptr);
        if (pipeline_layout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
        if (descriptor_pool_ != VK_NULL_HANDLE) vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
        if (descriptor_layout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device_, descriptor_layout_, nullptr);
        destroy_buffer(device_, affine_buffer_, affine_memory_);
    }
    texture_.reset();
    physical_device_ = VK_NULL_HANDLE; device_ = VK_NULL_HANDLE; queue_ = VK_NULL_HANDLE;
    queue_family_ = UINT32_MAX; descriptor_set_ = VK_NULL_HANDLE;
    descriptor_layout_ = VK_NULL_HANDLE; descriptor_pool_ = VK_NULL_HANDLE;
    pipeline_layout_ = VK_NULL_HANDLE; shader_module_ = VK_NULL_HANDLE; pipeline_ = VK_NULL_HANDLE;
    vocabulary_ = dimensions_ = tiles_per_token_ = token_columns_ = atlas_width_ = atlas_height_ = 0;
}

bool astc_vulkan_embedding_gpu_session::init(
        VkPhysicalDevice physical_device, VkDevice device, VkQueue queue, uint32_t queue_family,
        const std::vector<uint8_t> & token_major_payload, const std::vector<float> & affine,
        uint32_t vocabulary, uint32_t dimensions, uint32_t tiles_per_token,
        const std::vector<uint32_t> & spirv, std::string & error) {
    reset();
    if (physical_device == VK_NULL_HANDLE || device == VK_NULL_HANDLE || queue == VK_NULL_HANDLE ||
        queue_family == UINT32_MAX || vocabulary == 0 || dimensions == 0 || tiles_per_token == 0 ||
        spirv.empty() || affine.size() != static_cast<size_t>(vocabulary) * 2 ||
        token_major_payload.size() != static_cast<size_t>(vocabulary) * tiles_per_token * 16) {
        error = "invalid E1 GPU embedding decode configuration"; return false;
    }
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physical_device, &properties);
    const uint64_t token_width = static_cast<uint64_t>(tiles_per_token) * 10;
    if (token_width == 0 || token_width > properties.limits.maxImageDimension2D) {
        error = "E1 token microtile width exceeds Vulkan image limit"; return false;
    }
    token_columns_ = std::min<uint32_t>(vocabulary,
        properties.limits.maxImageDimension2D / static_cast<uint32_t>(token_width));
    if (token_columns_ == 0) { error = "Vulkan image limit cannot hold an E1 token row"; return false; }
    atlas_width_ = token_columns_ * static_cast<uint32_t>(token_width);
    atlas_height_ = ((vocabulary + token_columns_ - 1) / token_columns_) * 5;
    if (atlas_height_ > properties.limits.maxImageDimension2D ||
        !astc_vulkan_supports_sampled_transfer_extent(physical_device,
            astc_vulkan_vk_format(static_cast<uint8_t>(astc_vulkan_footprint::k10x5)), atlas_width_, atlas_height_)) {
        error = "Vulkan device cannot sample the E1 10x5 embedding atlas extent"; return false;
    }
    // The cache is token-major; Vulkan block rows are atlas-major. Reorder
    // once at materialization, never on the hot lookup path.
    const uint32_t blocks_per_row = atlas_width_ / 10;
    const uint32_t block_rows = atlas_height_ / 5;
    std::vector<uint8_t> atlas_payload(static_cast<size_t>(blocks_per_row) * block_rows * 16, 0);
    for (uint32_t token = 0; token < vocabulary; ++token) {
        const uint32_t bx = (token % token_columns_) * tiles_per_token;
        const uint32_t by = token / token_columns_;
        for (uint32_t tile = 0; tile < tiles_per_token; ++tile) {
            const size_t destination = (static_cast<size_t>(by) * blocks_per_row + bx + tile) * 16;
            const size_t source = (static_cast<size_t>(token) * tiles_per_token + tile) * 16;
            std::memcpy(atlas_payload.data() + destination, token_major_payload.data() + source, 16);
        }
    }
    // Padding is unreachable but must still contain legal-looking data for a
    // complete compressed image upload. Reuse token zero's first block.
    for (size_t offset = 0; offset < atlas_payload.size(); offset += 16) {
        bool empty = true; for (uint32_t i = 0; i < 16; ++i) empty &= atlas_payload[offset + i] == 0;
        if (empty) std::memcpy(atlas_payload.data() + offset, token_major_payload.data(), 16);
    }
    if (!texture_.upload(physical_device, device, queue, queue_family,
                         static_cast<uint8_t>(astc_vulkan_footprint::k10x5),
                         atlas_width_, atlas_height_, atlas_payload, error)) return false;
    physical_device_ = physical_device; device_ = device; queue_ = queue; queue_family_ = queue_family;
    vocabulary_ = vocabulary; dimensions_ = dimensions; tiles_per_token_ = tiles_per_token;
    const VkDeviceSize affine_bytes = static_cast<VkDeviceSize>(affine.size()) * sizeof(float);
    if (!create_upload_buffer(physical_device_, device_, affine_bytes, affine_buffer_, affine_memory_) ||
        !upload(device_, affine_memory_, affine.data(), static_cast<size_t>(affine_bytes))) {
        error = "cannot create E1 embedding affine buffer"; reset(); return false;
    }
    const VkDescriptorSetLayoutBinding bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    const VkDescriptorSetLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        nullptr, 0, 4, bindings};
    if (vkCreateDescriptorSetLayout(device_, &layout_info, nullptr, &descriptor_layout_) != VK_SUCCESS) {
        error = "cannot create E1 embedding descriptor layout"; reset(); return false;
    }
    const VkDescriptorPoolSize pool_sizes[] = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,3}};
    const VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,nullptr,0,1,2,pool_sizes};
    if (vkCreateDescriptorPool(device_, &pool_info, nullptr, &descriptor_pool_) != VK_SUCCESS) {
        error = "cannot create E1 embedding descriptor pool"; reset(); return false;
    }
    const VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,nullptr,descriptor_pool_,1,&descriptor_layout_};
    if (vkAllocateDescriptorSets(device_, &allocate, &descriptor_set_) != VK_SUCCESS) {
        error = "cannot allocate E1 embedding descriptor set"; reset(); return false;
    }
    const VkDescriptorImageInfo image{texture_.sampler(), texture_.view(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    const VkDescriptorBufferInfo affine_info{affine_buffer_, 0, affine_bytes};
    const VkWriteDescriptorSet writes[] = {
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,nullptr,descriptor_set_,0,0,1,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,&image,nullptr,nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,nullptr,descriptor_set_,3,0,1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,nullptr,&affine_info,nullptr},
    };
    vkUpdateDescriptorSets(device_, 2, writes, 0, nullptr);
    const VkShaderModuleCreateInfo shader{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,nullptr,0,spirv.size()*sizeof(uint32_t),spirv.data()};
    if (vkCreateShaderModule(device_, &shader, nullptr, &shader_module_) != VK_SUCCESS) {
        error = "cannot create E1 embedding shader module"; reset(); return false;
    }
    const VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(push_constants)};
    const VkPipelineLayoutCreateInfo pipeline_layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,nullptr,0,1,&descriptor_layout_,1,&range};
    if (vkCreatePipelineLayout(device_, &pipeline_layout, nullptr, &pipeline_layout_) != VK_SUCCESS) {
        error = "cannot create E1 embedding pipeline layout"; reset(); return false;
    }
    const VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_COMPUTE_BIT,shader_module_,"main",nullptr};
    const VkComputePipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,nullptr,0,stage,pipeline_layout_,VK_NULL_HANDLE,-1};
    if (vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipeline, nullptr, &pipeline_) != VK_SUCCESS) {
        error = "cannot create E1 embedding decode pipeline"; reset(); return false;
    }
    error.clear(); return true;
}

bool astc_vulkan_embedding_gpu_session::record_external(
        VkCommandBuffer command_buffer, VkBuffer token_buffer, VkDeviceSize token_offset, VkDeviceSize token_size,
        VkBuffer output_buffer, VkDeviceSize output_offset, VkDeviceSize output_size,
        uint32_t token_count, std::string & error) {
    if (!ready() || command_buffer == VK_NULL_HANDLE || token_buffer == VK_NULL_HANDLE ||
        output_buffer == VK_NULL_HANDLE || token_count == 0 ||
        token_size < static_cast<VkDeviceSize>(token_count) * sizeof(int32_t) ||
        output_size < static_cast<VkDeviceSize>(token_count) * dimensions_ * sizeof(float)) {
        error = "invalid E1 embedding external dispatch buffers"; return false;
    }
    const VkDescriptorBufferInfo tokens{token_buffer, token_offset, token_size};
    const VkDescriptorBufferInfo output{output_buffer, output_offset, output_size};
    const VkWriteDescriptorSet writes[] = {
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,nullptr,descriptor_set_,1,0,1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,nullptr,&tokens,nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,nullptr,descriptor_set_,2,0,1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,nullptr,&output,nullptr},
    };
    vkUpdateDescriptorSets(device_, 2, writes, 0, nullptr);
    const push_constants constants{dimensions_, vocabulary_, tiles_per_token_, token_columns_, token_count};
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_,0,1,&descriptor_set_,0,nullptr);
    vkCmdPushConstants(command_buffer, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(constants),&constants);
    vkCmdDispatch(command_buffer, (dimensions_ + 63) / 64, token_count, 1);
    error.clear(); return true;
}

bool astc_vulkan_embedding_gpu_session::run(
        const std::vector<int32_t> & token_ids, std::vector<float> & output,
        std::string & error) {
    if (!ready() || token_ids.empty()) { error = "invalid E1 GPU embedding smoke input"; return false; }
    const VkDeviceSize token_bytes = static_cast<VkDeviceSize>(token_ids.size()) * sizeof(int32_t);
    const VkDeviceSize output_bytes = static_cast<VkDeviceSize>(token_ids.size()) * dimensions_ * sizeof(float);
    VkBuffer token_buffer = VK_NULL_HANDLE, output_buffer = VK_NULL_HANDLE;
    VkDeviceMemory token_memory = VK_NULL_HANDLE, output_memory = VK_NULL_HANDLE;
    VkCommandPool pool = VK_NULL_HANDLE; VkCommandBuffer command = VK_NULL_HANDLE; VkFence fence = VK_NULL_HANDLE;
    const auto cleanup = [&]() {
        if (fence != VK_NULL_HANDLE) vkDestroyFence(device_, fence, nullptr);
        if (pool != VK_NULL_HANDLE) vkDestroyCommandPool(device_, pool, nullptr);
        destroy_buffer(device_, token_buffer, token_memory);
        destroy_buffer(device_, output_buffer, output_memory);
    };
    if (!create_upload_buffer(physical_device_, device_, token_bytes, token_buffer, token_memory) ||
        !create_upload_buffer(physical_device_, device_, output_bytes, output_buffer, output_memory) ||
        !upload(device_, token_memory, token_ids.data(), static_cast<size_t>(token_bytes))) {
        cleanup(); error = "cannot allocate E1 GPU smoke buffers"; return false;
    }
    const VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,nullptr,VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,queue_family_};
    if (vkCreateCommandPool(device_, &pool_info, nullptr, &pool) != VK_SUCCESS) { cleanup(); error = "cannot create E1 GPU smoke command pool"; return false; }
    const VkCommandBufferAllocateInfo alloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,nullptr,pool,VK_COMMAND_BUFFER_LEVEL_PRIMARY,1};
    if (vkAllocateCommandBuffers(device_, &alloc, &command) != VK_SUCCESS) { cleanup(); error = "cannot allocate E1 GPU smoke command buffer"; return false; }
    const VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,nullptr,0};
    if (vkCreateFence(device_, &fence_info, nullptr, &fence) != VK_SUCCESS) { cleanup(); error = "cannot create E1 GPU smoke fence"; return false; }
    const VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,nullptr,VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,nullptr};
    if (vkBeginCommandBuffer(command, &begin) != VK_SUCCESS ||
        !record_external(command, token_buffer, 0, token_bytes, output_buffer, 0, output_bytes,
                         static_cast<uint32_t>(token_ids.size()), error)) { cleanup(); return false; }
    const VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,nullptr,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_HOST_READ_BIT,VK_QUEUE_FAMILY_IGNORED,VK_QUEUE_FAMILY_IGNORED,output_buffer,0,output_bytes};
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,0,nullptr,1,&barrier,0,nullptr);
    if (vkEndCommandBuffer(command) != VK_SUCCESS) { cleanup(); error = "cannot end E1 GPU smoke command buffer"; return false; }
    const VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO,nullptr,0,nullptr,0,1,&command,0,nullptr};
    if (vkQueueSubmit(queue_, 1, &submit, fence) != VK_SUCCESS ||
        vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
        cleanup(); error = "E1 GPU smoke submission failed"; return false;
    }
    void * mapped = nullptr;
    if (vkMapMemory(device_, output_memory, 0, output_bytes, 0, &mapped) != VK_SUCCESS) { cleanup(); error = "cannot map E1 GPU smoke output"; return false; }
    output.resize(static_cast<size_t>(token_ids.size()) * dimensions_);
    std::memcpy(output.data(), mapped, static_cast<size_t>(output_bytes));
    vkUnmapMemory(device_, output_memory);
    cleanup(); error.clear(); return true;
}
