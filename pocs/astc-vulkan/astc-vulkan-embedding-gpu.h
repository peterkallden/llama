#pragma once

#include "astc-vulkan-resource.h"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>
#include <vector>

// Native Vulkan decode/session for the E1 token-local 10x5 annex. This is
// intentionally representation-specific: generic D1/D2 matvec code must not
// learn embedding lookup geometry. It borrows ggml Vulkan buffers only while
// recording an external operation.
class astc_vulkan_embedding_gpu_session {
public:
    ~astc_vulkan_embedding_gpu_session();
    astc_vulkan_embedding_gpu_session() = default;
    astc_vulkan_embedding_gpu_session(const astc_vulkan_embedding_gpu_session &) = delete;
    astc_vulkan_embedding_gpu_session & operator=(const astc_vulkan_embedding_gpu_session &) = delete;

    bool init(VkPhysicalDevice physical_device, VkDevice device, VkQueue queue,
              uint32_t queue_family, const std::vector<uint8_t> & token_major_payload,
              const std::vector<float> & affine, uint32_t vocabulary, uint32_t dimensions,
              uint32_t tiles_per_token, const std::vector<uint32_t> & spirv,
              std::string & error);
    bool record_external(VkCommandBuffer command_buffer, VkBuffer token_buffer,
                         VkDeviceSize token_offset, VkDeviceSize token_size,
                         VkBuffer output_buffer, VkDeviceSize output_offset,
                         VkDeviceSize output_size, uint32_t token_count,
                         std::string & error);
    bool run(const std::vector<int32_t> & token_ids, std::vector<float> & output,
             std::string & error);
    void reset();
    bool ready() const { return device_ != VK_NULL_HANDLE && pipeline_ != VK_NULL_HANDLE; }
    VkDevice device() const { return device_; }
    uint32_t token_columns() const { return token_columns_; }
    uint32_t atlas_width() const { return atlas_width_; }
    uint32_t atlas_height() const { return atlas_height_; }

private:
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queue_family_ = UINT32_MAX;
    astc_vulkan_texture texture_;
    VkBuffer affine_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory affine_memory_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkShaderModule shader_module_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    uint32_t vocabulary_ = 0;
    uint32_t dimensions_ = 0;
    uint32_t tiles_per_token_ = 0;
    uint32_t token_columns_ = 0;
    uint32_t atlas_width_ = 0;
    uint32_t atlas_height_ = 0;
};
