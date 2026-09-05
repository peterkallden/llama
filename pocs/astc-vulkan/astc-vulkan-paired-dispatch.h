#pragma once

#include "astc-vulkan-resource.h"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>
#include <vector>

// Runtime dispatch for the exported paired-D2 artifact contract. It is kept
// separate from the D1 affine matvec because D2 has a physical half-height
// texture and a mandatory storage-buffer layout map. Both paths intentionally
// share astc_vulkan_tensor_session for ASTC image ownership and validation.
struct astc_vulkan_paired_matvec_push_constants {
    uint32_t width = 0;
    uint32_t logical_height = 0;
    uint32_t sample_index = 0;
    uint32_t block_width = 0;
    uint32_t block_height = 0;
    uint32_t layout_map_words = 0;
    uint32_t paired_semantic = 0;
    uint32_t row_scale_count = 0;
    uint32_t dispatch_height = 0;
    uint32_t row_base = 0;
    uint32_t texture_row_base = 0;
    float scale = 1.0f;
    float offset = 0.0f;
};

static_assert(sizeof(astc_vulkan_paired_matvec_push_constants) == 52,
              "paired-D2 push constants must match the GLSL block");

class astc_vulkan_paired_matvec_session {
public:
    astc_vulkan_paired_matvec_session() = default;
    ~astc_vulkan_paired_matvec_session();
    astc_vulkan_paired_matvec_session(const astc_vulkan_paired_matvec_session &) = delete;
    astc_vulkan_paired_matvec_session & operator=(const astc_vulkan_paired_matvec_session &) = delete;

    bool init(VkPhysicalDevice physical_device, VkDevice device, VkQueue queue,
              uint32_t queue_family, const astc_vulkan_tensor_session & tensor,
              const std::vector<uint8_t> & layout_map, const std::vector<uint32_t> & spirv,
              uint32_t width, uint32_t logical_height, uint32_t samples, std::string & error,
              astc_vulkan_paired_semantic semantic = astc_vulkan_paired_semantic::direct_rgb,
              const std::vector<float> & row_scales = {},
              uint32_t storage_height = 0, uint32_t texture_row_base = 0);
    bool run(const std::vector<float> & activations,
             const astc_vulkan_reconstruction & reconstruction,
             std::vector<float> & output, std::string & error);
    bool rebind_texture(const astc_vulkan_tensor_session & tensor,
                        uint32_t storage_height, uint32_t texture_row_base,
                        std::string & error);
    bool run_band(const std::vector<float> & activations,
                  const astc_vulkan_reconstruction & reconstruction,
                  uint32_t row_base, uint32_t band_height,
                  std::vector<float> & output, std::string & error);
    void reset();
    bool ready() const { return device_ != VK_NULL_HANDLE && pipeline_ != VK_NULL_HANDLE; }

private:
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queue_family_ = UINT32_MAX;
    uint32_t width_ = 0;
    uint32_t logical_height_ = 0;
    uint32_t block_width_ = 0;
    uint32_t block_height_ = 0;
    uint32_t storage_height_ = 0;
    uint32_t texture_row_base_ = 0;
    uint32_t layout_map_words_ = 0;
    uint32_t paired_semantic_ = 0;
    uint32_t row_scale_count_ = 0;
    uint32_t samples_ = 0;
    VkBuffer activation_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory activation_memory_ = VK_NULL_HANDLE;
    VkBuffer output_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory output_memory_ = VK_NULL_HANDLE;
    VkBuffer layout_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory layout_memory_ = VK_NULL_HANDLE;
    VkBuffer row_scale_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory row_scale_memory_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkShaderModule shader_module_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
};
