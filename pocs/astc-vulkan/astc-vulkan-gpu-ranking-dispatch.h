#pragma once

#include "astc-vulkan-gpu-ranking.h"
#include "astc-vulkan-resource.h"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>
#include <vector>

// Vulkan execution for the offline paired-D2 candidate-ranking transport.
//
// Candidate construction and ASTC encoding stay on the CPU. This session is
// deliberately limited to the device-resident part of the experiment:
// fixed-function ASTC decode, paired semantic reconstruction, and activation
// delta construction. The host may select this session or the CPU oracle via
// astc_vulkan_ranking_plan; neither path changes an exported ASTC payload.

struct astc_vulkan_gpu_ranking_push_constants {
    uint32_t candidate_count = 0;
    uint32_t calibration_samples = 0;
    uint32_t tensor_width = 0;
    uint32_t tensor_logical_height = 0;
    uint32_t source_blocks_x = 0;
    uint32_t block_width = 0;
    uint32_t block_height = 0;
    float reconstruction_scale = 1.0f;
};

static_assert(sizeof(astc_vulkan_gpu_ranking_push_constants) == 32,
              "GPU ranking push constants must match astc-paired-candidate-delta.comp");

class astc_vulkan_gpu_ranking_session {
public:
    astc_vulkan_gpu_ranking_session() = default;
    ~astc_vulkan_gpu_ranking_session();
    astc_vulkan_gpu_ranking_session(const astc_vulkan_gpu_ranking_session &) = delete;
    astc_vulkan_gpu_ranking_session & operator=(const astc_vulkan_gpu_ranking_session &) = delete;

    bool init(VkPhysicalDevice physical_device, VkDevice device, VkQueue queue,
              uint32_t queue_family, const astc_vulkan_gpu_ranking_atlas & atlas,
              uint32_t source_blocks_x, uint32_t tensor_width,
              uint32_t tensor_logical_height, uint32_t calibration_samples,
              const std::vector<uint32_t> & spirv, std::string & error);

    bool run(const std::vector<float> & activations, float reconstruction_scale,
             std::vector<float> & deltas, std::string & error);
    void reset();
    bool ready() const { return device_ != VK_NULL_HANDLE && pipeline_ != VK_NULL_HANDLE; }

private:
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queue_family_ = UINT32_MAX;
    astc_vulkan_texture atlas_texture_;
    uint32_t candidate_count_ = 0;
    uint32_t source_blocks_x_ = 0;
    uint32_t tensor_width_ = 0;
    uint32_t tensor_logical_height_ = 0;
    uint32_t calibration_samples_ = 0;
    uint32_t block_width_ = 0;
    uint32_t block_height_ = 0;
    VkBuffer records_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory records_memory_ = VK_NULL_HANDLE;
    VkBuffer baseline_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory baseline_memory_ = VK_NULL_HANDLE;
    VkBuffer activation_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory activation_memory_ = VK_NULL_HANDLE;
    VkBuffer delta_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory delta_memory_ = VK_NULL_HANDLE;
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
