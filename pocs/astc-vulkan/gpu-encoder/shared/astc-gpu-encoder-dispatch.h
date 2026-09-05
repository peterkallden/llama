#pragma once

// Vulkan execution for the isolated GPU ASTC proposer experiment.
//
// This is deliberately an offline, physical-only session. It accepts D1/D2
// frontend source texels and emits symbolic proposals; it never creates an
// ASTC image, never packs BISE, and never participates in inference runtime.

#include "astc-gpu-encoder.h"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>
#include <vector>

class astc_gpu_encoder_session {
public:
    astc_gpu_encoder_session() = default;
    ~astc_gpu_encoder_session();
    astc_gpu_encoder_session(const astc_gpu_encoder_session &) = delete;
    astc_gpu_encoder_session & operator=(const astc_gpu_encoder_session &) = delete;

    // `footprint` fixes the reusable texel-stride and is part of the shader
    // ABI. Every footprint must have its own explicitly verified kernel.
    bool init(VkPhysicalDevice physical_device, VkDevice device, VkQueue queue,
              uint32_t queue_family, astc_vulkan_footprint footprint,
              uint32_t max_blocks,
              const std::vector<uint32_t> & spirv, std::string & error);
    bool run(const astc_gpu_encoder_request & request,
             std::vector<astc_gpu_encoder_proposal> & proposals,
             std::string & error);
    void reset();
    bool ready() const { return device_ != VK_NULL_HANDLE && pipeline_ != VK_NULL_HANDLE; }
    uint32_t capacity() const { return capacity_; }

private:
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queue_family_ = UINT32_MAX;
    astc_vulkan_footprint footprint_ = astc_vulkan_footprint::k4x4;
    uint32_t texels_per_block_ = 0;
    uint32_t capacity_ = 0;
    VkBuffer source_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory source_memory_ = VK_NULL_HANDLE;
    VkBuffer source_id_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory source_id_memory_ = VK_NULL_HANDLE;
    VkBuffer proposal_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory proposal_memory_ = VK_NULL_HANDLE;
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

// Convenience smoke/CLI backend. Production callers should create one
// session and reuse it across batches; this helper only owns the temporary
// instance/device required for standalone verification.
bool astc_gpu_encoder_propose_gpu_default(
    const std::string & spirv_path, const astc_gpu_encoder_request & request,
    std::vector<astc_gpu_encoder_proposal> & proposals, std::string & error);

// Explicit multi-footprint entry point used by separately compiled, gated D1
// kernels. The legacy/default helper remains the D1 scalar 4x4 convenience
// path so callers cannot accidentally select a different geometry.
bool astc_gpu_encoder_propose_gpu_for_footprint_default(
    const std::string & spirv_path, astc_vulkan_footprint footprint,
    const astc_gpu_encoder_request & request,
    std::vector<astc_gpu_encoder_proposal> & proposals, std::string & error);

struct astc_gpu_encoder_benchmark_result {
    uint32_t block_count = 0;
    uint32_t warmup_iterations = 0;
    uint32_t measured_iterations = 0;
    // End-to-end reusable-session time: host upload, queue work, fence, and
    // proposal readback. It intentionally excludes source construction and
    // CPU astcenc finishing, which are measured separately later.
    double mean_roundtrip_ms = 0.0;
    double blocks_per_second = 0.0;
};

bool astc_gpu_encoder_benchmark_gpu_default(
    const std::string & spirv_path, astc_vulkan_footprint footprint,
    const astc_gpu_encoder_request & request, uint32_t warmup_iterations,
    uint32_t measured_iterations, astc_gpu_encoder_benchmark_result & result,
    std::string & error);
