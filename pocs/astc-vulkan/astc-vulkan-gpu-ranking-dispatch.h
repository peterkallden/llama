#pragma once

#include "astc-vulkan-gpu-ranking.h"
#include "astc-vulkan-gpu-d1-ranking.h"
#include "astc-vulkan-resource.h"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>
#include <vector>

// Vulkan execution for the offline paired-D2 candidate-ranking transport.
//
// Candidate construction and ASTC encoding stay on the CPU. This session is
// deliberately limited to the device-resident part of the experiment:
// fixed-function ASTC decode, paired semantic reconstruction, activation
// delta construction, and proposal-gain reduction. The host may select this
// session or the CPU oracle via astc_vulkan_ranking_plan; neither path changes
// an exported ASTC payload. Candidate construction stays CPU-side, while a
// persistent session batches the device stages and only needs to read one gain
// per candidate for the normal proposal path.

struct astc_vulkan_gpu_ranking_push_constants {
    uint32_t candidate_count = 0;
    uint32_t calibration_samples = 0;
    uint32_t tensor_width = 0;
    uint32_t tensor_logical_height = 0;
    uint32_t source_blocks_x = 0;
    uint32_t block_width = 0;
    uint32_t block_height = 0;
    float reconstruction_scale = 1.0f;
    uint32_t logical_rows_per_physical = 2;
};

static_assert(sizeof(astc_vulkan_gpu_ranking_push_constants) == 36,
              "GPU ranking push constants must match candidate-delta/gain shaders");

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
              const std::vector<uint32_t> & delta_spirv,
              const std::vector<uint32_t> & proposal_gain_spirv,
              const std::vector<uint32_t> & local_select_spirv,
              std::string & error);
    bool init(VkPhysicalDevice physical_device, VkDevice device, VkQueue queue,
              uint32_t queue_family, const astc_vulkan_gpu_ranking_atlas & atlas,
              uint32_t source_blocks_x, uint32_t tensor_width,
              uint32_t tensor_logical_height, uint32_t calibration_samples,
              const std::vector<uint32_t> & delta_spirv,
              const std::vector<uint32_t> & proposal_gain_spirv,
              std::string & error) {
        return init(physical_device, device, queue, queue_family, atlas,
                    source_blocks_x, tensor_width, tensor_logical_height,
                    calibration_samples, delta_spirv, proposal_gain_spirv,
                    {}, error);
    }

    // D1 scalar semantic adapter. The physical atlas/session resources are
    // shared with D2; only the delta shader and logical row geometry differ.
    bool init_d1(VkPhysicalDevice physical_device, VkDevice device, VkQueue queue,
                 uint32_t queue_family, const astc_vulkan_gpu_d1_ranking_atlas & atlas,
                 uint32_t source_blocks_x, uint32_t tensor_width,
                 uint32_t tensor_logical_height, uint32_t calibration_samples,
                 const std::vector<uint32_t> & delta_spirv,
                 const std::vector<uint32_t> & proposal_gain_spirv,
                 const std::vector<uint32_t> & local_select_spirv,
                 std::string & error);
    bool init_d1(VkPhysicalDevice physical_device, VkDevice device, VkQueue queue,
                 uint32_t queue_family, const astc_vulkan_gpu_d1_ranking_atlas & atlas,
                 uint32_t source_blocks_x, uint32_t tensor_width,
                 uint32_t tensor_logical_height, uint32_t calibration_samples,
                 const std::vector<uint32_t> & delta_spirv,
                 const std::vector<uint32_t> & proposal_gain_spirv,
                 std::string & error) {
        return init_d1(physical_device, device, queue, queue_family, atlas,
                       source_blocks_x, tensor_width, tensor_logical_height,
                       calibration_samples, delta_spirv, proposal_gain_spirv,
                       {}, error);
    }

    bool run(const std::vector<float> & activations, float reconstruction_scale,
             std::vector<float> & deltas, std::string & error);
    // Computes 2<R,d>-||d||^2 for every candidate entirely on the device after
    // the ASTC delta pass. `residuals` is [calibration sample][logical output
    // row]. This is the compact batch result consumed by CPU-side commit order.
    bool run_proposal_gains(const std::vector<float> & activations,
                            const std::vector<float> & residuals,
                            float reconstruction_scale,
                            std::vector<float> & gains, std::string & error);
    // GPU-local argmax over candidates belonging to each source block. This
    // is deliberately separate from conflict-aware residual commit. The
    // current first step uploads the compact gains, performs the argmax on
    // device, and reads only one candidate id per source block.
    bool run_local_selection(const std::vector<float> & gains,
                             std::vector<uint32_t> & selected_records,
                             std::string & error);
    // Fused device path: delta -> proposal gain -> local argmax, with only
    // one selected record id per source block returning to the host.
    bool run_proposal_gains_and_local_selection(
            const std::vector<float> & activations,
            const std::vector<float> & residuals,
            float reconstruction_scale,
            std::vector<uint32_t> & selected_records,
            std::string & error);
    // Upload trace/residual inputs once and reuse them across candidate-batch
    // updates. This avoids repeating the same host-to-device copies for every
    // D2 batch belonging to one tensor.
    bool upload_inputs(const std::vector<float> & activations,
                       const std::vector<float> & residuals,
                       std::string & error);
    // Reuses all Vulkan objects for a new fixed-size atlas batch. The caller
    // must keep atlas dimensions/footprint unchanged and stay within the
    // initial candidate capacity; record indices are rebuilt per batch.
    bool update_batch(const astc_vulkan_gpu_ranking_atlas & atlas, std::string & error);
    void reset();
    bool ready() const { return device_ != VK_NULL_HANDLE && pipeline_ != VK_NULL_HANDLE; }

private:
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queue_family_ = UINT32_MAX;
    astc_vulkan_texture atlas_texture_;
    uint32_t candidate_count_ = 0;
    uint32_t candidate_capacity_ = 0;
    uint32_t source_block_count_ = 0;
    uint32_t source_block_capacity_ = 0;
    astc_vulkan_footprint footprint_ = astc_vulkan_footprint::k8x5;
    uint32_t source_blocks_x_ = 0;
    uint32_t tensor_width_ = 0;
    uint32_t tensor_logical_height_ = 0;
    uint32_t calibration_samples_ = 0;
    uint32_t block_width_ = 0;
    uint32_t block_height_ = 0;
    uint32_t logical_rows_per_physical_ = 2;
    bool d1_scalar_ = false;
    bool activation_resident_ = false;
    bool residual_resident_ = false;
    VkBuffer records_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory records_memory_ = VK_NULL_HANDLE;
    VkBuffer baseline_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory baseline_memory_ = VK_NULL_HANDLE;
    VkBuffer activation_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory activation_memory_ = VK_NULL_HANDLE;
    VkBuffer delta_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory delta_memory_ = VK_NULL_HANDLE;
    VkBuffer residual_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory residual_memory_ = VK_NULL_HANDLE;
    VkBuffer gain_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory gain_memory_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkShaderModule shader_module_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkShaderModule gain_shader_module_ = VK_NULL_HANDLE;
    VkPipeline gain_pipeline_ = VK_NULL_HANDLE;
    VkBuffer selected_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory selected_memory_ = VK_NULL_HANDLE;
    VkShaderModule select_shader_module_ = VK_NULL_HANDLE;
    VkPipeline select_pipeline_ = VK_NULL_HANDLE;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
};
