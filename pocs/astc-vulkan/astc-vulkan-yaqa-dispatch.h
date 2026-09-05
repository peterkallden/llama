#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>
#include <vector>

// Resident two-stage YAQA trace scorer for offline candidate ranking.
//
// The input/output traces are uploaded once per tensor and remain resident
// while candidate error matrices are replaced batch by batch. Only one final
// score per candidate is read back. This class deliberately owns no ASTC
// semantics: callers provide decoded candidate error matrices in
// [candidate][row][column] order.
class astc_vulkan_yaqa_session {
public:
    astc_vulkan_yaqa_session() = default;
    ~astc_vulkan_yaqa_session();
    astc_vulkan_yaqa_session(const astc_vulkan_yaqa_session &) = delete;
    astc_vulkan_yaqa_session & operator=(const astc_vulkan_yaqa_session &) = delete;

    bool init(VkPhysicalDevice physical_device, VkDevice device, VkQueue queue,
              uint32_t queue_family, uint32_t rows, uint32_t columns,
              uint32_t samples, const std::vector<uint32_t> & partial_spirv,
              const std::vector<uint32_t> & reduce_spirv,
              uint32_t candidate_capacity, std::string & error);

    // Trace layout is [sample][column] and [sample][row], respectively.
    // It is intentionally separate from candidate errors so it can be reused
    // across many offline candidate batches.
    bool upload_traces(const std::vector<float> & input_trace,
                       const std::vector<float> & output_trace,
                       std::string & error);

    // Error layout is [candidate][row][column]. `candidate_count` may be
    // smaller than the capacity supplied to init().
    bool run(const std::vector<float> & errors, uint32_t candidate_count,
             std::vector<float> & scores, std::string & error);

    void reset();
    bool ready() const { return device_ != VK_NULL_HANDLE && partial_pipeline_ != VK_NULL_HANDLE; }
    bool traces_resident() const { return traces_resident_; }
    uint32_t rows() const { return rows_; }
    uint32_t columns() const { return columns_; }
    uint32_t samples() const { return samples_; }
    uint32_t candidate_capacity() const { return candidate_capacity_; }

private:
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queue_family_ = UINT32_MAX;
    uint32_t rows_ = 0;
    uint32_t columns_ = 0;
    uint32_t samples_ = 0;
    uint32_t pairs_per_candidate_ = 0;
    uint32_t candidate_capacity_ = 0;
    bool traces_resident_ = false;

    VkBuffer errors_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory errors_memory_ = VK_NULL_HANDLE;
    VkBuffer input_trace_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory input_trace_memory_ = VK_NULL_HANDLE;
    VkBuffer output_trace_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory output_trace_memory_ = VK_NULL_HANDLE;
    VkBuffer partials_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory partials_memory_ = VK_NULL_HANDLE;
    VkBuffer scores_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory scores_memory_ = VK_NULL_HANDLE;

    VkDescriptorSetLayout descriptor_layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout reduce_descriptor_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;
    VkDescriptorSet reduce_descriptor_set_ = VK_NULL_HANDLE;
    VkPipelineLayout partial_pipeline_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout reduce_pipeline_layout_ = VK_NULL_HANDLE;
    VkShaderModule partial_shader_module_ = VK_NULL_HANDLE;
    VkShaderModule reduce_shader_module_ = VK_NULL_HANDLE;
    VkPipeline partial_pipeline_ = VK_NULL_HANDLE;
    VkPipeline reduce_pipeline_ = VK_NULL_HANDLE;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
};
