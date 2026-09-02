#pragma once

#include "astc-vulkan-manifest.h"
#include "astc-vulkan-resource.h"

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Stable interface shared by the sidecar dispatch and astc-ffn-matvec.comp.
// Vulkan push constants are scalar-aligned here, so the C++ and GLSL layouts
// are six consecutive 32-bit values (24 bytes).
struct astc_vulkan_matvec_push_constants {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t sample_index = 0;
    float scale_l = 1.0f;
    float scale_a = 0.0f;
    float offset = 0.0f;
};

static_assert(sizeof(astc_vulkan_matvec_push_constants) == 24,
              "ASTC matvec push constants must match the GLSL block");

enum class astc_vulkan_descriptor_binding : uint32_t {
    kWeights = 0,
    kActivations = 1,
    kOutput = 2,
};

inline astc_vulkan_matvec_push_constants astc_vulkan_make_push_constants(
        uint32_t width, uint32_t height, uint32_t sample_index,
        const astc_vulkan_reconstruction & reconstruction) {
    return {width, height, sample_index, reconstruction.scale_l,
            reconstruction.scale_a, reconstruction.offset};
}

// Owns the descriptor/pipeline/buffer lifetime for one sampled ASTC tensor.
// The tensor session must outlive this object because it owns the sampled
// image and sampler referenced by the descriptor set.
class astc_vulkan_matvec_session {
public:
    astc_vulkan_matvec_session() = default;
    ~astc_vulkan_matvec_session();
    astc_vulkan_matvec_session(const astc_vulkan_matvec_session &) = delete;
    astc_vulkan_matvec_session & operator=(const astc_vulkan_matvec_session &) = delete;

    bool init(VkPhysicalDevice physical_device, VkDevice device, VkQueue queue,
              uint32_t queue_family, const astc_vulkan_tensor_session & tensor,
              const std::vector<uint32_t> & spirv, uint32_t width, uint32_t height,
              uint32_t samples, std::string & error);
    bool run(const std::vector<float> & activations,
             const astc_vulkan_reconstruction & reconstruction,
             std::vector<float> & output, std::string & error);
    void reset();
    bool ready() const { return device_ != VK_NULL_HANDLE && pipeline_ != VK_NULL_HANDLE; }

private:
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queue_family_ = UINT32_MAX;
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    uint32_t samples_ = 0;
    VkBuffer activation_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory activation_memory_ = VK_NULL_HANDLE;
    VkBuffer output_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory output_memory_ = VK_NULL_HANDLE;
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
