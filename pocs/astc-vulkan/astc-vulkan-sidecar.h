#pragma once

#include "astc-vulkan-dispatch.h"
#include "astc-vulkan-ffn-adapter.h"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>
#include <vector>

// Opt-in sidecar owner for one ASTC format. It deliberately lives beside
// llama's Vulkan backend and has no scheduler or ggml-vulkan dependency.
class astc_vulkan_sidecar {
public:
    astc_vulkan_sidecar() = default;
    ~astc_vulkan_sidecar();
    astc_vulkan_sidecar(const astc_vulkan_sidecar &) = delete;
    astc_vulkan_sidecar & operator=(const astc_vulkan_sidecar &) = delete;

    bool init(astc_vulkan_footprint footprint, std::string & error);
    bool load_manifest(const std::string & path, std::string & error);
    bool set_manifest(const astc_vulkan_manifest & manifest, std::string & error);
    bool bind_tensor(const std::string & tensor_name, uint32_t expected_columns,
                     uint32_t expected_rows, const std::vector<uint8_t> & payload,
                     astc_vulkan_ffn_binding & binding, std::string & error);
    bool run(const std::vector<uint32_t> & spirv, const std::vector<float> & activations,
             std::vector<float> & output, std::string & error);
    void reset();

    bool ready() const { return device_ != VK_NULL_HANDLE; }
    const astc_vulkan_manifest & manifest() const { return manifest_; }

private:
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queue_family_ = UINT32_MAX;
    astc_vulkan_footprint footprint_ = astc_vulkan_footprint::k6x6;
    astc_vulkan_manifest manifest_;
    astc_vulkan_ffn_adapter adapter_;
    astc_vulkan_ffn_binding binding_;
    astc_vulkan_matvec_session dispatch_;
    std::vector<uint32_t> dispatch_spirv_;
    uint32_t dispatch_samples_ = 0;
};
