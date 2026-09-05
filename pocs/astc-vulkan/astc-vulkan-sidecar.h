#pragma once

#include "astc-vulkan-dispatch.h"
#include "astc-vulkan-ffn-adapter.h"
#include "astc-vulkan-budget.h"
#include "astc-vulkan-paired-dispatch.h"

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

    bool init(astc_vulkan_footprint footprint, std::string & error,
              bool allow_experimental = false);
    bool load_manifest(const std::string & path, std::string & error);
    bool set_manifest(const astc_vulkan_manifest & manifest, std::string & error);
    bool bind_tensor(const std::string & tensor_name, uint32_t expected_columns,
                     uint32_t expected_rows, const std::vector<uint8_t> & payload,
                     astc_vulkan_ffn_binding & binding, std::string & error,
                     const std::vector<uint8_t> & paired_layout = {},
                     astc_vulkan_paired_semantic paired_semantic = astc_vulkan_paired_semantic::direct_rgb,
                     const std::vector<float> & row_scales = {});
    bool run(const std::vector<uint32_t> & spirv, const std::vector<float> & activations,
             std::vector<float> & output, std::string & error);
    // Offline/replay-only streamed execution. The payload is read in
    // footprint-aligned block-row ranges and only one band image is resident
    // at a time. The resident run() path remains the production fast path.
    bool run_streamed(const std::string & payload_path, uint64_t payload_offset,
                      uint64_t max_resident_payload_bytes,
                      const std::vector<uint32_t> & spirv,
                      const std::vector<float> & activations,
                      std::vector<float> & output, std::string & error);
    void reset();

    bool ready() const { return device_ != VK_NULL_HANDLE; }
    const astc_vulkan_manifest & manifest() const { return manifest_; }
    const astc_vulkan_memory_budget & memory_budget() const { return memory_budget_; }
    astc_vulkan_paired_semantic paired_semantic() const { return paired_semantic_; }
    const std::vector<float> & row_scales() const { return paired_row_scales_; }

private:
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queue_family_ = UINT32_MAX;
    astc_vulkan_footprint footprint_ = astc_vulkan_footprint::k6x6;
    astc_vulkan_manifest manifest_;
    astc_vulkan_ffn_adapter adapter_;
    astc_vulkan_tensor_session stream_tensor_;
    astc_vulkan_ffn_binding binding_;
    astc_vulkan_matvec_session dispatch_;
    astc_vulkan_tensor_session paired_tensor_;
    astc_vulkan_paired_matvec_session paired_dispatch_;
    std::vector<uint8_t> paired_layout_;
    std::vector<float> paired_row_scales_;
    astc_vulkan_paired_semantic paired_semantic_ = astc_vulkan_paired_semantic::direct_rgb;
    std::vector<uint32_t> dispatch_spirv_;
    uint32_t dispatch_samples_ = 0;
    astc_vulkan_memory_budget memory_budget_{};
};
