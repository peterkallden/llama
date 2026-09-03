#pragma once

#include "astc-vulkan-driver.h"
#include "astc-vulkan-resource.h"

#include <string>
#include <vector>

enum class astc_vulkan_binding_status {
    kReady,
    kFallback,
};

// The production caller owns the normal ggml fallback execution. The sidecar
// still reports a deterministic preference so an unsupported ASTC artifact can
// never leave format selection implicit.
enum class astc_vulkan_fallback_format {
    kQ4_K_M,
    kQ3_K_M,
};

struct astc_vulkan_fallback_policy {
    astc_vulkan_fallback_format preferred = astc_vulkan_fallback_format::kQ4_K_M;
    astc_vulkan_fallback_format low_memory = astc_vulkan_fallback_format::kQ3_K_M;
};

struct astc_vulkan_ffn_binding {
    astc_vulkan_binding_status status = astc_vulkan_binding_status::kFallback;
    astc_vulkan_tensor_record record;
    astc_vulkan_reconstruction reconstruction;
    astc_vulkan_fallback_policy fallback_policy;
    std::string fallback_reason;
};

// PoC-facing adapter for FFN-down tensors. It performs model-shape and device
// capability gating while leaving fallback execution to the normal llama path.
class astc_vulkan_ffn_adapter {
public:
    bool prepare(const astc_vulkan_manifest & manifest, const std::string & tensor_name,
                 uint32_t expected_columns, bool sampled_astc_supported,
                 uint32_t expected_rows,
                 astc_vulkan_ffn_binding & binding, std::string & error) const;
    bool upload(VkPhysicalDevice physical_device, VkDevice device, VkQueue queue,
                uint32_t queue_family, const astc_vulkan_ffn_binding & binding,
                const std::vector<uint8_t> & payload, std::string & error);
    void reset() { session_.reset(); }
    const astc_vulkan_tensor_session & session() const { return session_; }

private:
    astc_vulkan_tensor_session session_;
};
