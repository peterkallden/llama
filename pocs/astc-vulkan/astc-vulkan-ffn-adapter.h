#pragma once

#include "astc-vulkan-driver.h"
#include "astc-vulkan-resource.h"

#include <string>
#include <vector>

enum class astc_vulkan_binding_status {
    kReady,
    kFallback,
};

struct astc_vulkan_ffn_binding {
    astc_vulkan_binding_status status = astc_vulkan_binding_status::kFallback;
    astc_vulkan_tensor_record record;
    astc_vulkan_reconstruction reconstruction;
    std::string fallback_reason;
};

// PoC-facing adapter for FFN-down tensors. It performs model-shape and device
// capability gating while leaving fallback execution to the normal llama path.
class astc_vulkan_ffn_adapter {
public:
    bool prepare(const astc_vulkan_manifest & manifest, const std::string & tensor_name,
                 uint32_t expected_columns, bool sampled_astc_supported,
                 astc_vulkan_ffn_binding & binding, std::string & error) const;
    bool upload(VkPhysicalDevice physical_device, VkDevice device, VkQueue queue,
                uint32_t queue_family, const astc_vulkan_ffn_binding & binding,
                const std::vector<uint8_t> & payload, std::string & error);
    void reset() { session_.reset(); }
    const astc_vulkan_tensor_session & session() const { return session_; }

private:
    astc_vulkan_tensor_session session_;
};
