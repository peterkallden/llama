#pragma once

#include "astc-vulkan-sidecar.h"

#include <cstdint>
#include <string>
#include <vector>

// Experimental scheduler-facing boundary. It deliberately owns no llama or
// ggml objects; a future scheduler can translate its tensor view into this
// contract and retain normal fallback when prepare() reports not-ready.
class astc_vulkan_scheduler_adapter {
public:
    bool prepare(const std::string & manifest_path, const std::string & payload_blob_path,
                 const std::string & tensor_name, astc_vulkan_footprint footprint,
                 std::string & error);
    bool run(const std::vector<uint32_t> & spirv, const std::vector<float> & activations,
             std::vector<float> & output, std::string & error);
    void reset() { sidecar_.reset(); payload_.clear(); tensor_name_.clear(); }
    bool ready() const { return sidecar_.ready() && binding_.status == astc_vulkan_binding_status::kReady; }
    const astc_vulkan_ffn_binding & binding() const { return binding_; }

private:
    astc_vulkan_sidecar sidecar_;
    astc_vulkan_ffn_binding binding_;
    std::vector<uint8_t> payload_;
    std::string tensor_name_;
};
