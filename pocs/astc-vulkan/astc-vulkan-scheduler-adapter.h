#pragma once

#include "astc-vulkan-artifact-policy.h"
#include "astc-vulkan-sidecar.h"

#include <cstdint>
#include <string>
#include <vector>

// Cache lookup is deliberately independent of execution. A cache hit is a
// fully verified, immutable offline artifact; it never triggers ASTC encoding
// or any other just-in-time model transformation during inference startup.
enum class astc_vulkan_scheduler_artifact_kind : uint8_t {
    kNone,
    kD1,
    kD2,
};

enum class astc_vulkan_scheduler_dispatch_kind : uint8_t {
    kD1Affine,
    kD2Paired,
};

struct astc_vulkan_scheduler_artifact {
    astc_vulkan_scheduler_artifact_kind kind = astc_vulkan_scheduler_artifact_kind::kNone;
    astc_vulkan_tensor_record record;
    std::string artifact_id;
    astc_vulkan_artifact_variant variant = astc_vulkan_artifact_variant::neutral;
    astc_vulkan_normalization normalization = astc_vulkan_normalization::none;
    astc_vulkan_paired_semantic paired_semantic = astc_vulkan_paired_semantic::direct_rgb;
    astc_vulkan_artifact_evidence evidence{};
    std::string cache_root;
    std::vector<uint8_t> payload;
    std::vector<uint8_t> layout;
    std::vector<float> row_scales;
};

// Experimental scheduler-facing boundary. It deliberately owns no llama or
// ggml objects; a future scheduler can translate its tensor view into this
// contract and retain normal fallback when prepare() reports not-ready.
class astc_vulkan_scheduler_adapter {
public:
    bool prepare(const std::string & manifest_path, const std::string & payload_blob_path,
                 const std::string & tensor_name, astc_vulkan_footprint footprint,
                 std::string & error, bool allow_experimental = false);
    // Resolves `auto` beside model_path or an explicit cache root/manifest,
    // verifies GGUF and artifact SHA-256 records, and binds a pre-approved
    // artifact. Standard D1 is available by default. Evidence-approved D2_6x5
    // or D2_8x5 L+A requires
    // allow_experimental=true plus v4 model/Vulkan evidence. A cache miss is a
    // normal fallback condition, never a partial binding.
    bool prepare_from_cache(const std::string & model_path, const std::string & cache_path,
                            const std::string & tensor_name, astc_vulkan_footprint footprint,
                            std::string & error, bool allow_experimental = false);
    // Reads one already-created cache artifact. This is the scheduler's only
    // artifact discovery API; it intentionally never invokes an encoder. D1
    // and D2 share validation, but D2 additionally owns a packed layout map.
    bool resolve_from_cache(const std::string & model_path, const std::string & cache_path,
                            const std::string & tensor_name, astc_vulkan_footprint footprint,
                            astc_vulkan_scheduler_artifact & artifact, std::string & error) const;
    // v4-only evidence-aware lookup. The policy ranks pre-validated artifacts
    // for a tensor; it never derives a representation at runtime.
    bool resolve_best_from_cache(const std::string & model_path, const std::string & cache_path,
                                 const std::string & tensor_name, astc_vulkan_footprint footprint,
                                 astc_vulkan_quality_policy policy,
                                 astc_vulkan_scheduler_artifact & artifact,
                                 std::string & error) const;
    bool run(const std::vector<uint32_t> & spirv, const std::vector<float> & activations,
             std::vector<float> & output, std::string & error);
    bool run_streamed(uint64_t max_resident_payload_bytes,
                      const std::vector<uint32_t> & spirv,
                      const std::vector<float> & activations,
                      std::vector<float> & output, std::string & error);
    // Read-only views used by the artifact replay oracle. Runtime inference
    // continues to consume the already-bound resources; these accessors do
    // not trigger encoding, cache generation, or additional uploads.
    const std::vector<uint8_t> & payload() const { return payload_; }
    const std::vector<uint8_t> & layout() const { return layout_; }
    astc_vulkan_paired_semantic paired_semantic() const {
        return sidecar_.paired_semantic();
    }
    const std::vector<float> & row_scales() const { return sidecar_.row_scales(); }
    void reset() {
        sidecar_.reset();
        binding_ = {};
        payload_.clear();
        layout_.clear();
        tensor_name_.clear();
        payload_path_.clear();
        payload_offset_ = 0;
    }
    bool ready() const { return sidecar_.ready() && binding_.status == astc_vulkan_binding_status::kReady; }
    const astc_vulkan_ffn_binding & binding() const { return binding_; }
    astc_vulkan_scheduler_dispatch_kind dispatch_kind() const {
        return binding_.record.representation == astc_vulkan_representation::kPairedD2 ?
            astc_vulkan_scheduler_dispatch_kind::kD2Paired :
            astc_vulkan_scheduler_dispatch_kind::kD1Affine;
    }
    static constexpr bool jit_cache_build_enabled() { return false; }

private:
    astc_vulkan_sidecar sidecar_;
    astc_vulkan_ffn_binding binding_;
    std::vector<uint8_t> payload_;
    std::vector<uint8_t> layout_;
    std::string tensor_name_;
    std::string payload_path_;
    uint64_t payload_offset_ = 0;
};
