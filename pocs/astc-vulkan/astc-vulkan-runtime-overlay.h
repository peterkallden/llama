#pragma once

#include "astc-vulkan-model-cache.h"
#include "astc-vulkan-page-owner.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Runtime-facing owner for one immutable ASTC model cache. It centralizes
// artifact policy, cache validation and static residency, but deliberately
// owns neither llama graph objects nor a Vulkan command buffer. That lets the
// same prepared overlay serve a CPU bridge today and a native ggml-vulkan op
// later without changing cache semantics.
struct astc_vulkan_runtime_overlay_options {
    std::string source_model_path;
    std::string runtime_model_path;
    std::string cache_path = "auto";
    astc_vulkan_model_cache_plan_options policy{};
    astc_vulkan_memory_budget memory_budget{};
    uint64_t max_page_payload_bytes = 0;
    uint32_t atlas_slots_x = 1;
};

struct astc_vulkan_runtime_overlay_summary {
    size_t planned_artifacts = 0;
    size_t resident_artifacts = 0;
    size_t native_fallbacks = 0;
    size_t pages = 0;
    size_t resident_pages = 0;
    uint64_t resident_device_bytes = 0;
    uint64_t resident_host_bytes = 0;
    bool preload_all = false;
    bool requires_streaming = false;
};

class astc_vulkan_runtime_overlay {
public:
    bool prepare(const astc_vulkan_runtime_overlay_options & options, std::string & error);

    // Resolves only policy-approved, resident artifacts. A cache miss, an
    // ineligible artifact or an evicted page is deliberately a successful
    // native fallback, not a partially-bound ASTC execution.
    bool resolve_tensor(const std::string & tensor_name,
                        astc_vulkan_page_material & material,
                        std::string & error) const;

    void reset();
    bool ready() const { return ready_; }
    const astc_vulkan_model_cache_catalog & catalog() const { return catalog_; }
    const astc_vulkan_page_owner & page_owner() const { return page_owner_; }
    const astc_vulkan_runtime_overlay_summary & summary() const { return summary_; }

private:
    bool ready_ = false;
    astc_vulkan_model_cache_catalog catalog_{};
    astc_vulkan_page_owner page_owner_{};
    astc_vulkan_runtime_overlay_summary summary_{};
};
