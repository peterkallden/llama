#pragma once

#include "astc-vulkan-format.h"
#include "astc-vulkan-manifest.h"
#include "astc-vulkan-model-cache.h"

#include <cstdint>
#include <string>
#include <vector>

// GGUF/source inventory data needed before any ASTC artifact exists. This is
// deliberately smaller than a manifest entry: discovery never claims that a
// payload or a quality gate exists.
struct astc_vulkan_discovery_tensor_input {
    std::string tensor_name;
    uint32_t columns = 0;
    uint32_t rows = 0;
    uint64_t source_bytes = 0;
    bool rank2 = false;
};

struct astc_vulkan_discovery_options {
    astc_vulkan_footprint footprint = astc_vulkan_footprint::k6x6;
    astc_vulkan_representation representation = astc_vulkan_representation::kScalar;
    uint64_t min_source_bytes = 0;
    uint64_t max_cache_bytes = 0; // zero means no budget cap
    size_t max_tensors = 0;       // zero means no count cap
    bool require_usage = true;
};

struct astc_vulkan_discovery_entry {
    std::string tensor_name;
    uint32_t columns = 0;
    uint32_t rows = 0;
    uint64_t source_bytes = 0;
    uint64_t estimated_astc_bytes = 0;
    uint64_t estimated_layout_bytes = 0;
    uint64_t estimated_bytes_saved = 0;
    uint64_t invocations = 0;
    uint64_t tokens_seen = 0;
    uint32_t execution_order = 0;
    double path_probability = 0.0;
    double heat_score = 0.0;
    double priority_score = 0.0;
    bool usage_available = false;
    bool selected = false;
    // A discovery result is only an admission/ordering hint. The selected
    // tensor still needs ASTC generation and model/Vulkan replay gates.
    bool quality_probe_required = true;
};

bool astc_vulkan_discover_cache_candidates(
        const std::vector<astc_vulkan_discovery_tensor_input> & tensors,
        const std::vector<astc_vulkan_tensor_usage_metrics> & usage,
        const astc_vulkan_discovery_options & options,
        std::vector<astc_vulkan_discovery_entry> & result,
        std::string & error);

bool astc_vulkan_write_discovery_report(
        const std::string & path,
        const astc_vulkan_discovery_options & options,
        const std::vector<astc_vulkan_discovery_entry> & entries,
        std::string & error);
