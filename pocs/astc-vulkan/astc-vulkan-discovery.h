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

// Cheap, calibration-only quality signals produced before any ASTC payload
// exists. A trace is tensor-specific: a missing or dimension-incompatible
// trace is represented by available=false, never by a fabricated score.
struct astc_vulkan_discovery_quality_probe {
    std::string tensor_name;
    bool available = false;
    std::string calibration_trace_hash;
    double row_absmax_spread = 0.0; // robust P90 / P10 row-range ratio
    double d1_proxy_error = 0.0;
    double d2_proxy_error = 0.0;
};

enum class astc_vulkan_discovery_candidate_kind : uint8_t {
    d1_10x8 = 0,
    d2_la_pairing_neutral = 1,
    d2_la_pairing_selected = 2,
    d2_la_pairing_absmax_neutral = 3,
    d2_la_pairing_absmax_selected = 4,
};

const char * astc_vulkan_discovery_candidate_name(
    astc_vulkan_discovery_candidate_kind kind);

// One row in the pre-build candidate plan. This is intentionally not an
// artifact: it has no payload, no selected prefix and no model gate yet.
struct astc_vulkan_discovery_candidate_plan_entry {
    std::string tensor_name;
    astc_vulkan_discovery_candidate_kind kind =
        astc_vulkan_discovery_candidate_kind::d2_la_pairing_neutral;
    bool conditional = false;
    bool quality_probe_available = false;
    double row_absmax_spread = 0.0;
    double d1_proxy_error = 0.0;
    double d2_proxy_error = 0.0;
};

struct astc_vulkan_discovery_candidate_plan_options {
    // Absmax is only proposed when it has a plausible structural reason;
    // exact ASTC + holdout replay still decide whether it survives.
    double absmax_spread_threshold = 2.0;
    bool include_d1_iso_rate_control = true;
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

// Derives a bounded candidate bank from static admission plus optional
// calibration-only probes. D1/D2 entries are deliberately co-listed at the
// same nominal rate; native remains the eventual fallback outside this plan.
bool astc_vulkan_make_discovery_candidate_plan(
        const std::vector<astc_vulkan_discovery_entry> & entries,
        const std::vector<astc_vulkan_discovery_quality_probe> & probes,
        const astc_vulkan_discovery_options & discovery_options,
        const astc_vulkan_discovery_candidate_plan_options & options,
        std::vector<astc_vulkan_discovery_candidate_plan_entry> & result,
        std::string & error);

bool astc_vulkan_write_discovery_candidate_plan(
        const std::string & path,
        const std::vector<astc_vulkan_discovery_candidate_plan_entry> & entries,
        std::string & error);
