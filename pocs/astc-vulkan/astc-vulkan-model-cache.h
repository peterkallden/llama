#pragma once

#include "astc-vulkan-artifact-policy.h"
#include "astc-vulkan-cache.h"
#include "astc-vulkan-manifest.h"
#include "astc-vulkan-residency.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Model-level cache planning is deliberately a thin orchestration layer. It
// does not encode tensors, own Vulkan images, or duplicate manifest parsing.
// The existing per-tensor cache producer remains the artifact generator;
// this module decides which already-produced artifact is eligible and in what
// order a runtime owner should request it.

struct astc_vulkan_model_cache_plan_options {
    astc_vulkan_quality_policy policy = astc_vulkan_quality_policy::balanced;
    bool allow_experimental = false;
    bool allow_unverified = false;
};

struct astc_vulkan_model_cache_entry {
    std::string tensor_name;
    std::string artifact_id;
    astc_vulkan_tensor_record storage;
    astc_vulkan_artifact_evidence evidence{};
    astc_vulkan_artifact_variant variant = astc_vulkan_artifact_variant::neutral;
    astc_vulkan_normalization normalization = astc_vulkan_normalization::none;
    bool use_native_fallback = false;
    // Conservative payload-based estimates. A Vulkan owner may replace these
    // with actual image/staging allocation sizes before residency planning.
    uint64_t device_bytes = 0;
    uint64_t host_bytes = 0;
};

struct astc_vulkan_model_cache_plan {
    std::vector<astc_vulkan_model_cache_entry> entries;
    astc_vulkan_residency_plan residency;
};

struct astc_vulkan_model_cache_catalog {
    astc_vulkan_cache_validation validation;
    astc_vulkan_cache_runtime_base runtime_base;
    astc_vulkan_model_cache_plan plan;
};

// Loads and validates one immutable model-adjacent cache, then builds the
// tensor-level artifact plan. Compatible Q3/Q4 bases remain an explicit
// admission/replay path in astc-vulkan-cache; this function intentionally
// handles the exact source model only.
bool astc_vulkan_model_cache_load_catalog(
    const std::string & model_path,
    const std::string & requested_cache_path,
    const astc_vulkan_model_cache_plan_options & options,
    astc_vulkan_model_cache_catalog & result,
    std::string & error);

// Same catalog load for an admitted logical-model-equivalent runtime GGUF.
// The compatible base must already have an admission record. Production use
// additionally requires runtime-specific model/Vulkan gates; passing
// allow_unverified is reserved for explicit replay/research.
bool astc_vulkan_model_cache_load_catalog_for_runtime(
    const std::string & source_model_path,
    const std::string & runtime_model_path,
    const std::string & requested_cache_path,
    const astc_vulkan_model_cache_plan_options & options,
    astc_vulkan_model_cache_catalog & result,
    std::string & error);

// Selects at most one validated artifact per tensor, preserving manifest
// order. Tensors without an eligible artifact remain explicit native-Q
// fallbacks; they are not silently dropped from the model plan.
bool astc_vulkan_model_cache_make_plan(
    const astc_vulkan_manifest & manifest,
    const astc_vulkan_model_cache_plan_options & options,
    astc_vulkan_model_cache_plan & result,
    std::string & error);

// Applies the existing ordered residency planner to the model-level entries.
// The model scheduler owns ordering and eviction; this function only supplies
// the same cumulative budget calculation already used by tensor streaming.
bool astc_vulkan_model_cache_plan_residency(
    const astc_vulkan_model_cache_plan & model_plan,
    const astc_vulkan_memory_budget & budget,
    astc_vulkan_model_cache_plan & result,
    std::string & error);

struct astc_vulkan_model_cache_build_state {
    std::string source_model;
    std::string output_root;
    std::vector<std::string> completed_tensors;
};

// Small resumable-build journal. It is intentionally separate from the
// binary manifest: a partial build is not runtime-visible until final cache
// publication succeeds.
bool astc_vulkan_model_cache_write_build_state(
    const std::string & path,
    const astc_vulkan_model_cache_build_state & state,
    std::string & error);

bool astc_vulkan_model_cache_read_build_state(
    const std::string & path,
    astc_vulkan_model_cache_build_state & state,
    std::string & error);

// One already-generated per-tensor artifact package. The package is normally
// produced by the existing D1/D2 cache tools; this layer only merges verified
// packages and never runs an encoder.
struct astc_vulkan_model_cache_fragment {
    std::string manifest_path;
    std::string payload_path;
    std::string layout_path;
    std::string row_scales_path;
};

// Merges v4 artifact fragments into one staging manifest and the three
// optional binary blobs. Payload/layout/row-scale ranges are copied in
// fragment order and their manifest offsets are rewritten accordingly.
// `result` is validated against the produced staging files before returning.
bool astc_vulkan_model_cache_merge_fragments(
    const std::vector<astc_vulkan_model_cache_fragment> & fragments,
    const std::string & output_manifest_path,
    const std::string & output_payload_path,
    const std::string & output_layout_path,
    const std::string & output_row_scales_path,
    astc_vulkan_manifest & result,
    std::string & error);

// Convenience wrapper for the normal offline flow: merge fragments into a
// staging directory, then delegate final validation and atomic publication to
// the existing cache publisher. The staging directory remains caller-owned so
// a build journal can resume or inspect it after a failure.
bool astc_vulkan_model_cache_publish_fragments(
    const std::string & source_model_path,
    const std::vector<astc_vulkan_model_cache_fragment> & fragments,
    const std::string & staging_root,
    const std::string & requested_cache_path,
    astc_vulkan_cache_paths & paths,
    std::string & error);
