#pragma once

#include "astc-vulkan-manifest.h"

#include <string>

// Versioned, model-adjacent ASTC artifact cache. The cache intentionally keeps
// GGUF separate: it is an overlay for approved ASTC tensors, never a second
// model format. A valid cache is bound to the exact source GGUF SHA-256.
//
// Layout (for `model.gguf`):
//   model.gguf.astc-vulkan/
//     manifest.astcv, payload.astcpack, [layout-map.bin], [row-scales.bin]
//     [provenance.txt], source.gguf.sha256, manifest.sha256, payload.sha256,
//     [layout-map.sha256], [row-scales.sha256]
//     compatible-bases/<runtime-gguf-sha256>.astcbase
//
// `layout-map.bin` is optional for D1 and required for a cache containing any
// paired-D2 record. Cache support does not imply scheduler support: the normal
// D1-only production adapter still rejects paired-D2 artifacts.

struct astc_vulkan_cache_paths {
    std::string root;
    std::string manifest;
    std::string payload;
    std::string layout;
    std::string row_scales;
    std::string provenance;
    std::string source_sha256;
    std::string manifest_sha256;
    std::string payload_sha256;
    std::string layout_sha256;
    std::string row_scales_sha256;
    std::string compatible_bases;
};

// A cache is generated from one exact source GGUF, but can be structurally
// admitted for an exact Q3/Q4/etc. runtime GGUF of the same logical model.
// Admission is deliberately separate from model-quality evidence: an admitted
// base may be used for explicit research replay, but production scheduling
// still requires a runtime-specific model gate.
struct astc_vulkan_cache_runtime_base {
    bool is_source = false;
    bool admitted = false;
    bool model_gate_passed = false;
    bool vulkan_gate_passed = false;
    std::string source_sha256;
    std::string runtime_sha256;
    std::string schema_sha256;
    std::string family;
};

struct astc_vulkan_cache_validation {
    astc_vulkan_cache_paths paths;
    astc_vulkan_manifest manifest;
    bool has_paired_d2 = false;
    bool has_row_scales = false;
    astc_vulkan_cache_runtime_base runtime_base;
};

// Empty or "auto" chooses the model-adjacent cache directory. An explicit
// cache path may name either its directory or manifest.astcv inside it.
bool astc_vulkan_cache_resolve(const std::string & model_path,
                               const std::string & requested_cache_path,
                               astc_vulkan_cache_paths & paths,
                               std::string & error);

// Validates source GGUF identity, all cache SHA-256 records, the runtime
// manifest and every per-tensor payload/layout checksum. It intentionally
// fails closed: callers should use their native Q4/Q3 fallback on failure.
bool astc_vulkan_cache_validate(const std::string & model_path,
                                const std::string & requested_cache_path,
                                astc_vulkan_cache_validation & result,
                                std::string & error);

// Registers an exact runtime GGUF as a structurally compatible base for an
// existing source-derived cache. Both GGUF files are parsed once and must have
// the same architecture/tensor-schema signature. This never modifies the ASTC
// payload and intentionally publishes no model/Vulkan quality gate.
bool astc_vulkan_cache_admit_runtime_base(const std::string & source_model_path,
                                          const std::string & runtime_model_path,
                                          const std::string & requested_cache_path,
                                          const std::string & family,
                                          astc_vulkan_cache_runtime_base & binding,
                                          std::string & error);

// Verifies an existing admission record for a runtime GGUF. This is the
// explicit research/replay path; it does not grant production eligibility.
bool astc_vulkan_cache_validate_runtime_base(const std::string & source_model_path,
                                             const std::string & runtime_model_path,
                                             const std::string & requested_cache_path,
                                             astc_vulkan_cache_runtime_base & binding,
                                             std::string & error);

// Builds a new cache by copying already-exported artifacts into a temporary
// sibling directory and atomically renaming it into place after validation.
// It never overwrites an existing cache. D2 is supported when `layout_input`
// contains the required packed layout map.
bool astc_vulkan_cache_create(const std::string & model_path,
                              const std::string & manifest_input,
                              const std::string & payload_input,
                              const std::string & layout_input,
                              const std::string & provenance_input,
                              const std::string & requested_cache_path,
                              astc_vulkan_cache_paths & paths,
                              std::string & error);

// v4 extension for artifacts whose runtime contract contains a scale blob.
// The existing overload remains the D1/direct-D2 compatibility path.
bool astc_vulkan_cache_create_with_row_scales(const std::string & model_path,
                                              const std::string & manifest_input,
                                              const std::string & payload_input,
                                              const std::string & layout_input,
                                              const std::string & row_scales_input,
                                              const std::string & provenance_input,
                                              const std::string & requested_cache_path,
                                              astc_vulkan_cache_paths & paths,
                                              std::string & error);
