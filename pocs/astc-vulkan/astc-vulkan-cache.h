#pragma once

#include "astc-vulkan-manifest.h"

#include <string>

// Versioned, model-adjacent ASTC artifact cache. The cache intentionally keeps
// GGUF separate: it is an overlay for approved ASTC tensors, never a second
// model format. A valid cache is bound to the exact source GGUF SHA-256.
//
// Layout (for `model.gguf`):
//   model.gguf.astc-vulkan/
//     manifest.astcv, payload.astcpack, [layout-map.bin], [provenance.txt]
//     source.gguf.sha256, manifest.sha256, payload.sha256, [layout-map.sha256]
//
// `layout-map.bin` is optional for D1 and required for a cache containing any
// paired-D2 record. Cache support does not imply scheduler support: the normal
// D1-only production adapter still rejects paired-D2 artifacts.

struct astc_vulkan_cache_paths {
    std::string root;
    std::string manifest;
    std::string payload;
    std::string layout;
    std::string provenance;
    std::string source_sha256;
    std::string manifest_sha256;
    std::string payload_sha256;
    std::string layout_sha256;
};

struct astc_vulkan_cache_validation {
    astc_vulkan_cache_paths paths;
    astc_vulkan_manifest manifest;
    bool has_paired_d2 = false;
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
