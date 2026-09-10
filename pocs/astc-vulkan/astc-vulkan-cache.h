#pragma once

#include "astc-vulkan-manifest.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

enum class astc_vulkan_cache_blob_kind : uint8_t {
    manifest,
    payload,
    layout,
    row_scales,
    pair_map,
    provenance,
    catalog,
};

struct astc_vulkan_cache_blob {
    const uint8_t * data = nullptr;
    uint64_t size = 0;
};

// Immutable cache storage independent of whether the bytes originate in a
// sidecar directory or an embedded compiled-model container. The owner of the
// pointed-to bytes must outlive this source and all users of its validation.
struct astc_vulkan_cache_blob_set {
    astc_vulkan_cache_blob manifest;
    astc_vulkan_cache_blob payload;
    astc_vulkan_cache_blob layout;
    astc_vulkan_cache_blob row_scales;
    astc_vulkan_cache_blob pair_map;
    astc_vulkan_cache_blob provenance;
    astc_vulkan_cache_blob catalog;
};

class astc_vulkan_cache_source {
public:
    virtual ~astc_vulkan_cache_source() = default;
    virtual bool blob(astc_vulkan_cache_blob_kind kind,
                      astc_vulkan_cache_blob & result,
                      std::string & error) const = 0;
    virtual bool read_range(astc_vulkan_cache_blob_kind kind, uint64_t offset,
                            uint64_t size, std::vector<uint8_t> & result,
                            std::string & error) const = 0;
    virtual bool sha256(astc_vulkan_cache_blob_kind kind, std::string & result,
                        std::string & error) const = 0;
};

std::shared_ptr<const astc_vulkan_cache_source> astc_vulkan_make_file_cache_source(
    const struct astc_vulkan_cache_paths & paths);
std::shared_ptr<const astc_vulkan_cache_source> astc_vulkan_make_memory_cache_source(
    const astc_vulkan_cache_blob_set & blobs,
    std::shared_ptr<const void> owner = {});

// Versioned, model-adjacent ASTC artifact cache. The cache intentionally keeps
// GGUF separate: it is an overlay for approved ASTC tensors, never a second
// model format. A valid cache is bound to the exact source GGUF SHA-256.
//
// Layout (for `model.gguf`):
//   model.gguf.astc-vulkan/
//     manifest.astcv, payload.astcpack, [layout-map.bin], [row-scales.bin],
//     [pair-map.bin]
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
    std::string pair_map;
    std::string provenance;
    std::string source_sha256;
    std::string manifest_sha256;
    std::string payload_sha256;
    std::string layout_sha256;
    std::string row_scales_sha256;
    std::string pair_map_sha256;
    // Optional metadata-only compiled catalog. New caches publish both the
    // catalog and its hash; older caches remain valid without either file.
    std::string catalog;
    std::string catalog_sha256;
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
    std::shared_ptr<const astc_vulkan_cache_source> source;
    astc_vulkan_manifest manifest;
    bool has_paired_d2 = false;
    bool has_row_scales = false;
    bool has_pair_map = false;
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

// Validates an immutable cache source against the supplied model path. The
// source may be backed by embedded compiled-model sections and therefore does
// not require a cache directory or sidecar files.
bool astc_vulkan_cache_validate_source(
    const std::string & model_path,
    const std::shared_ptr<const astc_vulkan_cache_source> & source,
    const std::string & expected_source_sha256,
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

// v5 extension for D2 artifacts with an explicit 10-row pair permutation.
// Pairing is offline-selected metadata; it never changes standard ASTC bytes.
bool astc_vulkan_cache_create_with_metadata(const std::string & model_path,
                                            const std::string & manifest_input,
                                            const std::string & payload_input,
                                            const std::string & layout_input,
                                            const std::string & row_scales_input,
                                            const std::string & pair_map_input,
                                            const std::string & provenance_input,
                                            const std::string & requested_cache_path,
                                            astc_vulkan_cache_paths & paths,
                                            std::string & error);
