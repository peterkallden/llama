#pragma once

#include <cstdint>
#include <string>
#include <vector>

// A self-contained ASTC model container. V1 embeds a byte-identical GGUF.
// V2 instead stores the canonical GGUF bootstrap (metadata, tokenizer and
// tensor descriptors) plus native storage and immutable ASTC resources. The
// v2 storage mode makes the fallback contract explicit:
//   hybrid - retain every native tensor and use ASTC opportunistically;
//   strict - omit matrix tensors backed by ASTC and fail closed if the overlay
//            cannot prepare them. It is a deployment assertion, not a quality
//            policy.
//
// This is deliberately a POC container and not a new ggml_type.  The runtime
// provider can later expose the sections through the existing native/ASTC
// storage boundary without changing ggml-vulkan.
enum class astc_vulkan_compiled_storage_mode : uint32_t {
    hybrid = 1,
    strict = 2,
};

struct astc_vulkan_compiled_model {
    static constexpr uint32_t kCurrentVersion = 2;

    // Keep direct struct construction source-compatible with v1 tests and
    // tools. `compiled-pack --bootstrap` explicitly selects v2.
    uint32_t version = 1;
    astc_vulkan_compiled_storage_mode storage_mode = astc_vulkan_compiled_storage_mode::hybrid;
    std::string source_model_fingerprint;
    std::vector<uint8_t> gguf;
    std::vector<uint8_t> manifest;
    std::vector<uint8_t> payload;
    std::vector<uint8_t> layout;
    std::vector<uint8_t> row_scales;
    std::vector<uint8_t> pair_map;
    std::vector<uint8_t> provenance;
    std::vector<uint8_t> catalog;

    // ASTCCM v2-only sections. `gguf` is a bootstrap GGUF ending at its data
    // offset, not a complete data file. Native bytes and their table cover all
    // logical tensors that have no selected ASTC artifact.
    std::vector<uint8_t> native_table;
    std::vector<uint8_t> native_payload;
    std::vector<uint8_t> embedding_metadata;
    std::vector<uint8_t> embedding_payload;
    std::vector<uint8_t> embedding_affine;

    bool is_bootstrap_v2() const { return version >= 2; }
    bool is_strict() const {
        return is_bootstrap_v2() && storage_mode == astc_vulkan_compiled_storage_mode::strict;
    }
};

// Build a self-contained container from a source GGUF and a validated cache
// directory (or manifest path).  The source GGUF is copied byte-for-byte;
// cache sections are copied from the published, checksum-validated files.
bool astc_vulkan_compiled_model_pack(
    const std::string & model_path,
    const std::string & requested_cache_path,
    const std::string & output_path,
    std::string & error);

// Same build operation, but emits a v2 bootstrap container. Hybrid is the
// safe default. Strict omits native copies only for matrix artifacts described
// by the validated manifest; it must fail closed at runtime if ASTC is absent.
bool astc_vulkan_compiled_model_pack_bootstrap(
    const std::string & model_path,
    const std::string & requested_cache_path,
    const std::string & output_path,
    astc_vulkan_compiled_storage_mode storage_mode,
    std::string & error);

// Converts a v2 hybrid container to strict without reading the source GGUF.
// The reverse operation is impossible because strict containers intentionally
// discard the native bytes they omit.
bool astc_vulkan_compiled_model_convert_to_strict(
    const std::string & input_path,
    const std::string & output_path,
    std::string & error);

const char * astc_vulkan_compiled_storage_mode_name(astc_vulkan_compiled_storage_mode mode);

bool astc_vulkan_compiled_model_write(
    const std::string & output_path,
    const astc_vulkan_compiled_model & model,
    std::string & error);

bool astc_vulkan_compiled_model_read(
    const std::string & input_path,
    astc_vulkan_compiled_model & model,
    std::string & error);

// Writes only the embedded GGUF.  This compatibility helper is useful while
// the llama model-loader adapter is being introduced; it never mutates the
// compiled container itself.
bool astc_vulkan_compiled_model_extract_gguf(
    const astc_vulkan_compiled_model & model,
    const std::string & output_path,
    std::string & error);

bool astc_vulkan_compiled_model_validate(
    const astc_vulkan_compiled_model & model,
    std::string & error);
