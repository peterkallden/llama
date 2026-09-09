#pragma once

#include <cstdint>
#include <string>
#include <vector>

// A self-contained, hybrid ASTC model container.  The embedded GGUF remains
// byte-identical, so it continues to own architecture, tokenizer and native
// tensor semantics.  ASTC resources are copied into independent sections and
// can therefore be consumed without the original GGUF or sidecar directory.
//
// This is deliberately a POC container and not a new ggml_type.  The runtime
// provider can later expose the sections through the existing native/ASTC
// storage boundary without changing ggml-vulkan.
struct astc_vulkan_compiled_model {
    static constexpr uint32_t kCurrentVersion = 1;

    uint32_t version = kCurrentVersion;
    std::string source_model_fingerprint;
    std::vector<uint8_t> gguf;
    std::vector<uint8_t> manifest;
    std::vector<uint8_t> payload;
    std::vector<uint8_t> layout;
    std::vector<uint8_t> row_scales;
    std::vector<uint8_t> pair_map;
    std::vector<uint8_t> provenance;
    std::vector<uint8_t> catalog;
};

// Build a self-contained container from a source GGUF and a validated cache
// directory (or manifest path).  The source GGUF is copied byte-for-byte;
// cache sections are copied from the published, checksum-validated files.
bool astc_vulkan_compiled_model_pack(
    const std::string & model_path,
    const std::string & requested_cache_path,
    const std::string & output_path,
    std::string & error);

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
