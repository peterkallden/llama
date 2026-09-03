#pragma once

#include <cstdint>
#include <string>

// Research-time identity for an exported ASTC artifact.  This is deliberately
// separate from the compact runtime manifest: it records how the bytes were
// produced without adding fields to the Vulkan binding contract.
struct astc_vulkan_provenance {
    uint32_t version = 1;
    std::string source_model;
    std::string source_tensor;
    std::string source_family;
    std::string footprint;
    std::string representation;
    std::string decoder_contract;
    std::string calibration_hash;
    std::string validation_hash;
    std::string holdout_hash;
    std::string selector_config;
    std::string validation_prefix;
    std::string commit_order_hash;
    std::string padding_contract;
    uint64_t payload_bytes = 0;
    std::string payload_sha256;
    std::string manifest_sha256;
};

// SHA-256 is used for artifact identity.  The existing runtime FNV checksum
// remains useful for fast payload validation, but is not a provenance hash.
std::string astc_vulkan_sha256_hex(const void * data, size_t size);

// Streams a file through the same SHA-256 implementation used for artifact
// provenance. This keeps multi-gigabyte GGUF identity checks bounded in
// memory when a cache is opened beside a model.
bool astc_vulkan_sha256_file_hex(const std::string & path, std::string & hash,
                                 std::string & error);

bool astc_vulkan_validate_provenance(const astc_vulkan_provenance & provenance,
                                     std::string & error);
bool astc_vulkan_write_provenance(const std::string & path,
                                  const astc_vulkan_provenance & provenance,
                                  std::string & error);
