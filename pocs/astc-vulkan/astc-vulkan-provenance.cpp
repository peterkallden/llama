#include "astc-vulkan-provenance.h"

#include "hash/hash.h"

#include <cctype>
#include <fstream>

namespace {

bool valid_value(const std::string & value) {
    for (const unsigned char character : value) {
        if (character == '\n' || character == '\r' || character == '=') return false;
    }
    return !value.empty();
}

bool write_field(std::ofstream & file, const char * key, const std::string & value) {
    file << key << '=' << value << '\n';
    return file.good();
}

} // namespace

std::string astc_vulkan_sha256_hex(const void * data, size_t size) {
    return hash_sha256_hex(data, size);
}

bool astc_vulkan_validate_provenance(const astc_vulkan_provenance & provenance,
                                     std::string & error) {
    if (provenance.version != 1 || provenance.payload_bytes == 0 ||
        provenance.payload_sha256.size() != 64 ||
        provenance.manifest_sha256.size() != 64 ||
        !valid_value(provenance.source_model) ||
        !valid_value(provenance.source_tensor) ||
        !valid_value(provenance.source_family) ||
        !valid_value(provenance.footprint) ||
        !valid_value(provenance.representation) ||
        !valid_value(provenance.decoder_contract) ||
        !valid_value(provenance.calibration_hash) ||
        !valid_value(provenance.validation_hash) ||
        !valid_value(provenance.holdout_hash) ||
        !valid_value(provenance.selector_config) ||
        !valid_value(provenance.validation_prefix) ||
        !valid_value(provenance.commit_order_hash) ||
        !valid_value(provenance.padding_contract)) {
        error = "invalid ASTC Vulkan provenance record";
        return false;
    }
    error.clear();
    return true;
}

bool astc_vulkan_write_provenance(const std::string & path,
                                  const astc_vulkan_provenance & provenance,
                                  std::string & error) {
    if (!astc_vulkan_validate_provenance(provenance, error)) return false;
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        error = "cannot open ASTC Vulkan provenance for writing";
        return false;
    }
    const bool ok = write_field(file, "version", std::to_string(provenance.version)) &&
                    write_field(file, "source_model", provenance.source_model) &&
                    write_field(file, "source_tensor", provenance.source_tensor) &&
                    write_field(file, "source_family", provenance.source_family) &&
                    write_field(file, "footprint", provenance.footprint) &&
                    write_field(file, "representation", provenance.representation) &&
                    write_field(file, "decoder_contract", provenance.decoder_contract) &&
                    write_field(file, "calibration_hash", provenance.calibration_hash) &&
                    write_field(file, "validation_hash", provenance.validation_hash) &&
                    write_field(file, "holdout_hash", provenance.holdout_hash) &&
                    write_field(file, "selector_config", provenance.selector_config) &&
                    write_field(file, "validation_prefix", provenance.validation_prefix) &&
                    write_field(file, "commit_order_hash", provenance.commit_order_hash) &&
                    write_field(file, "padding_contract", provenance.padding_contract) &&
                    write_field(file, "payload_bytes", std::to_string(provenance.payload_bytes)) &&
                    write_field(file, "payload_sha256", provenance.payload_sha256) &&
                    write_field(file, "manifest_sha256", provenance.manifest_sha256);
    if (!ok) {
        error = "cannot write ASTC Vulkan provenance";
        return false;
    }
    error.clear();
    return true;
}
