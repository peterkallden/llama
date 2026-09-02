#include "astc-vulkan-manifest.h"
#include "astc-vulkan-provenance.h"

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace {

std::vector<uint8_t> read_bytes(const std::string & path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return {};
    const std::streamsize size = file.tellg();
    if (size <= 0) return {};
    std::vector<uint8_t> result(static_cast<size_t>(size));
    file.seekg(0);
    file.read(reinterpret_cast<char *>(result.data()), size);
    return file ? result : std::vector<uint8_t>();
}

std::string read_text(const std::string & path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return {};
    const std::streamsize size = file.tellg();
    if (size < 0) return {};
    std::string result(static_cast<size_t>(size), '\0');
    file.seekg(0);
    file.read(result.data(), size);
    return file ? result : std::string();
}

bool read_metadata(const std::string & path, float & scale_l, float & scale_a,
                   float & offset) {
    std::ifstream file(path);
    if (!file) return false;
    bool have_l = false, have_a = false, have_offset = false;
    std::string line;
    while (std::getline(file, line)) {
        const size_t separator = line.find('=');
        if (separator == std::string::npos) continue;
        const std::string key = line.substr(0, separator);
        const std::string value = line.substr(separator + 1);
        try {
            if (key == "scale_l") { scale_l = std::stof(value); have_l = true; }
            else if (key == "scale_a") { scale_a = std::stof(value); have_a = true; }
            else if (key == "offset") { offset = std::stof(value); have_offset = true; }
        } catch (...) { return false; }
    }
    return have_l && have_a && have_offset;
}

bool parse_footprint(const std::string & value, astc_vulkan_footprint & footprint) {
    if (value == "4x4") footprint = astc_vulkan_footprint::k4x4;
    else if (value == "5x5") footprint = astc_vulkan_footprint::k5x5;
    else if (value == "6x6") footprint = astc_vulkan_footprint::k6x6;
    else if (value == "8x6") footprint = astc_vulkan_footprint::k8x6;
    else if (value == "10x6") footprint = astc_vulkan_footprint::k10x6;
    else if (value == "8x8") footprint = astc_vulkan_footprint::k8x8;
    else return false;
    return true;
}

bool parse_representation(const std::string & value, astc_vulkan_representation & representation) {
    if (value == "scalar") representation = astc_vulkan_representation::kScalar;
    else if (value == "gauge") representation = astc_vulkan_representation::kGaugeLumaAlpha;
    else if (value == "c-delta") representation = astc_vulkan_representation::kCDelta;
    else return false;
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    std::string input, metadata, manifest_path, payload_path, tensor_name,
                fingerprint, provenance_path, source_family, calibration_hash,
                validation_hash, holdout_hash, selector_config, validation_prefix,
                commit_order_hash, padding_contract, footprint_name = "6x6",
                representation_name = "scalar";
    uint32_t width = 0, height = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string option = argv[i];
        const std::string value = argv[i + 1];
        if (option == "--input") input = value;
        else if (option == "--metadata") metadata = value;
        else if (option == "--manifest") manifest_path = value;
        else if (option == "--payload") payload_path = value;
        else if (option == "--tensor") tensor_name = value;
        else if (option == "--model-fingerprint") fingerprint = value;
        else if (option == "--provenance") provenance_path = value;
        else if (option == "--source-family") source_family = value;
        else if (option == "--calibration-hash") calibration_hash = value;
        else if (option == "--validation-hash") validation_hash = value;
        else if (option == "--holdout-hash") holdout_hash = value;
        else if (option == "--selector-config") selector_config = value;
        else if (option == "--validation-prefix") validation_prefix = value;
        else if (option == "--commit-order-hash") commit_order_hash = value;
        else if (option == "--padding-contract") padding_contract = value;
        else if (option == "--width") width = static_cast<uint32_t>(std::stoul(value));
        else if (option == "--height") height = static_cast<uint32_t>(std::stoul(value));
        else if (option == "--footprint") footprint_name = value;
        else if (option == "--representation") representation_name = value;
        else return 2;
    }
    astc_vulkan_footprint footprint;
    astc_vulkan_representation representation;
    const std::vector<uint8_t> bytes = read_bytes(input);
    float scale_l = 1.0f, scale_a = 0.0f, offset = 0.0f;
    if (input.empty() || metadata.empty() || manifest_path.empty() || payload_path.empty() ||
        tensor_name.empty() || width == 0 || height == 0 || bytes.empty() ||
        !parse_footprint(footprint_name, footprint) ||
        !parse_representation(representation_name, representation) ||
        !read_metadata(metadata, scale_l, scale_a, offset)) return 2;
    astc_vulkan_manifest manifest;
    manifest.model_fingerprint = fingerprint;
    astc_vulkan_tensor_record record;
    record.name = tensor_name;
    record.width = width;
    record.height = height;
    record.footprint = footprint;
    record.byte_size = bytes.size();
    record.representation = representation;
    record.scale_l = scale_l;
    record.scale_a = scale_a;
    record.offset = offset;
    record.payload_hash64 = astc_vulkan_payload_hash64(bytes.data(), bytes.size());
    manifest.tensors.push_back(record);
    std::string error;
    if (!astc_vulkan_write_manifest(manifest_path, manifest, error)) return 1;
    std::ofstream payload(payload_path, std::ios::binary);
    payload.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    if (!payload.good()) return 1;
    if (!provenance_path.empty()) {
        const std::string manifest_bytes = read_text(manifest_path);
        astc_vulkan_provenance provenance;
        provenance.source_model = fingerprint;
        provenance.source_tensor = tensor_name;
        provenance.source_family = source_family;
        provenance.footprint = footprint_name;
        provenance.representation = representation_name;
        provenance.decoder_contract = "scale_l:" + std::to_string(scale_l) +
                                     ",scale_a:" + std::to_string(scale_a) +
                                     ",offset:" + std::to_string(offset);
        provenance.calibration_hash = calibration_hash;
        provenance.validation_hash = validation_hash;
        provenance.holdout_hash = holdout_hash;
        provenance.selector_config = selector_config;
        provenance.validation_prefix = validation_prefix;
        provenance.commit_order_hash = commit_order_hash;
        provenance.padding_contract = padding_contract;
        provenance.payload_bytes = bytes.size();
        provenance.payload_sha256 = astc_vulkan_sha256_hex(bytes.data(), bytes.size());
        provenance.manifest_sha256 = astc_vulkan_sha256_hex(
            manifest_bytes.data(), manifest_bytes.size());
        if (!astc_vulkan_write_provenance(provenance_path, provenance, error)) {
            std::fprintf(stderr, "cannot write ASTC Vulkan provenance: %s\n", error.c_str());
            return 1;
        }
    }
    return 0;
}
