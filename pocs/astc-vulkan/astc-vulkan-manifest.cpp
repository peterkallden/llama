#include "astc-vulkan-manifest.h"
#include "astc-vulkan-hash.h"
#include "astc-vulkan-paired-layout.h"
#include "astc-vulkan-d2-pair-transform.h"

#include "astc-vulkan-format.h"

#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <unordered_set>

namespace {

constexpr std::array<char, 8> kMagic = {'K', 'A', 'S', 'T', 'C', 'V', 'M', '1'};
constexpr uint32_t kLegacyManifestVersion = 1;
constexpr uint32_t kAffineManifestVersion = 2;
constexpr uint32_t kPairedLayoutManifestVersion = 3;
constexpr uint32_t kArtifactManifestVersion = 4;
constexpr uint32_t kPairMapManifestVersion = 5;
constexpr uint32_t kRobustEvidenceManifestVersion = 6;
constexpr uint32_t kCurrentManifestVersion = kRobustEvidenceManifestVersion;
constexpr uint32_t kMaxStringBytes = 1u << 20;
constexpr uint32_t kMaxTensorRecords = 1u << 20;

template<typename T>
bool write_scalar(std::ofstream & file, T value) {
    file.write(reinterpret_cast<const char *>(&value), sizeof(value));
    return file.good();
}

template<typename T>
bool read_scalar(std::ifstream & file, T & value) {
    file.read(reinterpret_cast<char *>(&value), sizeof(value));
    return file.good();
}

bool write_string(std::ofstream & file, const std::string & value) {
    if (value.size() > kMaxStringBytes) return false;
    const uint32_t size = static_cast<uint32_t>(value.size());
    return write_scalar(file, size) &&
           (size == 0 || (file.write(value.data(), size), file.good()));
}

bool read_string(std::ifstream & file, std::string & value) {
    uint32_t size = 0;
    if (!read_scalar(file, size) || size > kMaxStringBytes) return false;
    value.resize(size);
    return size == 0 || (file.read(value.data(), size), file.good());
}

bool valid_representation(astc_vulkan_representation representation) {
    return representation == astc_vulkan_representation::kScalar ||
           representation == astc_vulkan_representation::kGaugeLumaAlpha ||
           representation == astc_vulkan_representation::kCDelta ||
           representation == astc_vulkan_representation::kPairedD2;
}

bool valid_artifact_variant(astc_vulkan_artifact_variant variant) {
    return variant == astc_vulkan_artifact_variant::neutral ||
           variant == astc_vulkan_artifact_variant::validation_selected;
}

bool valid_normalization(astc_vulkan_normalization normalization) {
    return normalization == astc_vulkan_normalization::none ||
           normalization == astc_vulkan_normalization::per_row_absmax;
}

bool valid_paired_semantic(astc_vulkan_paired_semantic semantic) {
    return semantic == astc_vulkan_paired_semantic::direct_rgb ||
           semantic == astc_vulkan_paired_semantic::luminance_alpha;
}

bool is_paired_d2(const astc_vulkan_tensor_record & tensor) {
    return tensor.representation == astc_vulkan_representation::kPairedD2;
}

bool paired_d2_footprint_is_supported(astc_vulkan_footprint footprint) {
    // All formats have a five-row physical texture stripe and map to a
    // ten-row logical D2 stripe. They nevertheless require separate Vulkan
    // image formats, so there is no generic H10 payload class.
    return footprint == astc_vulkan_footprint::k6x5 ||
           footprint == astc_vulkan_footprint::k8x5 ||
           footprint == astc_vulkan_footprint::k10x5;
}

uint32_t storage_height(const astc_vulkan_tensor_record & tensor) {
    return is_paired_d2(tensor) ? astc_vulkan_paired_storage_height(tensor.height) : tensor.height;
}

uint64_t expected_payload_bytes(const astc_vulkan_tensor_record & tensor) {
    return astc_vulkan_image_bytes(tensor.footprint, tensor.width, storage_height(tensor));
}

uint64_t expected_layout_bytes(const astc_vulkan_tensor_record & tensor) {
    return is_paired_d2(tensor) ? astc_vulkan_paired_layout_bytes(
        tensor.footprint, tensor.width, tensor.height) : 0;
}

bool validate_tensor_record(const astc_vulkan_tensor_record & tensor, uint32_t version,
                            std::string & error) {
    if (tensor.name.empty() || tensor.name.size() > kMaxStringBytes ||
        tensor.width == 0 || tensor.height == 0 ||
        astc_vulkan_format(tensor.footprint).block_width == 0 ||
        !valid_representation(tensor.representation) ||
        !std::isfinite(tensor.scale_l) || !std::isfinite(tensor.scale_a) ||
        !std::isfinite(tensor.offset)) {
        error = "invalid ASTC Vulkan tensor record";
        return false;
    }
    if (is_paired_d2(tensor) &&
        (version < kPairedLayoutManifestVersion || !paired_d2_footprint_is_supported(tensor.footprint) ||
         tensor.layout_byte_size != expected_layout_bytes(tensor) ||
         tensor.layout_byte_size == 0 ||
         tensor.layout_byte_size > std::numeric_limits<uint64_t>::max() - tensor.layout_byte_offset)) {
        error = "invalid ASTC Vulkan paired-D2 layout metadata";
        return false;
    }
    if (!is_paired_d2(tensor) &&
        (tensor.layout_byte_offset != 0 || tensor.layout_byte_size != 0 || tensor.layout_hash64 != 0)) {
        error = "non-paired ASTC Vulkan tensor has layout metadata";
        return false;
    }
    const uint64_t expected_size = expected_payload_bytes(tensor);
    if (expected_size == 0 || tensor.byte_size != expected_size ||
        tensor.byte_size > std::numeric_limits<uint64_t>::max() - tensor.byte_offset) {
        error = "invalid ASTC Vulkan tensor byte range";
        return false;
    }
    return true;
}

bool validate_artifact_record(const astc_vulkan_artifact_record & artifact,
                              std::string & error) {
    if (artifact.id.empty() || artifact.id.size() > kMaxStringBytes ||
        artifact.encoder_profile.size() > kMaxStringBytes ||
        artifact.evidence.calibration_validation_hash.size() > kMaxStringBytes ||
        artifact.evidence.replay_corpus_hash.size() > kMaxStringBytes ||
        !valid_artifact_variant(artifact.variant) ||
        !valid_normalization(artifact.normalization) ||
        !valid_paired_semantic(artifact.paired_semantic) ||
        !std::isfinite(artifact.evidence.activation_mse) ||
        !std::isfinite(artifact.evidence.logits_relative_mse) ||
        !std::isfinite(artifact.evidence.loss_delta) ||
        !std::isfinite(artifact.evidence.top1_agreement) ||
        !validate_tensor_record(artifact.storage, kCurrentManifestVersion, error)) {
        if (error.empty()) error = "invalid ASTC Vulkan artifact record";
        return false;
    }
    const auto & evidence = artifact.evidence;
    if (evidence.replay_case_count > 1 &&
        (!std::isfinite(evidence.median_loss_delta) ||
         !std::isfinite(evidence.worst_loss_delta) ||
         !std::isfinite(evidence.worst_top1_agreement) ||
         evidence.worst_top1_agreement < 0.0f || evidence.worst_top1_agreement > 1.0f)) {
        error = "invalid ASTC Vulkan robust replay evidence";
        return false;
    }
    if (artifact.paired_semantic == astc_vulkan_paired_semantic::luminance_alpha &&
        !is_paired_d2(artifact.storage)) {
        error = "luminance-alpha semantic requires paired-D2 storage";
        return false;
    }
    const uint64_t expected_scale_bytes = artifact.normalization == astc_vulkan_normalization::per_row_absmax ?
        static_cast<uint64_t>(artifact.storage.height) * sizeof(float) : 0;
    if (artifact.row_scale_byte_size != expected_scale_bytes ||
        (expected_scale_bytes == 0 &&
         (artifact.row_scale_byte_offset != 0 || artifact.row_scale_hash64 != 0)) ||
        (expected_scale_bytes != 0 &&
         artifact.row_scale_byte_size > std::numeric_limits<uint64_t>::max() - artifact.row_scale_byte_offset)) {
        error = "invalid ASTC Vulkan artifact row-scale metadata";
        return false;
    }
    const uint64_t expected_pair_map_bytes =
        artifact.storage.representation == astc_vulkan_representation::kPairedD2 &&
        artifact.pair_map_byte_size != 0
            ? static_cast<uint64_t>((artifact.storage.height + astc_vulkan_d2_pair_group_rows - 1) /
                                    astc_vulkan_d2_pair_group_rows) *
                  astc_vulkan_d2_pair_group_rows
            : 0;
    if (artifact.pair_map_byte_size != expected_pair_map_bytes ||
        (expected_pair_map_bytes == 0 &&
         (artifact.pair_map_byte_offset != 0 || artifact.pair_map_hash64 != 0)) ||
        (expected_pair_map_bytes != 0 &&
         artifact.pair_map_byte_size > std::numeric_limits<uint64_t>::max() - artifact.pair_map_byte_offset)) {
        error = "invalid ASTC Vulkan artifact pair-map metadata";
        return false;
    }
    error.clear();
    return true;
}

bool write_tensor_record(std::ofstream & file, const astc_vulkan_tensor_record & tensor,
                         uint32_t version) {
    if (!write_string(file, tensor.name) || !write_scalar(file, tensor.width) ||
        !write_scalar(file, tensor.height) ||
        !write_scalar(file, static_cast<uint8_t>(tensor.footprint)) ||
        !write_scalar(file, tensor.byte_offset) || !write_scalar(file, tensor.byte_size)) {
        return false;
    }
    if (version >= kAffineManifestVersion &&
        (!write_scalar(file, static_cast<uint8_t>(tensor.representation)) ||
         !write_scalar(file, tensor.scale_l) || !write_scalar(file, tensor.scale_a) ||
         !write_scalar(file, tensor.offset) || !write_scalar(file, tensor.payload_hash64))) {
        return false;
    }
    return version < kPairedLayoutManifestVersion ||
           (write_scalar(file, tensor.layout_byte_offset) &&
            write_scalar(file, tensor.layout_byte_size) &&
            write_scalar(file, tensor.layout_hash64));
}

bool read_tensor_record(std::ifstream & file, astc_vulkan_tensor_record & tensor,
                        uint32_t version) {
    uint8_t footprint = 0;
    uint8_t representation = 0;
    if (!read_string(file, tensor.name) || !read_scalar(file, tensor.width) ||
        !read_scalar(file, tensor.height) || !read_scalar(file, footprint) ||
        !read_scalar(file, tensor.byte_offset) || !read_scalar(file, tensor.byte_size)) {
        return false;
    }
    tensor.footprint = static_cast<astc_vulkan_footprint>(footprint);
    if (version >= kAffineManifestVersion &&
        (!read_scalar(file, representation) || !read_scalar(file, tensor.scale_l) ||
         !read_scalar(file, tensor.scale_a) || !read_scalar(file, tensor.offset) ||
         !read_scalar(file, tensor.payload_hash64))) {
        return false;
    }
    if (version >= kAffineManifestVersion) {
        tensor.representation = static_cast<astc_vulkan_representation>(representation);
    }
    return version < kPairedLayoutManifestVersion ||
           (read_scalar(file, tensor.layout_byte_offset) &&
            read_scalar(file, tensor.layout_byte_size) &&
            read_scalar(file, tensor.layout_hash64));
}

bool write_artifact_record(std::ofstream & file, const astc_vulkan_artifact_record & artifact,
                           uint32_t version) {
    const auto & evidence = artifact.evidence;
    return write_string(file, artifact.id) &&
           write_tensor_record(file, artifact.storage, kCurrentManifestVersion) &&
           write_scalar(file, static_cast<uint8_t>(artifact.variant)) &&
           write_scalar(file, static_cast<uint8_t>(artifact.normalization)) &&
           write_scalar(file, static_cast<uint8_t>(artifact.paired_semantic)) &&
           write_string(file, artifact.encoder_profile) &&
           write_scalar(file, static_cast<uint8_t>(evidence.model_gate_passed)) &&
           write_scalar(file, static_cast<uint8_t>(evidence.vulkan_gate_passed)) &&
           write_scalar(file, evidence.activation_mse) &&
           write_scalar(file, evidence.logits_relative_mse) &&
           write_scalar(file, evidence.loss_delta) &&
           write_scalar(file, evidence.top1_agreement) &&
           write_string(file, evidence.calibration_validation_hash) &&
           write_string(file, evidence.replay_corpus_hash) &&
           (version < kRobustEvidenceManifestVersion ||
            (write_scalar(file, evidence.replay_case_count) &&
             write_scalar(file, evidence.median_loss_delta) &&
             write_scalar(file, evidence.worst_loss_delta) &&
             write_scalar(file, evidence.worst_top1_agreement))) &&
           write_scalar(file, artifact.row_scale_byte_offset) &&
           write_scalar(file, artifact.row_scale_byte_size) &&
           write_scalar(file, artifact.row_scale_hash64) &&
           (version < kPairMapManifestVersion ||
            (write_scalar(file, artifact.pair_map_byte_offset) &&
             write_scalar(file, artifact.pair_map_byte_size) &&
             write_scalar(file, artifact.pair_map_hash64)));
}

bool read_artifact_record(std::ifstream & file, astc_vulkan_artifact_record & artifact,
                          uint32_t version) {
    uint8_t variant = 0;
    uint8_t normalization = 0;
    uint8_t semantic = 0;
    uint8_t model_gate = 0;
    uint8_t vulkan_gate = 0;
    auto & evidence = artifact.evidence;
    if (!read_string(file, artifact.id) ||
        !read_tensor_record(file, artifact.storage, kCurrentManifestVersion) ||
        !read_scalar(file, variant) || !read_scalar(file, normalization) ||
        !read_scalar(file, semantic) || !read_string(file, artifact.encoder_profile) ||
        !read_scalar(file, model_gate) || !read_scalar(file, vulkan_gate) ||
        !read_scalar(file, evidence.activation_mse) ||
        !read_scalar(file, evidence.logits_relative_mse) ||
        !read_scalar(file, evidence.loss_delta) || !read_scalar(file, evidence.top1_agreement) ||
        !read_string(file, evidence.calibration_validation_hash) ||
        !read_string(file, evidence.replay_corpus_hash) ||
        (version >= kRobustEvidenceManifestVersion &&
         (!read_scalar(file, evidence.replay_case_count) ||
          !read_scalar(file, evidence.median_loss_delta) ||
          !read_scalar(file, evidence.worst_loss_delta) ||
          !read_scalar(file, evidence.worst_top1_agreement))) ||
        !read_scalar(file, artifact.row_scale_byte_offset) ||
        !read_scalar(file, artifact.row_scale_byte_size) ||
        !read_scalar(file, artifact.row_scale_hash64) ||
        (version >= kPairMapManifestVersion &&
         (!read_scalar(file, artifact.pair_map_byte_offset) ||
          !read_scalar(file, artifact.pair_map_byte_size) ||
          !read_scalar(file, artifact.pair_map_hash64)))) {
        return false;
    }
    artifact.variant = static_cast<astc_vulkan_artifact_variant>(variant);
    artifact.normalization = static_cast<astc_vulkan_normalization>(normalization);
    artifact.paired_semantic = static_cast<astc_vulkan_paired_semantic>(semantic);
    evidence.model_gate_passed = model_gate != 0;
    evidence.vulkan_gate_passed = vulkan_gate != 0;
    return true;
}

template<typename Callback>
bool for_each_storage_record(const astc_vulkan_manifest & manifest, Callback callback) {
    if (manifest.version >= kArtifactManifestVersion) {
        for (const auto & artifact : manifest.artifacts) {
            if (!callback(artifact.storage)) return false;
        }
        return true;
    }
    for (const auto & tensor : manifest.tensors) {
        if (!callback(tensor)) return false;
    }
    return true;
}

} // namespace

uint64_t astc_vulkan_payload_hash64(const uint8_t * data, size_t size) {
    return astc_vulkan_fnv1a64(data, size);
}

bool astc_vulkan_validate_payload(const astc_vulkan_tensor_record & tensor,
                                  const uint8_t * data, size_t size,
                                  std::string & error) {
    const uint64_t expected = expected_payload_bytes(tensor);
    if (expected == 0 || tensor.byte_size != expected || size != tensor.byte_size) {
        error = "ASTC Vulkan tensor payload size does not match its record";
        return false;
    }
    if (tensor.payload_hash64 != 0 &&
        astc_vulkan_payload_hash64(data, size) != tensor.payload_hash64) {
        error = "ASTC Vulkan tensor payload checksum mismatch";
        return false;
    }
    error.clear();
    return true;
}

bool astc_vulkan_validate_layout_map(const astc_vulkan_tensor_record & tensor,
                                     const uint8_t * data, size_t size,
                                     std::string & error) {
    const uint64_t expected = expected_layout_bytes(tensor);
    if (expected == 0 || tensor.layout_byte_size != expected || size != expected || data == nullptr) {
        std::ostringstream detail;
        detail << "ASTC Vulkan paired layout map size does not match its record"
               << " (expected=" << expected << ", record=" << tensor.layout_byte_size
               << ", actual=" << size << ")";
        error = detail.str();
        return false;
    }
    if (tensor.layout_hash64 != 0 && astc_vulkan_payload_hash64(data, size) != tensor.layout_hash64) {
        error = "ASTC Vulkan paired layout map checksum mismatch";
        return false;
    }
    error.clear();
    return true;
}

bool astc_vulkan_validate_pair_map(const astc_vulkan_artifact_record & artifact,
                                   const uint8_t * data, size_t size,
                                   std::string & error) {
    if (artifact.pair_map_byte_size == 0) {
        if (size != 0) {
            error = "ASTC Vulkan artifact unexpectedly has pair-map bytes";
            return false;
        }
        error.clear();
        return true;
    }
    const uint64_t groups = (artifact.storage.height + astc_vulkan_d2_pair_group_rows - 1) /
                            astc_vulkan_d2_pair_group_rows;
    const uint64_t expected = groups * astc_vulkan_d2_pair_group_rows;
    if (artifact.storage.representation != astc_vulkan_representation::kPairedD2 ||
        artifact.pair_map_byte_size != expected || size != expected || data == nullptr) {
        error = "ASTC Vulkan pair-map size does not match its artifact";
        return false;
    }
    if (artifact.pair_map_hash64 != 0 &&
        astc_vulkan_payload_hash64(data, size) != artifact.pair_map_hash64) {
        error = "ASTC Vulkan pair-map checksum mismatch";
        return false;
    }
    for (uint64_t group = 0; group < groups; ++group) {
        astc_vulkan_d2_pairing pairing{};
        for (uint32_t index = 0; index < astc_vulkan_d2_pair_group_rows; ++index) {
            pairing.row_order[index] = data[group * astc_vulkan_d2_pair_group_rows + index];
        }
        if (!astc_vulkan_d2_pairing_is_valid(pairing)) {
            error = "ASTC Vulkan pair-map contains an invalid row permutation";
            return false;
        }
    }
    error.clear();
    return true;
}

bool astc_vulkan_validate_payload_blob(const astc_vulkan_manifest & manifest,
                                       uint64_t blob_size, std::string & error) {
    if (!astc_vulkan_validate_manifest(manifest, error)) return false;
    const bool ok = for_each_storage_record(manifest, [&](const astc_vulkan_tensor_record & tensor) {
        if (tensor.byte_offset > blob_size ||
            tensor.byte_size > blob_size - tensor.byte_offset) {
            error = "ASTC Vulkan tensor range exceeds payload blob";
            return false;
        }
        return true;
    });
    if (!ok) return false;
    error.clear();
    return true;
}

bool astc_vulkan_validate_layout_blob(const astc_vulkan_manifest & manifest,
                                      uint64_t blob_size, std::string & error) {
    if (!astc_vulkan_validate_manifest(manifest, error)) return false;
    const bool ok = for_each_storage_record(manifest, [&](const astc_vulkan_tensor_record & tensor) {
        if (!is_paired_d2(tensor)) return true;
        if (tensor.layout_byte_offset > blob_size ||
            tensor.layout_byte_size > blob_size - tensor.layout_byte_offset) {
            error = "ASTC Vulkan paired layout range exceeds layout blob";
            return false;
        }
        return true;
    });
    if (!ok) return false;
    error.clear();
    return true;
}

bool astc_vulkan_validate_manifest(const astc_vulkan_manifest & manifest,
                                   std::string & error) {
    if (manifest.version != kLegacyManifestVersion &&
        manifest.version != kAffineManifestVersion &&
        manifest.version != kPairedLayoutManifestVersion &&
        manifest.version != kArtifactManifestVersion &&
        manifest.version != kPairMapManifestVersion &&
        manifest.version != kCurrentManifestVersion) {
        error = "unsupported ASTC Vulkan manifest version";
        return false;
    }
    if (manifest.tensors.size() > kMaxTensorRecords || manifest.artifacts.size() > kMaxTensorRecords) {
        error = "too many ASTC Vulkan tensor records";
        return false;
    }
    if (manifest.version >= kArtifactManifestVersion && manifest.artifacts.empty()) {
        error = "ASTC Vulkan artifact manifest has no artifact records";
        return false;
    }
    if (manifest.version < kArtifactManifestVersion && !manifest.artifacts.empty()) {
        error = "pre-v4 ASTC Vulkan manifest cannot contain artifact records";
        return false;
    }
    uint64_t previous_end = 0;
    std::unordered_set<std::string> names;
    names.reserve(manifest.tensors.size() + manifest.artifacts.size());
    for (const astc_vulkan_tensor_record & tensor : manifest.tensors) {
        if (!validate_tensor_record(tensor, manifest.version, error)) return false;
        if (!names.insert(tensor.name).second) {
            error = "duplicate ASTC Vulkan tensor name";
            return false;
        }
        if (tensor.byte_offset < previous_end) {
            error = "invalid ASTC Vulkan tensor byte range";
            return false;
        }
        previous_end = tensor.byte_offset + tensor.byte_size;
    }
    std::unordered_set<std::string> artifact_ids;
    artifact_ids.reserve(manifest.artifacts.size());
    for (const auto & artifact : manifest.artifacts) {
        if (!validate_artifact_record(artifact, error)) return false;
        if (!artifact_ids.insert(artifact.id).second) {
            error = "duplicate ASTC Vulkan artifact id";
            return false;
        }
        if (artifact.storage.byte_offset < previous_end) {
            error = "invalid ASTC Vulkan artifact payload range";
            return false;
        }
        previous_end = artifact.storage.byte_offset + artifact.storage.byte_size;
    }
    error.clear();
    return true;
}

const astc_vulkan_tensor_record * astc_vulkan_find_tensor(
    const astc_vulkan_manifest & manifest, const std::string & name) {
    for (const astc_vulkan_tensor_record & tensor : manifest.tensors) {
        if (tensor.name == name) return &tensor;
    }
    return nullptr;
}

const astc_vulkan_artifact_record * astc_vulkan_find_artifact(
    const astc_vulkan_manifest & manifest, const std::string & id) {
    for (const auto & artifact : manifest.artifacts) {
        if (artifact.id == id) return &artifact;
    }
    return nullptr;
}

bool astc_vulkan_write_manifest(const std::string & path,
                                const astc_vulkan_manifest & manifest,
                                std::string & error) {
    if (!astc_vulkan_validate_manifest(manifest, error)) return false;
    std::ofstream file(path, std::ios::binary);
    if (!file) { error = "cannot open ASTC Vulkan manifest for writing"; return false; }
    file.write(kMagic.data(), kMagic.size());
    const bool ok = file.good() && write_scalar(file, manifest.version) &&
                    write_string(file, manifest.model_fingerprint) &&
                    write_scalar(file, static_cast<uint32_t>(manifest.tensors.size())) &&
                    (manifest.version < kArtifactManifestVersion ||
                     write_scalar(file, static_cast<uint32_t>(manifest.artifacts.size())));
    if (!ok) { error = "cannot write ASTC Vulkan manifest header"; return false; }
    for (const astc_vulkan_tensor_record & tensor : manifest.tensors) {
        if (!write_tensor_record(file, tensor, manifest.version)) {
            error = "cannot write ASTC Vulkan tensor record";
            return false;
        }
    }
    if (manifest.version >= kArtifactManifestVersion) {
        for (const auto & artifact : manifest.artifacts) {
            if (!write_artifact_record(file, artifact, manifest.version)) {
                error = "cannot write ASTC Vulkan artifact record";
                return false;
            }
        }
    }
    error.clear();
    return true;
}

bool astc_vulkan_read_manifest(const std::string & path,
                               astc_vulkan_manifest & manifest,
                               std::string & error) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { error = "cannot open ASTC Vulkan manifest for reading"; return false; }
    std::array<char, 8> magic{};
    file.read(magic.data(), magic.size());
    uint32_t version = 0;
    uint32_t count = 0;
    uint32_t artifact_count = 0;
    if (!file.good() || magic != kMagic || !read_scalar(file, version) ||
        !read_string(file, manifest.model_fingerprint) || !read_scalar(file, count) ||
        count > kMaxTensorRecords) {
        error = "invalid ASTC Vulkan manifest header";
        return false;
    }
    manifest.version = version;
    if (version >= kArtifactManifestVersion &&
        (!read_scalar(file, artifact_count) || artifact_count > kMaxTensorRecords)) {
        error = "invalid ASTC Vulkan artifact manifest header";
        return false;
    }
    manifest.tensors.clear();
    manifest.artifacts.clear();
    manifest.tensors.reserve(count);
    for (uint32_t index = 0; index < count; ++index) {
        astc_vulkan_tensor_record tensor;
        if (!read_tensor_record(file, tensor, version)) {
            error = "truncated ASTC Vulkan tensor record";
            return false;
        }
        manifest.tensors.push_back(std::move(tensor));
    }
    manifest.artifacts.reserve(artifact_count);
    for (uint32_t index = 0; index < artifact_count; ++index) {
        astc_vulkan_artifact_record artifact;
        if (!read_artifact_record(file, artifact, version)) {
            error = "truncated ASTC Vulkan artifact record";
            return false;
        }
        manifest.artifacts.push_back(std::move(artifact));
    }
    return astc_vulkan_validate_manifest(manifest, error);
}
