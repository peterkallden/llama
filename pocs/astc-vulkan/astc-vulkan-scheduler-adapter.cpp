#include "astc-vulkan-scheduler-adapter.h"

#include "astc-vulkan-cache.h"

#include <filesystem>
#include <fstream>
#include <limits>
#include <cstring>
#include <cmath>
#include <utility>

namespace {
bool read_range(const std::string & path, uint64_t offset, uint64_t size,
                std::vector<uint8_t> & result) {
    if (size == 0 || size > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
        offset > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max()) ||
        size > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) return false;
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;
    file.seekg(static_cast<std::streamoff>(offset));
    if (!file) return false;
    result.resize(static_cast<size_t>(size));
    file.read(reinterpret_cast<char *>(result.data()), static_cast<std::streamsize>(size));
    return file.good() || file.gcount() == static_cast<std::streamsize>(size);
}

bool get_file_size(const std::string & path, uint64_t & size) {
    std::error_code ec;
    size = std::filesystem::file_size(path, ec);
    return !ec;
}

double rate_bpw(const astc_vulkan_tensor_record & record) {
    const uint64_t logical_weights = static_cast<uint64_t>(record.width) * record.height;
    return logical_weights == 0 ? 0.0 : static_cast<double>(record.byte_size) * 8.0 /
        static_cast<double>(logical_weights);
}

bool read_row_scales(const astc_vulkan_cache_validation & cache,
                     const astc_vulkan_artifact_record & source,
                     astc_vulkan_scheduler_artifact & artifact,
                     std::string & error) {
    if (source.normalization == astc_vulkan_normalization::none) return true;
    if (source.row_scale_byte_size != static_cast<uint64_t>(source.storage.height) * sizeof(float) ||
        source.row_scale_byte_size > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        error = "ASTC scheduler adapter cannot stream artifact row scales";
        return false;
    }
    std::vector<uint8_t> raw;
    if (!read_range(cache.paths.row_scales, source.row_scale_byte_offset,
                    source.row_scale_byte_size, raw)) {
        error = "ASTC scheduler adapter cannot stream artifact row scales";
        return false;
    }
    artifact.row_scales.resize(source.storage.height);
    std::memcpy(artifact.row_scales.data(), raw.data(),
                static_cast<size_t>(source.row_scale_byte_size));
    return true;
}

bool read_pair_map(const astc_vulkan_cache_validation & cache,
                   const astc_vulkan_artifact_record & source,
                   astc_vulkan_scheduler_artifact & artifact,
                   std::string & error) {
    if (source.pair_map_byte_size == 0) return true;
    uint64_t file_size = 0;
    if (!get_file_size(cache.paths.pair_map, file_size) ||
        source.pair_map_byte_offset > file_size ||
        source.pair_map_byte_size > file_size - source.pair_map_byte_offset ||
        !read_range(cache.paths.pair_map, source.pair_map_byte_offset,
                    source.pair_map_byte_size, artifact.pair_map) ||
        !astc_vulkan_validate_pair_map(source, artifact.pair_map.data(),
                                       artifact.pair_map.size(), error)) {
        if (error.empty()) error = "ASTC scheduler adapter cannot stream artifact pair map";
        return false;
    }
    return true;
}

bool materialize_artifact(const astc_vulkan_cache_validation & cache,
                          const astc_vulkan_artifact_record * source,
                          astc_vulkan_scheduler_artifact & artifact,
                          std::string & error) {
    if (source == nullptr) {
        error = "ASTC cache has no matching evidence-approved artifact";
        return false;
    }
    const auto & record = source->storage;
    uint64_t payload_size = 0;
    uint64_t layout_size = 0;
    if (!get_file_size(cache.paths.payload, payload_size) ||
        record.byte_offset > payload_size || record.byte_size > payload_size - record.byte_offset ||
        !read_range(cache.paths.payload, record.byte_offset, record.byte_size, artifact.payload)) {
        error = "ASTC scheduler adapter cannot stream tensor payload";
        return false;
    }
    artifact.record = record;
    artifact.artifact_id = source->id;
    artifact.variant = source->variant;
    artifact.normalization = source->normalization;
    artifact.paired_semantic = source->paired_semantic;
    artifact.evidence = source->evidence;
    artifact.cache_root = cache.paths.root;
    if (!read_row_scales(cache, *source, artifact, error)) {
        artifact = {};
        return false;
    }
    if (!read_pair_map(cache, *source, artifact, error)) {
        artifact = {};
        return false;
    }
    if (record.representation == astc_vulkan_representation::kPairedD2) {
        if (!get_file_size(cache.paths.layout, layout_size) ||
            record.layout_byte_offset > layout_size || record.layout_byte_size > layout_size - record.layout_byte_offset ||
            !read_range(cache.paths.layout, record.layout_byte_offset,
                        record.layout_byte_size, artifact.layout) ||
            !astc_vulkan_validate_layout_map(record, artifact.layout.data(), artifact.layout.size(), error)) {
            if (error.empty()) error = "ASTC scheduler adapter cannot stream paired-D2 layout";
            artifact = {};
            return false;
        }
        artifact.kind = astc_vulkan_scheduler_artifact_kind::kD2;
    } else if (record.representation == astc_vulkan_representation::kScalar ||
               record.representation == astc_vulkan_representation::kGaugeLumaAlpha) {
        artifact.kind = astc_vulkan_scheduler_artifact_kind::kD1;
    } else {
        error = "ASTC cache representation has no scheduler runtime";
        artifact = {};
        return false;
    }
    error.clear();
    return true;
}

// Paired D2 admission is deliberately opt-in and artifact/evidence based:
// direct manifests and legacy D2 caches cannot accidentally activate it.
// 6x5 and 8x5 are the profiles with current full-shape evidence gates.
bool is_explicit_d2_candidate(const astc_vulkan_scheduler_artifact & artifact,
                              bool allow_experimental, bool allow_unverified) {
    const bool supported_shape =
        artifact.record.footprint == astc_vulkan_footprint::k6x5 ||
        artifact.record.footprint == astc_vulkan_footprint::k8x5;
    if (artifact.kind != astc_vulkan_scheduler_artifact_kind::kD2 ||
        !supported_shape ||
        artifact.paired_semantic != astc_vulkan_paired_semantic::luminance_alpha) {
        return false;
    }
    const bool evidence_passed = artifact.evidence.model_gate_passed &&
        artifact.evidence.vulkan_gate_passed;
    return (evidence_passed || (allow_experimental && allow_unverified)) &&
           (allow_unverified || evidence_passed);
}
}

bool astc_vulkan_scheduler_adapter::resolve_from_cache(
        const std::string & model_path, const std::string & cache_path,
        const std::string & tensor_name, astc_vulkan_footprint footprint,
        astc_vulkan_scheduler_artifact & artifact, std::string & error,
        bool allow_unverified) const {
    artifact = {};
    astc_vulkan_cache_validation cache;
    if (!astc_vulkan_cache_validate(model_path, cache_path, cache, error)) return false;
    if (cache.manifest.version == 4) {
        return resolve_best_from_cache(model_path, cache_path, tensor_name, footprint,
                                       astc_vulkan_quality_policy::balanced, artifact, error,
                                       allow_unverified);
    }
    if (cache.manifest.version >= 5) {
        for (const auto & source : cache.manifest.artifacts) {
            if (source.storage.name == tensor_name && source.storage.footprint == footprint) {
                return materialize_artifact(cache, &source, artifact, error);
            }
        }
        error = "ASTC cache has no matching tensor artifact";
        return false;
    }
    const astc_vulkan_tensor_record * record = astc_vulkan_find_tensor(cache.manifest, tensor_name);
    if (record == nullptr || record->footprint != footprint) {
        error = "ASTC cache has no matching tensor artifact";
        return false;
    }
    uint64_t payload_size = 0;
    uint64_t layout_size = 0;
    if (!get_file_size(cache.paths.payload, payload_size) ||
        record->byte_offset > payload_size || record->byte_size > payload_size - record->byte_offset ||
        !read_range(cache.paths.payload, record->byte_offset, record->byte_size, artifact.payload)) {
        error = "ASTC scheduler adapter cannot stream tensor payload";
        return false;
    }
    artifact.record = *record;
    artifact.cache_root = cache.paths.root;
    if (record->representation == astc_vulkan_representation::kPairedD2) {
        if (!get_file_size(cache.paths.layout, layout_size) ||
            record->layout_byte_offset > layout_size || record->layout_byte_size > layout_size - record->layout_byte_offset ||
            !read_range(cache.paths.layout, record->layout_byte_offset,
                        record->layout_byte_size, artifact.layout) ||
            !astc_vulkan_validate_layout_map(*record, artifact.layout.data(), artifact.layout.size(), error)) {
            if (error.empty()) error = "ASTC scheduler adapter cannot stream paired-D2 layout";
            artifact = {};
            return false;
        }
        artifact.kind = astc_vulkan_scheduler_artifact_kind::kD2;
    } else if (record->representation == astc_vulkan_representation::kScalar ||
               record->representation == astc_vulkan_representation::kGaugeLumaAlpha) {
        artifact.kind = astc_vulkan_scheduler_artifact_kind::kD1;
    } else {
        error = "ASTC cache representation has no scheduler runtime";
        artifact = {};
        return false;
    }
    error.clear();
    return true;
}

bool astc_vulkan_scheduler_adapter::resolve_best_from_cache(
        const std::string & model_path, const std::string & cache_path,
        const std::string & tensor_name, astc_vulkan_footprint footprint,
        astc_vulkan_quality_policy policy, astc_vulkan_scheduler_artifact & artifact,
        std::string & error, bool allow_unverified) const {
    artifact = {};
    astc_vulkan_cache_validation cache;
    if (!astc_vulkan_cache_validate(model_path, cache_path, cache, error)) return false;
    if (cache.manifest.version != 4) {
        error = "ASTC cache does not provide an evidence-aware artifact index";
        return false;
    }
    const astc_vulkan_artifact_record * best = nullptr;
    astc_vulkan_artifact_candidate best_candidate;
    for (const auto & source : cache.manifest.artifacts) {
        if (source.storage.name != tensor_name || source.storage.footprint != footprint) continue;
        astc_vulkan_artifact_candidate candidate;
        candidate.tensor = &source.storage;
        candidate.variant = source.variant;
        candidate.normalization = source.normalization;
        candidate.evidence = source.evidence;
        candidate.rate_bpw = rate_bpw(source.storage);
        const bool metadata_eligible = candidate.tensor != nullptr &&
            std::isfinite(candidate.evidence.loss_delta) &&
            std::isfinite(candidate.evidence.logits_relative_mse) &&
            std::isfinite(candidate.rate_bpw) && candidate.rate_bpw > 0.0;
        if (allow_unverified ? !metadata_eligible :
                               !astc_vulkan_artifact_is_eligible(candidate, true, true)) continue;
        if (best == nullptr || astc_vulkan_artifact_policy_precedes(candidate, best_candidate, policy)) {
            best = &source;
            best_candidate = std::move(candidate);
        }
    }
    return materialize_artifact(cache, best, artifact, error);
}

bool astc_vulkan_scheduler_adapter::prepare(
        const std::string & manifest_path, const std::string & payload_blob_path,
        const std::string & tensor_name, astc_vulkan_footprint footprint,
        std::string & error, bool allow_experimental) {
    // Preparation is transactional: a failed reload must not leave a previous
    // tensor executable through ready() or run().
    reset();
    const auto fallback = [&](const std::string & reason, bool result) {
        binding_.status = astc_vulkan_binding_status::kFallback;
        binding_.fallback_reason = reason;
        error = reason;
        return result;
    };
    astc_vulkan_manifest manifest;
    uint64_t blob_size = 0;
    if (!get_file_size(payload_blob_path, blob_size) || blob_size == 0 ||
        !astc_vulkan_read_manifest(manifest_path, manifest, error) ||
        !astc_vulkan_validate_payload_blob(manifest, blob_size, error)) {
        return fallback(error.empty() ? "ASTC scheduler artifact is invalid" : error, false);
    }
    const astc_vulkan_tensor_record * record = astc_vulkan_find_tensor(manifest, tensor_name);
    if (record == nullptr || record->footprint != footprint ||
        record->byte_offset > blob_size || record->byte_size > blob_size - record->byte_offset) {
        return fallback("ASTC scheduler adapter tensor artifact is invalid", false);
    }
    // Direct manifests do not carry v4 artifact evidence. The production
    // boundary therefore remains restricted to standard D1 payloads; paired
    // D2 must arrive via prepare_from_cache() below.
    if (astc_vulkan_footprint_is_experimental(footprint) ||
        (record->representation != astc_vulkan_representation::kScalar &&
         record->representation != astc_vulkan_representation::kGaugeLumaAlpha)) {
        binding_.record = *record;
        return fallback("ASTC production adapter accepts only standard 4x4/5x5/6x6 scalar/gauge", true);
    }
    if (!read_range(payload_blob_path, record->byte_offset, record->byte_size, payload_)) {
        return fallback("ASTC scheduler adapter cannot stream tensor payload", false);
    }
    tensor_name_ = tensor_name;
    if (!sidecar_.set_manifest(manifest, error) ||
        !(shared_device_ ? sidecar_.init(shared_device_, footprint, error, allow_experimental) :
                          sidecar_.init(footprint, error, allow_experimental))) {
        return fallback(error.empty() ? "ASTC Vulkan device is unavailable" : error, false);
    }
    if (!sidecar_.bind_tensor(tensor_name, record->width, record->height, payload_, binding_, error)) {
        reset();
        return fallback(error.empty() ? "ASTC production adapter binding failed" : error, false);
    }
    if (binding_.status != astc_vulkan_binding_status::kReady) {
        if (binding_.fallback_reason.empty()) {
            binding_.fallback_reason = "ASTC production adapter binding requires normal fallback";
        }
        error = binding_.fallback_reason;
        return false;
    }
    error.clear();
    return true;
}

bool astc_vulkan_scheduler_adapter::prepare_from_cache(
        const std::string & model_path, const std::string & cache_path,
        const std::string & tensor_name, astc_vulkan_footprint footprint,
        std::string & error, bool allow_experimental, bool allow_unverified) {
    reset();
    astc_vulkan_scheduler_artifact artifact;
    if (!resolve_from_cache(model_path, cache_path, tensor_name, footprint, artifact, error,
                            allow_unverified)) {
        binding_.status = astc_vulkan_binding_status::kFallback;
        binding_.fallback_reason = error.empty() ? "ASTC cache is unavailable" : error;
        error = binding_.fallback_reason;
        return false;
    }
    return bind_materialized_artifact(std::move(artifact), error,
                                      allow_experimental, allow_unverified);
}

bool astc_vulkan_scheduler_adapter::prepare_artifact_from_cache(
        const std::string & model_path, const std::string & cache_path,
        const std::string & tensor_name, const std::string & artifact_id,
        std::string & error, bool allow_experimental, bool allow_unverified) {
    reset();
    astc_vulkan_cache_validation cache;
    if (!astc_vulkan_cache_validate(model_path, cache_path, cache, error)) {
        binding_.status = astc_vulkan_binding_status::kFallback;
        binding_.fallback_reason = error;
        return false;
    }
    if (cache.manifest.version != 4) {
        error = "exact artifact binding requires a v4 artifact index";
        binding_.status = astc_vulkan_binding_status::kFallback;
        binding_.fallback_reason = error;
        return false;
    }
    const astc_vulkan_artifact_record * source =
        astc_vulkan_find_artifact(cache.manifest, artifact_id);
    if (source == nullptr || source->storage.name != tensor_name) {
        error = "ASTC cache has no matching artifact id for tensor";
        binding_.status = astc_vulkan_binding_status::kFallback;
        binding_.fallback_reason = error;
        return false;
    }
    astc_vulkan_artifact_candidate candidate;
    candidate.tensor = &source->storage;
    candidate.variant = source->variant;
    candidate.normalization = source->normalization;
    candidate.evidence = source->evidence;
    candidate.rate_bpw = rate_bpw(source->storage);
    const bool metadata_eligible = std::isfinite(candidate.evidence.loss_delta) &&
        std::isfinite(candidate.evidence.logits_relative_mse) &&
        std::isfinite(candidate.rate_bpw) && candidate.rate_bpw > 0.0;
    if ((allow_unverified && !metadata_eligible) ||
        (!allow_unverified && !astc_vulkan_artifact_is_eligible(candidate, true, true))) {
        error = "requested ASTC artifact is not evidence-eligible";
        binding_.status = astc_vulkan_binding_status::kFallback;
        binding_.fallback_reason = error;
        return false;
    }
    astc_vulkan_scheduler_artifact artifact;
    if (!materialize_artifact(cache, source, artifact, error)) {
        binding_.status = astc_vulkan_binding_status::kFallback;
        binding_.fallback_reason = error;
        return false;
    }
    return bind_materialized_artifact(std::move(artifact), error,
                                      allow_experimental, allow_unverified);
}

bool astc_vulkan_scheduler_adapter::prepare_validated_artifact(
        astc_vulkan_scheduler_artifact artifact, std::string & error,
        bool allow_experimental, bool allow_unverified) {
    return bind_materialized_artifact(std::move(artifact), error,
                                      allow_experimental, allow_unverified);
}

bool astc_vulkan_scheduler_adapter::bind_materialized_artifact(
        astc_vulkan_scheduler_artifact artifact, std::string & error,
        bool allow_experimental, bool allow_unverified) {
    reset();
    if (artifact.kind == astc_vulkan_scheduler_artifact_kind::kD1 &&
        (astc_vulkan_footprint_is_experimental(artifact.record.footprint) ||
         (artifact.record.representation != astc_vulkan_representation::kScalar &&
          artifact.record.representation != astc_vulkan_representation::kGaugeLumaAlpha))) {
        binding_.record = artifact.record;
        binding_.status = astc_vulkan_binding_status::kFallback;
        binding_.fallback_reason = "ASTC production adapter accepts only standard 4x4/5x5/6x6 scalar/gauge";
        error = binding_.fallback_reason;
        return true;
    }
    if (artifact.kind == astc_vulkan_scheduler_artifact_kind::kD2 &&
        !is_explicit_d2_candidate(artifact, allow_experimental, allow_unverified)) {
        binding_.record = artifact.record;
        binding_.status = astc_vulkan_binding_status::kFallback;
        binding_.fallback_reason =
            "ASTC paired-D2 requires an explicit, evidence-approved D2_6x5 or D2_8x5 L+A artifact";
        error = binding_.fallback_reason;
        return true;
    }
    payload_ = std::move(artifact.payload);
    layout_ = std::move(artifact.layout);
    pair_map_ = std::move(artifact.pair_map);
    tensor_name_ = artifact.record.name;
    payload_path_ = artifact.cache_root.empty() ? std::string() :
        (std::filesystem::path(artifact.cache_root) / "payload.astcpack").string();
    payload_offset_ = artifact.record.byte_offset;
    binding_.record = artifact.record;
    astc_vulkan_manifest runtime_manifest;
    runtime_manifest.version = 3;
    runtime_manifest.tensors = {artifact.record};
    if (!sidecar_.set_manifest(runtime_manifest, error) ||
        !(shared_device_ ? sidecar_.init(shared_device_, artifact.record.footprint, error, allow_experimental) :
                          sidecar_.init(artifact.record.footprint, error, allow_experimental)) ||
        !sidecar_.bind_tensor(tensor_name_, artifact.record.width, artifact.record.height,
                              payload_, binding_, error, layout_, artifact.paired_semantic,
                              artifact.row_scales, pair_map_)) {
        binding_.status = astc_vulkan_binding_status::kFallback;
        binding_.fallback_reason = error.empty() ? "ASTC scheduler binding failed" : error;
        error = binding_.fallback_reason;
        return false;
    }
    if (binding_.status != astc_vulkan_binding_status::kReady) {
        if (binding_.fallback_reason.empty()) {
            binding_.fallback_reason = "ASTC scheduler binding requires normal fallback";
        }
        error = binding_.fallback_reason;
        return true;
    }
    error.clear();
    return true;
}

bool astc_vulkan_scheduler_adapter::run(const std::vector<uint32_t> & spirv,
                                        const std::vector<float> & activations,
                                        std::vector<float> & output, std::string & error) {
    if (!ready()) {
        error = "ASTC scheduler adapter is not ready; use normal fallback";
        return false;
    }
    return sidecar_.run(spirv, activations, output, error);
}

bool astc_vulkan_scheduler_adapter::run_streamed(
        uint64_t max_resident_payload_bytes, const std::vector<uint32_t> & spirv,
        const std::vector<float> & activations, std::vector<float> & output,
        std::string & error) {
    if (!ready() || payload_path_.empty()) {
        error = "ASTC scheduler streamed run requires a cache-backed artifact";
        return false;
    }
    // The cache-backed prepare path materializes the payload once to validate
    // and bind the artifact. Streaming replaces that resident image, so drop
    // the full host payload before range reads begin; only the current band
    // and the (small) paired metadata remain resident.
    std::vector<uint8_t>().swap(payload_);
    return sidecar_.run_streamed(payload_path_, payload_offset_,
                                 max_resident_payload_bytes, spirv, activations,
                                 output, error);
}
