#include "astc-vulkan-model-cache.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <unordered_set>
#include <utility>

namespace {

namespace fs = std::filesystem;

double rate_bpw(const astc_vulkan_tensor_record & record) {
    const uint64_t weights = static_cast<uint64_t>(record.width) * record.height;
    return weights == 0 ? 0.0 : static_cast<double>(record.byte_size) * 8.0 /
        static_cast<double>(weights);
}

bool metadata_eligible(const astc_vulkan_artifact_record & artifact,
                       bool allow_unverified,
                       bool allow_experimental) {
    if (!allow_experimental && astc_vulkan_footprint_is_experimental(
            artifact.storage.footprint)) {
        return false;
    }
    if (allow_unverified) {
        return std::isfinite(artifact.evidence.loss_delta) &&
               std::isfinite(artifact.evidence.logits_relative_mse) &&
               rate_bpw(artifact.storage) > 0.0;
    }
    astc_vulkan_artifact_candidate candidate;
    candidate.tensor = &artifact.storage;
    candidate.variant = artifact.variant;
    candidate.normalization = artifact.normalization;
    candidate.evidence = artifact.evidence;
    candidate.rate_bpw = rate_bpw(artifact.storage);
    return astc_vulkan_artifact_is_eligible(candidate, true, true);
}

bool tensor_already_added(const std::vector<astc_vulkan_model_cache_entry> & entries,
                          const std::string & name) {
    return std::any_of(entries.begin(), entries.end(), [&](const auto & entry) {
        return entry.tensor_name == name;
    });
}

void add_fallback(const std::string & name,
                  astc_vulkan_model_cache_plan & result) {
    astc_vulkan_model_cache_entry entry;
    entry.tensor_name = name;
    entry.use_native_fallback = true;
    result.entries.push_back(std::move(entry));
}

bool append_range(const std::string & source_path,
                  uint64_t source_offset,
                  uint64_t source_size,
                  std::ofstream & output,
                  uint64_t & output_offset,
                  std::string & error) {
    std::error_code ec;
    const uint64_t source_file_size = fs::file_size(source_path, ec);
    if (ec || source_offset > source_file_size ||
        source_size > source_file_size - source_offset) {
        error = "fragment range exceeds source file: " + source_path;
        return false;
    }
    if (source_size == 0) {
        output_offset = static_cast<uint64_t>(output.tellp());
        return output.good();
    }
    if (source_offset > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max()) ||
        source_size > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) {
        error = "fragment range is too large for stream I/O";
        return false;
    }
    std::ifstream input(source_path, std::ios::binary);
    if (!input) {
        error = "cannot open fragment payload: " + source_path;
        return false;
    }
    input.seekg(static_cast<std::streamoff>(source_offset));
    if (!input) {
        error = "cannot seek fragment payload: " + source_path;
        return false;
    }
    output_offset = static_cast<uint64_t>(output.tellp());
    std::vector<char> buffer(1u << 20);
    uint64_t remaining = source_size;
    while (remaining != 0) {
        const std::streamsize count = static_cast<std::streamsize>(
            std::min<uint64_t>(remaining, buffer.size()));
        input.read(buffer.data(), count);
        if (input.gcount() != count) {
            error = "cannot read fragment payload: " + source_path;
            return false;
        }
        output.write(buffer.data(), count);
        if (!output) {
            error = "cannot write merged ASTC cache payload";
            return false;
        }
        remaining -= static_cast<uint64_t>(count);
    }
    return true;
}

bool open_output_blob(const std::string & path, std::ofstream & output,
                      bool required, std::string & error) {
    if (path.empty()) {
        if (required) error = "merged cache requires an output blob path";
        return !required;
    }
    output.open(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        error = "cannot open merged cache blob: " + path;
        return false;
    }
    return true;
}

} // namespace

bool astc_vulkan_model_cache_make_plan(
    const astc_vulkan_manifest & manifest,
    const astc_vulkan_model_cache_plan_options & options,
    astc_vulkan_model_cache_plan & result,
    std::string & error) {
    result = {};
    if (manifest.version == 4 && manifest.artifacts.empty()) {
        error = "model cache plan has no v4 artifacts";
        return false;
    }

    if (manifest.version < 4) {
        for (const auto & tensor : manifest.tensors) {
            astc_vulkan_model_cache_entry entry;
            entry.tensor_name = tensor.name;
            entry.storage = tensor;
            entry.device_bytes = tensor.byte_size;
            entry.host_bytes = tensor.byte_size + tensor.layout_byte_size;
            result.entries.push_back(std::move(entry));
        }
        error.clear();
        return true;
    }

    for (const auto & artifact : manifest.artifacts) {
        if (tensor_already_added(result.entries, artifact.storage.name)) continue;
        const astc_vulkan_artifact_record * best = nullptr;
        astc_vulkan_artifact_candidate best_candidate;
        for (const auto & candidate_artifact : manifest.artifacts) {
            if (candidate_artifact.storage.name != artifact.storage.name ||
                !metadata_eligible(candidate_artifact, options.allow_unverified,
                                   options.allow_experimental)) continue;
            astc_vulkan_artifact_candidate candidate;
            candidate.tensor = &candidate_artifact.storage;
            candidate.variant = candidate_artifact.variant;
            candidate.normalization = candidate_artifact.normalization;
            candidate.evidence = candidate_artifact.evidence;
            candidate.rate_bpw = rate_bpw(candidate_artifact.storage);
            if (best == nullptr || astc_vulkan_artifact_policy_precedes(
                    candidate, best_candidate, options.policy)) {
                best = &candidate_artifact;
                best_candidate = candidate;
            }
        }
        if (best == nullptr) {
            add_fallback(artifact.storage.name, result);
            continue;
        }
        astc_vulkan_model_cache_entry entry;
        entry.tensor_name = best->storage.name;
        entry.artifact_id = best->id;
        entry.storage = best->storage;
        entry.evidence = best->evidence;
        entry.variant = best->variant;
        entry.normalization = best->normalization;
        entry.device_bytes = best->storage.byte_size;
        entry.host_bytes = best->storage.byte_size + best->storage.layout_byte_size +
            best->row_scale_byte_size;
        result.entries.push_back(std::move(entry));
    }
    error.clear();
    return true;
}

bool astc_vulkan_model_cache_load_catalog(
    const std::string & model_path,
    const std::string & requested_cache_path,
    const astc_vulkan_model_cache_plan_options & options,
    astc_vulkan_model_cache_catalog & result,
    std::string & error) {
    result = {};
    if (!astc_vulkan_cache_validate(
            model_path, requested_cache_path, result.validation, error)) return false;
    if (!astc_vulkan_model_cache_make_plan(
            result.validation.manifest, options, result.plan, error)) return false;
    error.clear();
    return true;
}

bool astc_vulkan_model_cache_plan_residency(
    const astc_vulkan_model_cache_plan & model_plan,
    const astc_vulkan_memory_budget & budget,
    astc_vulkan_model_cache_plan & result,
    std::string & error) {
    result = model_plan;
    std::vector<astc_vulkan_residency_item> items;
    items.reserve(model_plan.entries.size());
    for (const auto & entry : model_plan.entries) {
        if (entry.use_native_fallback) continue;
        items.push_back({entry.tensor_name, entry.device_bytes, entry.host_bytes});
    }
    astc_vulkan_residency_plan residency;
    if (!astc_vulkan_plan_residency(items, budget, 0, residency, error)) return false;
    result.residency = std::move(residency);
    error.clear();
    return true;
}

bool astc_vulkan_model_cache_write_build_state(
    const std::string & path,
    const astc_vulkan_model_cache_build_state & state,
    std::string & error) {
    const std::string partial = path + ".partial";
    std::ofstream file(partial, std::ios::trunc);
    if (!file) { error = "cannot open model cache build-state staging file"; return false; }
    file << "version=1\nsource_model=" << state.source_model <<
        "\noutput_root=" << state.output_root << "\n";
    for (const auto & tensor : state.completed_tensors) file << "completed=" << tensor << "\n";
    file.close();
    if (!file) { error = "cannot write model cache build-state"; return false; }
    std::error_code ec;
    std::filesystem::rename(partial, path, ec);
    if (ec) { error = "cannot publish model cache build-state"; return false; }
    error.clear();
    return true;
}

bool astc_vulkan_model_cache_read_build_state(
    const std::string & path,
    astc_vulkan_model_cache_build_state & state,
    std::string & error) {
    state = {};
    std::ifstream file(path);
    if (!file) { error = "cannot open model cache build-state"; return false; }
    std::string line;
    while (std::getline(file, line)) {
        const size_t split = line.find('=');
        if (split == std::string::npos) continue;
        const std::string key = line.substr(0, split);
        const std::string value = line.substr(split + 1);
        if (key == "source_model") state.source_model = value;
        else if (key == "output_root") state.output_root = value;
        else if (key == "completed") state.completed_tensors.push_back(value);
    }
    if (state.source_model.empty() || state.output_root.empty()) {
        error = "invalid model cache build-state";
        return false;
    }
    error.clear();
    return true;
}

bool astc_vulkan_model_cache_merge_fragments(
    const std::vector<astc_vulkan_model_cache_fragment> & fragments,
    const std::string & output_manifest_path,
    const std::string & output_payload_path,
    const std::string & output_layout_path,
    const std::string & output_row_scales_path,
    astc_vulkan_manifest & result,
    std::string & error) {
    result = {};
    if (fragments.empty() || output_manifest_path.empty() || output_payload_path.empty()) {
        error = "model cache merge requires fragments, manifest, and payload paths";
        return false;
    }
    std::ofstream payload;
    if (!open_output_blob(output_payload_path, payload, true, error)) return false;
    std::ofstream layout;
    if (!open_output_blob(output_layout_path, layout, false, error)) return false;
    std::ofstream row_scales;
    if (!open_output_blob(output_row_scales_path, row_scales, false, error)) return false;

    result.version = 4;
    std::unordered_set<std::string> artifact_ids;
    for (const auto & fragment : fragments) {
        astc_vulkan_manifest local;
        if (!astc_vulkan_read_manifest(fragment.manifest_path, local, error)) return false;
        if (local.version != 4 || local.artifacts.empty()) {
            error = "model cache fragments must contain v4 artifacts";
            return false;
        }
        if (result.model_fingerprint.empty()) result.model_fingerprint = local.model_fingerprint;
        if (result.model_fingerprint != local.model_fingerprint) {
            error = "model cache fragment fingerprints differ";
            return false;
        }
        for (const auto & local_artifact : local.artifacts) {
            if (!artifact_ids.insert(local_artifact.id).second) {
                error = "duplicate artifact id in model cache fragments: " + local_artifact.id;
                return false;
            }
            auto artifact = local_artifact;
            uint64_t merged_offset = 0;
            if (!append_range(fragment.payload_path, artifact.storage.byte_offset,
                              artifact.storage.byte_size, payload, merged_offset, error)) {
                return false;
            }
            artifact.storage.byte_offset = merged_offset;
            if (artifact.storage.representation == astc_vulkan_representation::kPairedD2) {
                if (fragment.layout_path.empty() || output_layout_path.empty()) {
                    error = "paired-D2 fragment is missing a layout blob";
                    return false;
                }
                if (!append_range(fragment.layout_path, artifact.storage.layout_byte_offset,
                                  artifact.storage.layout_byte_size, layout,
                                  merged_offset, error)) return false;
                artifact.storage.layout_byte_offset = merged_offset;
            }
            if (artifact.normalization == astc_vulkan_normalization::per_row_absmax) {
                if (fragment.row_scales_path.empty() || output_row_scales_path.empty()) {
                    error = "scaled fragment is missing a row-scale blob";
                    return false;
                }
                if (!append_range(fragment.row_scales_path, artifact.row_scale_byte_offset,
                                  artifact.row_scale_byte_size, row_scales,
                                  merged_offset, error)) return false;
                artifact.row_scale_byte_offset = merged_offset;
            }
            result.artifacts.push_back(std::move(artifact));
        }
    }
    payload.close();
    layout.close();
    row_scales.close();
    if (!astc_vulkan_validate_manifest(result, error)) return false;

    std::error_code ec;
    const uint64_t payload_size = fs::file_size(output_payload_path, ec);
    if (ec || !astc_vulkan_validate_payload_blob(result, payload_size, error)) return false;
    bool has_layout = false;
    bool has_row_scales = false;
    for (const auto & artifact : result.artifacts) {
        has_layout |= artifact.storage.representation == astc_vulkan_representation::kPairedD2;
        has_row_scales |= artifact.normalization == astc_vulkan_normalization::per_row_absmax;
    }
    if (has_layout) {
        const uint64_t layout_size = fs::file_size(output_layout_path, ec);
        if (ec || !astc_vulkan_validate_layout_blob(result, layout_size, error)) return false;
    }
    if (has_row_scales) {
        const uint64_t row_scale_size = fs::file_size(output_row_scales_path, ec);
        if (ec) {
            error = "cannot stat merged row-scale blob";
            return false;
        }
        for (const auto & artifact : result.artifacts) {
            if (artifact.normalization == astc_vulkan_normalization::per_row_absmax &&
                (artifact.row_scale_byte_offset > row_scale_size ||
                 artifact.row_scale_byte_size > row_scale_size - artifact.row_scale_byte_offset)) {
                error = "merged row-scale range exceeds blob";
                return false;
            }
        }
    }
    if (!astc_vulkan_write_manifest(output_manifest_path, result, error)) return false;
    error.clear();
    return true;
}

bool astc_vulkan_model_cache_publish_fragments(
    const std::string & source_model_path,
    const std::vector<astc_vulkan_model_cache_fragment> & fragments,
    const std::string & staging_root,
    const std::string & requested_cache_path,
    astc_vulkan_cache_paths & paths,
    std::string & error) {
    if (source_model_path.empty() || staging_root.empty()) {
        error = "model cache publication requires source model and staging root";
        return false;
    }
    std::error_code ec;
    if (!fs::create_directories(staging_root, ec) && ec) {
        error = "cannot create model cache staging root";
        return false;
    }
    const fs::path root(staging_root);
    const std::string manifest = (root / "manifest.astcv").string();
    const std::string payload = (root / "payload.astcpack").string();
    const std::string layout = (root / "layout-map.bin").string();
    const std::string row_scales = (root / "row-scales.bin").string();
    astc_vulkan_manifest merged;
    if (!astc_vulkan_model_cache_merge_fragments(
            fragments, manifest, payload, layout, row_scales, merged, error)) return false;
    std::string layout_input;
    std::string row_scales_input;
    for (const auto & artifact : merged.artifacts) {
        if (artifact.storage.representation == astc_vulkan_representation::kPairedD2) layout_input = layout;
        if (artifact.normalization == astc_vulkan_normalization::per_row_absmax) row_scales_input = row_scales;
    }
    return astc_vulkan_cache_create_with_row_scales(
        source_model_path, manifest, payload, layout_input, row_scales_input,
        {}, requested_cache_path, paths, error);
}
