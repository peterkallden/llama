#include "astc-vulkan-model-cache.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <utility>

namespace {

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
