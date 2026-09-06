#include "astc-vulkan-model-cache.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <unordered_map>
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

const astc_vulkan_tensor_usage_metrics * find_usage(
        const std::unordered_map<std::string, const astc_vulkan_tensor_usage_metrics *> & by_name,
        const std::string & name) {
    const auto it = by_name.find(name);
    return it == by_name.end() ? nullptr : it->second;
}

double finite_nonnegative(double value) {
    return std::isfinite(value) && value > 0.0 ? value : 0.0;
}

bool same_storage_key(const astc_vulkan_model_cache_storage_key & lhs,
                      const astc_vulkan_model_cache_storage_key & rhs) {
    return lhs.footprint == rhs.footprint &&
           lhs.representation == rhs.representation &&
           lhs.paired_semantic == rhs.paired_semantic &&
           lhs.normalization == rhs.normalization &&
           lhs.has_row_scales == rhs.has_row_scales;
}

bool parse_u64(const std::string & text, uint64_t & result) {
    try {
        size_t consumed = 0;
        const unsigned long long value = std::stoull(text, &consumed, 10);
        if (consumed != text.size()) return false;
        result = static_cast<uint64_t>(value);
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_double(const std::string & text, double & result) {
    try {
        size_t consumed = 0;
        result = std::stod(text, &consumed);
        return consumed == text.size() && std::isfinite(result);
    } catch (...) {
        return false;
    }
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
            entry.paired_semantic = astc_vulkan_paired_semantic::direct_rgb;
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
        entry.paired_semantic = best->paired_semantic;
        entry.has_row_scales = best->row_scale_byte_size != 0;
        entry.row_scale_byte_offset = best->row_scale_byte_offset;
        entry.row_scale_byte_size = best->row_scale_byte_size;
        entry.device_bytes = best->storage.byte_size;
        entry.host_bytes = best->storage.byte_size + best->storage.layout_byte_size +
            best->row_scale_byte_size;
        result.entries.push_back(std::move(entry));
    }
    error.clear();
    return true;
}

bool astc_vulkan_model_cache_plan_usage(
    const astc_vulkan_model_cache_plan & base_plan,
    const std::vector<astc_vulkan_tensor_usage_metrics> & usage,
    const astc_vulkan_model_cache_usage_options & options,
    const astc_vulkan_memory_budget & budget,
    astc_vulkan_model_cache_plan & result,
    std::string & error) {
    std::unordered_map<std::string, const astc_vulkan_tensor_usage_metrics *> by_name;
    by_name.reserve(usage.size());
    for (const auto & metric : usage) {
        if (metric.tensor_name.empty()) {
            error = "tensor usage metrics contain an empty tensor name";
            return false;
        }
        if (!by_name.emplace(metric.tensor_name, &metric).second) {
            error = "tensor usage metrics contain a duplicate tensor name: " + metric.tensor_name;
            return false;
        }
    }

    result = base_plan;
    for (auto & entry : result.entries) {
        entry.usage_available = false;
        entry.heat_score = 0.0;
        entry.benefit_score = 0.0;
        entry.expected_gpu_time_saved_ns = 0.0;
        if (entry.use_native_fallback) continue;

        const auto * metric = find_usage(by_name, entry.tensor_name);
        if (metric == nullptr) {
            if (options.require_usage_metrics) entry.use_native_fallback = true;
            continue;
        }
        entry.usage_available = true;
        const double invocations = metric->invocations != 0 ?
            static_cast<double>(metric->invocations) :
            static_cast<double>(metric->tokens_seen);
        const double path_probability = std::clamp(
            finite_nonnegative(metric->path_probability), 0.0, 1.0);
        const double multiplier = invocations * path_probability;
        const double time_saved_per_call = std::max(
            finite_nonnegative(metric->native_gpu_time_ns) -
            finite_nonnegative(metric->astc_gpu_time_ns), 0.0);
        const double bytes_saved_per_call = metric->native_bytes_read > metric->astc_bytes_read ?
            static_cast<double>(metric->native_bytes_read - metric->astc_bytes_read) : 0.0;
        entry.expected_gpu_time_saved_ns = time_saved_per_call * multiplier;
        const double byte_benefit = options.allow_byte_benefit_fallback ?
            bytes_saved_per_call * multiplier : 0.0;
        const double benefit = entry.expected_gpu_time_saved_ns > 0.0 ?
            entry.expected_gpu_time_saved_ns : byte_benefit;
        const double astc_bytes_per_call = metric->astc_bytes_read != 0 ?
            static_cast<double>(metric->astc_bytes_read) :
            static_cast<double>(entry.device_bytes);
        entry.heat_score = astc_bytes_per_call * multiplier;
        const double quality_cost = finite_nonnegative(entry.evidence.loss_delta);
        entry.benefit_score = quality_cost > 0.0 ? benefit / quality_cost : benefit;
        if (options.require_positive_benefit && benefit <= 0.0) {
            entry.use_native_fallback = true;
        }
    }

    std::stable_sort(result.entries.begin(), result.entries.end(),
        [&](const auto & lhs, const auto & rhs) {
            if (lhs.use_native_fallback != rhs.use_native_fallback) {
                return !lhs.use_native_fallback;
            }
            if (lhs.benefit_score != rhs.benefit_score) {
                return lhs.benefit_score > rhs.benefit_score;
            }
            if (lhs.heat_score != rhs.heat_score) {
                return lhs.heat_score > rhs.heat_score;
            }
            const auto * left_usage = find_usage(by_name, lhs.tensor_name);
            const auto * right_usage = find_usage(by_name, rhs.tensor_name);
            const uint32_t left_order = left_usage == nullptr ? UINT32_MAX : left_usage->execution_order;
            const uint32_t right_order = right_usage == nullptr ? UINT32_MAX : right_usage->execution_order;
            if (left_order != right_order) return left_order < right_order;
            return false;
        });

    const astc_vulkan_model_cache_plan ordered_plan = result;
    if (!astc_vulkan_model_cache_plan_residency(ordered_plan, budget, result, error)) {
        return false;
    }
    error.clear();
    return true;
}

bool astc_vulkan_model_cache_make_storage_pages(
    const astc_vulkan_model_cache_plan & plan,
    uint64_t max_page_payload_bytes,
    std::vector<astc_vulkan_model_cache_storage_page> & pages,
    std::string & error) {
    pages.clear();
    for (size_t index = 0; index < plan.entries.size(); ++index) {
        const auto & entry = plan.entries[index];
        if (entry.use_native_fallback) continue;
        astc_vulkan_model_cache_storage_key key;
        key.footprint = entry.storage.footprint;
        key.representation = entry.storage.representation;
        key.paired_semantic = entry.paired_semantic;
        key.normalization = entry.normalization;
        key.has_row_scales = entry.has_row_scales;

        astc_vulkan_model_cache_storage_page * page = nullptr;
        for (auto & candidate : pages) {
            if (same_storage_key(candidate.key, key) &&
                (max_page_payload_bytes == 0 ||
                 candidate.payload_bytes + entry.device_bytes <= max_page_payload_bytes)) {
                page = &candidate;
                break;
            }
        }
        if (page == nullptr) {
            pages.push_back({});
            page = &pages.back();
            page->key = key;
        }
        page->entry_indices.push_back(index);
        page->payload_bytes += entry.device_bytes;
        page->host_bytes += entry.host_bytes;
    }
    error.clear();
    return true;
}

bool astc_vulkan_read_tensor_usage_metrics(
    const std::string & path,
    std::vector<astc_vulkan_tensor_usage_metrics> & result,
    std::string & error) {
    result.clear();
    std::ifstream input(path);
    if (!input) {
        error = "cannot open ASTC usage metrics: " + path;
        return false;
    }
    std::string line;
    size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        std::istringstream fields(line);
        std::string name;
        if (!(fields >> name) || name[0] == '#') continue;
        std::array<std::string, 8> values{};
        for (auto & value : values) {
            if (!(fields >> value)) {
                error = "invalid ASTC usage metrics at line " + std::to_string(line_number);
                return false;
            }
        }
        std::string extra;
        if (fields >> extra) {
            error = "too many ASTC usage metrics fields at line " + std::to_string(line_number);
            return false;
        }
        astc_vulkan_tensor_usage_metrics metric;
        metric.tensor_name = name;
        uint64_t execution_order = 0;
        if (!parse_u64(values[0], metric.invocations) ||
            !parse_u64(values[1], metric.tokens_seen) ||
            !parse_u64(values[2], metric.native_bytes_read) ||
            !parse_u64(values[3], metric.astc_bytes_read) ||
            !parse_double(values[4], metric.native_gpu_time_ns) ||
            !parse_double(values[5], metric.astc_gpu_time_ns) ||
            !parse_double(values[6], metric.path_probability) ||
            !parse_u64(values[7], execution_order) || execution_order > UINT32_MAX ||
            metric.path_probability < 0.0 || metric.path_probability > 1.0) {
            error = "invalid ASTC usage metrics value at line " + std::to_string(line_number);
            return false;
        }
        metric.execution_order = static_cast<uint32_t>(execution_order);
        if (std::any_of(result.begin(), result.end(), [&](const auto & existing) {
                return existing.tensor_name == metric.tensor_name;
            })) {
            error = "duplicate ASTC usage metrics tensor: " + metric.tensor_name;
            return false;
        }
        result.push_back(std::move(metric));
    }
    if (!input.eof() && input.fail()) {
        error = "cannot read ASTC usage metrics: " + path;
        return false;
    }
    error.clear();
    return true;
}

bool astc_vulkan_write_tensor_usage_metrics(
    const std::string & path,
    const std::vector<astc_vulkan_tensor_usage_metrics> & metrics,
    std::string & error) {
    std::unordered_set<std::string> names;
    const std::string partial = path + ".partial";
    std::ofstream output(partial, std::ios::trunc);
    if (!output) {
        error = "cannot open ASTC usage metrics output: " + path;
        return false;
    }
    output << "# astc-usage-v1\n"
              "# tensor invocations tokens native_bytes astc_bytes native_gpu_ns "
              "astc_gpu_ns path_probability execution_order\n";
    output << std::setprecision(17);
    for (const auto & metric : metrics) {
        if (metric.tensor_name.empty() || !names.insert(metric.tensor_name).second ||
            !std::isfinite(metric.native_gpu_time_ns) ||
            !std::isfinite(metric.astc_gpu_time_ns) ||
            !std::isfinite(metric.path_probability) ||
            metric.path_probability < 0.0 || metric.path_probability > 1.0) {
            output.close();
            std::error_code cleanup_ec;
            std::filesystem::remove(partial, cleanup_ec);
            error = "invalid or duplicate ASTC usage metrics tensor";
            return false;
        }
        output << metric.tensor_name << ' ' << metric.invocations << ' '
               << metric.tokens_seen << ' ' << metric.native_bytes_read << ' '
               << metric.astc_bytes_read << ' ' << metric.native_gpu_time_ns << ' '
               << metric.astc_gpu_time_ns << ' ' << metric.path_probability << ' '
               << metric.execution_order << '\n';
    }
    output.close();
    if (!output) {
        std::error_code cleanup_ec;
        std::filesystem::remove(partial, cleanup_ec);
        error = "cannot write ASTC usage metrics output: " + path;
        return false;
    }
    std::error_code ec;
    std::filesystem::rename(partial, path, ec);
    if (ec) {
        std::filesystem::remove(partial, ec);
        error = "cannot publish ASTC usage metrics output: " + path;
        return false;
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
    result.runtime_base = result.validation.runtime_base;
    if (!astc_vulkan_model_cache_make_plan(
            result.validation.manifest, options, result.plan, error)) return false;
    error.clear();
    return true;
}

bool astc_vulkan_model_cache_load_catalog_for_runtime(
    const std::string & source_model_path,
    const std::string & runtime_model_path,
    const std::string & requested_cache_path,
    const astc_vulkan_model_cache_plan_options & options,
    astc_vulkan_model_cache_catalog & result,
    std::string & error) {
    result = {};
    if (source_model_path.empty() || runtime_model_path.empty()) {
        error = "runtime catalog requires source and runtime GGUF paths";
        return false;
    }
    if (!astc_vulkan_cache_validate(
            source_model_path, requested_cache_path, result.validation, error)) return false;
    if (source_model_path == runtime_model_path) {
        result.runtime_base = result.validation.runtime_base;
    } else if (!astc_vulkan_cache_validate_runtime_base(
                   source_model_path, runtime_model_path, requested_cache_path,
                   result.runtime_base, error)) {
        return false;
    }
    if (!options.allow_unverified &&
        (!result.runtime_base.model_gate_passed || !result.runtime_base.vulkan_gate_passed)) {
        error = "runtime GGUF has structural ASTC admission but no model/Vulkan quality gate";
        return false;
    }
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
