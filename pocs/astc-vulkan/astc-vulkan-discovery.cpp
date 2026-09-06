#include "astc-vulkan-discovery.h"

#include "astc-vulkan-paired-layout.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <unordered_map>

namespace {

const astc_vulkan_tensor_usage_metrics * find_usage(
        const std::unordered_map<std::string, const astc_vulkan_tensor_usage_metrics *> & by_name,
        const std::string & name) {
    const auto it = by_name.find(name);
    return it == by_name.end() ? nullptr : it->second;
}

uint32_t storage_height(astc_vulkan_representation representation, uint32_t rows) {
    return representation == astc_vulkan_representation::kPairedD2 ?
        astc_vulkan_paired_storage_height(rows) : rows;
}

uint64_t layout_bytes(astc_vulkan_representation representation,
                      astc_vulkan_footprint footprint, uint32_t columns,
                      uint32_t rows) {
    return representation == astc_vulkan_representation::kPairedD2 ?
        astc_vulkan_paired_layout_bytes(footprint, columns, rows) : 0;
}

} // namespace

bool astc_vulkan_discover_cache_candidates(
        const std::vector<astc_vulkan_discovery_tensor_input> & tensors,
        const std::vector<astc_vulkan_tensor_usage_metrics> & usage,
        const astc_vulkan_discovery_options & options,
        std::vector<astc_vulkan_discovery_entry> & result,
        std::string & error) {
    result.clear();
    if (!astc_vulkan_footprint_is_valid(options.footprint)) {
        error = "discovery requires a valid ASTC footprint";
        return false;
    }
    if (options.representation != astc_vulkan_representation::kScalar &&
        options.representation != astc_vulkan_representation::kGaugeLumaAlpha &&
        options.representation != astc_vulkan_representation::kPairedD2) {
        error = "discovery supports scalar, gauge-la or paired-d2 representations";
        return false;
    }
    if (options.representation == astc_vulkan_representation::kPairedD2 &&
        (options.footprint != astc_vulkan_footprint::k6x5 &&
         options.footprint != astc_vulkan_footprint::k8x5 &&
         options.footprint != astc_vulkan_footprint::k10x5)) {
        error = "paired-d2 discovery supports only 6x5, 8x5 and 10x5";
        return false;
    }

    std::unordered_map<std::string, const astc_vulkan_tensor_usage_metrics *> by_name;
    by_name.reserve(usage.size());
    for (const auto & metric : usage) {
        if (metric.tensor_name.empty() || !by_name.emplace(metric.tensor_name, &metric).second) {
            error = "discovery usage contains an empty or duplicate tensor name";
            return false;
        }
        if (!std::isfinite(metric.path_probability) || metric.path_probability < 0.0) {
            error = "discovery usage contains an invalid path probability";
            return false;
        }
    }

    for (const auto & tensor : tensors) {
        if (!tensor.rank2 || tensor.tensor_name.empty() || tensor.columns == 0 || tensor.rows == 0 ||
            tensor.source_bytes < options.min_source_bytes) continue;
        const auto * metric = find_usage(by_name, tensor.tensor_name);
        if (metric == nullptr && options.require_usage) continue;
        astc_vulkan_discovery_entry entry;
        entry.tensor_name = tensor.tensor_name;
        entry.columns = tensor.columns;
        entry.rows = tensor.rows;
        entry.source_bytes = tensor.source_bytes;
        entry.estimated_astc_bytes = astc_vulkan_image_bytes(
            options.footprint, tensor.columns,
            storage_height(options.representation, tensor.rows));
        entry.estimated_layout_bytes = layout_bytes(options.representation, options.footprint,
                                                    tensor.columns, tensor.rows);
        const uint64_t total_astc = entry.estimated_astc_bytes + entry.estimated_layout_bytes;
        entry.estimated_bytes_saved = tensor.source_bytes > total_astc ?
            tensor.source_bytes - total_astc : 0;
        if (metric != nullptr) {
            entry.usage_available = true;
            entry.invocations = metric->invocations;
            entry.tokens_seen = metric->tokens_seen;
            entry.execution_order = metric->execution_order;
            entry.path_probability = metric->path_probability;
            const double traffic = metric->native_bytes_read > 0 ?
                static_cast<double>(metric->native_bytes_read) :
                static_cast<double>(tensor.source_bytes);
            entry.heat_score = static_cast<double>(metric->invocations) *
                std::max(0.0, metric->path_probability) * traffic;
            const double saved_fraction = tensor.source_bytes == 0 ? 0.0 :
                static_cast<double>(entry.estimated_bytes_saved) /
                static_cast<double>(tensor.source_bytes);
            entry.priority_score = entry.heat_score * std::max(0.0, saved_fraction);
        }
        result.push_back(std::move(entry));
    }

    std::stable_sort(result.begin(), result.end(), [](const auto & lhs, const auto & rhs) {
        if (lhs.priority_score != rhs.priority_score) return lhs.priority_score > rhs.priority_score;
        if (lhs.execution_order != rhs.execution_order) return lhs.execution_order < rhs.execution_order;
        return lhs.tensor_name < rhs.tensor_name;
    });
    uint64_t selected_bytes = 0;
    size_t selected_count = 0;
    for (auto & entry : result) {
        const bool count_ok = options.max_tensors == 0 || selected_count < options.max_tensors;
        const uint64_t entry_bytes = entry.estimated_astc_bytes + entry.estimated_layout_bytes;
        const bool budget_ok = options.max_cache_bytes == 0 ||
            (selected_bytes <= options.max_cache_bytes &&
             entry_bytes <= options.max_cache_bytes - selected_bytes);
        if (count_ok && budget_ok && entry.priority_score > 0.0) {
            entry.selected = true;
            selected_bytes += entry_bytes;
            ++selected_count;
        }
    }
    error.clear();
    return true;
}

bool astc_vulkan_write_discovery_report(
        const std::string & path,
        const astc_vulkan_discovery_options & options,
        const std::vector<astc_vulkan_discovery_entry> & entries,
        std::string & error) {
    if (path.empty()) {
        error = "discovery report path is empty";
        return false;
    }
    const std::filesystem::path output_path(path);
    const std::filesystem::path partial = output_path.string() + ".partial";
    std::ofstream output(partial, std::ios::trunc);
    if (!output) {
        error = "cannot open discovery report: " + path;
        return false;
    }
    const auto format = astc_vulkan_format(options.footprint);
    output << "# astc-discovery-v1\n"
           << "# footprint=" << format.block_width << 'x' << format.block_height
           << " representation=" << static_cast<unsigned>(options.representation) << '\n'
           << "# tensor columns rows source_bytes astc_bytes layout_bytes bytes_saved"
              " invocations tokens path_probability heat priority usage selected quality_probe\n";
    output << std::setprecision(17);
    for (const auto & entry : entries) {
        output << entry.tensor_name << '\t' << entry.columns << '\t' << entry.rows << '\t'
               << entry.source_bytes << '\t' << entry.estimated_astc_bytes << '\t'
               << entry.estimated_layout_bytes << '\t' << entry.estimated_bytes_saved << '\t'
               << entry.invocations << '\t' << entry.tokens_seen << '\t'
               << entry.path_probability << '\t' << entry.heat_score << '\t'
               << entry.priority_score << '\t' << (entry.usage_available ? 1 : 0) << '\t'
               << (entry.selected ? 1 : 0) << '\t' << (entry.quality_probe_required ? 1 : 0)
               << '\n';
    }
    output.close();
    if (!output) {
        error = "cannot write discovery report: " + path;
        return false;
    }
    std::error_code ec;
    std::filesystem::rename(partial, output_path, ec);
    if (ec) {
        std::filesystem::remove(partial);
        error = "cannot publish discovery report: " + path;
        return false;
    }
    error.clear();
    return true;
}
