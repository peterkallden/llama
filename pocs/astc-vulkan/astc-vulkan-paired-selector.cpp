#include "astc-vulkan-paired-selector.h"

#include <algorithm>
#include <cmath>

namespace {

bool valid_vector(const std::vector<double> & values, size_t expected_size) {
    return values.size() == expected_size && std::all_of(values.begin(), values.end(),
        [](double value) { return std::isfinite(value); });
}

bool zero_vector(const std::vector<double> & values) {
    return std::all_of(values.begin(), values.end(),
        [](double value) { return value == 0.0; });
}

double squared_norm(const std::vector<double> & values) {
    double result = 0.0;
    for (double value : values) result += value * value;
    return result;
}

double gain(const std::vector<double> & residual, const std::vector<double> & delta) {
    double residual_dot_delta = 0.0;
    double delta_norm = 0.0;
    for (size_t index = 0; index < residual.size(); ++index) {
        residual_dot_delta += residual[index] * delta[index];
        delta_norm += delta[index] * delta[index];
    }
    return 2.0 * residual_dot_delta - delta_norm;
}

void subtract(std::vector<double> & residual, const std::vector<double> & delta) {
    for (size_t index = 0; index < residual.size(); ++index) residual[index] -= delta[index];
}

} // namespace

bool astc_vulkan_select_paired_candidates(
        const astc_vulkan_paired_selector_config & config,
        const std::vector<double> & initial_calibration_residual,
        const std::vector<double> & initial_validation_residual,
        const std::vector<std::vector<astc_vulkan_paired_candidate_delta>> & candidates,
        astc_vulkan_paired_selection_result & result) {
    const size_t calibration_size = static_cast<size_t>(config.logical_output_rows) *
                                    config.calibration_samples;
    const size_t validation_size = static_cast<size_t>(config.logical_output_rows) *
                                  config.validation_samples;
    const bool has_validation = config.validation_samples != 0;
    if (config.logical_output_rows == 0 || config.calibration_samples == 0 || candidates.empty() ||
        !valid_vector(initial_calibration_residual, calibration_size) ||
        (has_validation ? !valid_vector(initial_validation_residual, validation_size) :
                          !initial_validation_residual.empty())) return false;

    for (const auto & block_candidates : candidates) {
        if (block_candidates.empty() ||
            !valid_vector(block_candidates.front().calibration_delta, calibration_size) ||
            !zero_vector(block_candidates.front().calibration_delta) ||
            (has_validation &&
             (!valid_vector(block_candidates.front().validation_delta, validation_size) ||
              !zero_vector(block_candidates.front().validation_delta)))) return false;
        for (const auto & candidate : block_candidates) {
            if (!valid_vector(candidate.calibration_delta, calibration_size) ||
                (has_validation && !valid_vector(candidate.validation_delta, validation_size)) ||
                (!has_validation && !candidate.validation_delta.empty())) return false;
        }
    }

    result = {};
    result.calibration_selected_candidates.assign(candidates.size(), 0);
    std::vector<double> calibration_residual = initial_calibration_residual;
    std::vector<bool> committed(candidates.size(), false);

    while (true) {
        uint32_t best_block = 0;
        uint32_t best_candidate = 0;
        double best_gain = 0.0;
        bool found = false;
        for (uint32_t block = 0; block < candidates.size(); ++block) {
            if (committed[block]) continue;
            const auto & block_candidates = candidates[block];
            for (uint32_t candidate = 1; candidate < block_candidates.size(); ++candidate) {
                const double candidate_gain = gain(calibration_residual,
                                                   block_candidates[candidate].calibration_delta);
                if (candidate_gain > best_gain ||
                    (candidate_gain == best_gain && found &&
                     (block < best_block || (block == best_block && candidate < best_candidate)))) {
                    best_block = block;
                    best_candidate = candidate;
                    best_gain = candidate_gain;
                    found = true;
                }
            }
        }
        if (!found || !(best_gain > 0.0)) break;
        subtract(calibration_residual, candidates[best_block][best_candidate].calibration_delta);
        committed[best_block] = true;
        result.calibration_selected_candidates[best_block] = best_candidate;
        result.commits.push_back({best_block, best_candidate, best_gain});
    }
    result.calibration_residual_loss = squared_norm(calibration_residual);

    if (!has_validation) {
        result.validation_selected_candidates = result.calibration_selected_candidates;
        result.validation_prefix = static_cast<uint32_t>(result.commits.size());
        return true;
    }

    std::vector<double> validation_residual = initial_validation_residual;
    result.validation_selected_candidates.assign(candidates.size(), 0);
    result.validation_residual_loss = squared_norm(validation_residual);
    for (uint32_t commit_index = 0; commit_index < result.commits.size(); ++commit_index) {
        const auto & commit = result.commits[commit_index];
        subtract(validation_residual, candidates[commit.block][commit.candidate].validation_delta);
        const double loss = squared_norm(validation_residual);
        if (loss < result.validation_residual_loss) {
            result.validation_residual_loss = loss;
            result.validation_prefix = commit_index + 1;
        }
    }
    for (uint32_t index = 0; index < result.validation_prefix; ++index) {
        const auto & commit = result.commits[index];
        result.validation_selected_candidates[commit.block] = commit.candidate;
    }
    return true;
}
