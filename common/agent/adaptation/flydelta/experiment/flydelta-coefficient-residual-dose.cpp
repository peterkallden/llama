#include "agent/adaptation/flydelta/flydelta-coefficient-search.h"

#include <algorithm>
#include <cmath>
#include <limits>

bool common_flydelta_intervention_norm(
        const common_flydelta_low_rank_basis & basis,
        const std::vector<float> & coefficients, float & value, std::string & error) {
    error.clear();
    value = 0;
    if (!common_flydelta_low_rank_basis_validate(basis, 16, error) ||
            coefficients.size() != basis.vectors.size()) {
        if (error.empty()) error = "FlyDelta intervention coefficient dimensions differ";
        return false;
    }
    for (float c : coefficients) if (!std::isfinite(c)) {
        error = "FlyDelta intervention coefficient is non-finite"; return false;
    }
    double squared = 0;
    for (size_t column = 0; column < basis.dimension; ++column) {
        double sum = 0;
        for (size_t row = 0; row < coefficients.size(); ++row)
            sum += double(coefficients[row]) * basis.vectors[row][column];
        squared += sum * sum;
    }
    const double result = std::sqrt(squared);
    if (!std::isfinite(result) || result > std::numeric_limits<float>::max()) {
        error = "FlyDelta intervention norm overflow"; return false;
    }
    value = static_cast<float>(result);
    return true;
}

bool common_flydelta_calibrate_residual_dose(
        const std::vector<common_flydelta_hidden_state_capture> & captures,
        int32_t layer_index, const std::string & reference,
        const std::string & token_selection,
        common_flydelta_residual_calibration & calibration, std::string & error) {
    calibration = {};
    error.clear();
    if (captures.empty() || captures.size() > 4096 || reference.empty() || token_selection.empty()) {
        error = "FlyDelta residual calibration requires bounded captures and provenance"; return false;
    }
    std::vector<double> norms;
    const auto & first = captures.front();
    if (layer_index < 0 || (first.position != common_flydelta_capture_position::prompt_row &&
            first.position != common_flydelta_capture_position::generation_boundary)) {
        error = "FlyDelta residual calibration position is invalid"; return false;
    }
    for (const auto & capture : captures) {
        if (!common_flydelta_hidden_state_capture_validate(capture, 64U * 1024U * 1024U, error)) return false;
        if (!capture.captured || capture.model_profile_fingerprint != first.model_profile_fingerprint ||
                capture.capture_layout_revision != first.capture_layout_revision ||
                capture.position != first.position || capture.n_embd != first.n_embd ||
                (capture.position == common_flydelta_capture_position::prompt_row &&
                 capture.token_index != first.token_index)) {
            error = "FlyDelta residual calibration capture contexts differ"; return false;
        }
        const auto it = std::find(capture.layer_indices.begin(), capture.layer_indices.end(), layer_index);
        if (it == capture.layer_indices.end()) {
            error = "FlyDelta residual calibration layer missing"; return false;
        }
        const size_t offset = size_t(it - capture.layer_indices.begin()) * capture.n_embd;
        double squared = 0;
        for (size_t i = 0; i < size_t(capture.n_embd); ++i) {
            const double v = capture.values[offset + i]; squared += v * v;
        }
        const double norm = std::sqrt(squared);
        if (!std::isfinite(norm) || norm <= 1e-12 || norm > std::numeric_limits<float>::max()) {
            error = "FlyDelta residual calibration norm is degenerate"; return false;
        }
        norms.push_back(norm);
    }
    std::sort(norms.begin(), norms.end());
    const size_t middle = norms.size() / 2;
    calibration.reference = reference;
    calibration.model_profile_fingerprint = first.model_profile_fingerprint;
    calibration.capture_layout_revision = first.capture_layout_revision;
    calibration.token_selection = token_selection;
    calibration.position = first.position;
    calibration.token_index = first.token_index;
    calibration.layer_index = layer_index;
    calibration.sample_count = norms.size();
    calibration.dimension = first.n_embd;
    calibration.sample_norms.assign(norms.begin(), norms.end());
    calibration.median_norm = static_cast<float>(norms.size() % 2
        ? norms[middle] : (norms[middle - 1] + norms[middle]) / 2);
    return true;
}

bool common_flydelta_propose_calibrated_coefficients(
        const common_flydelta_experiment_fixture & fixture,
        const common_flydelta_low_rank_basis & basis,
        const common_flydelta_coefficient_search_config & config,
        std::vector<std::vector<float>> & proposals, std::string & error,
        std::vector<float> * requested_fractions) {
    proposals.clear(); error.clear();
    if (requested_fractions) requested_fractions->clear();
    if (!common_flydelta_coefficient_search_config_validate(config, basis.vectors.size(), error) ||
            !common_flydelta_low_rank_basis_validate(basis, 16, error)) return false;
    if (config.experimental_dose_mode == common_flydelta_coefficient_dose_mode::raw_coefficients)
        return common_flydelta_propose_low_rank_coefficients(config, basis.vectors.size(), proposals, error);
    const auto & c = config.residual_calibration;
    if (c.schema_version != 1 || c.reference.empty() || c.sample_count == 0 || c.sample_count > 4096 ||
            c.dimension != basis.dimension || c.sample_norms.size() != c.sample_count ||
            !std::isfinite(c.median_norm) || c.median_norm <= 0 ||
            c.model_profile_fingerprint != fixture.model_profile_fingerprint ||
            c.tokenizer_fingerprint.empty() || c.tokenizer_fingerprint != fixture.tokenizer_fingerprint ||
            c.template_fingerprint.empty() || c.template_fingerprint != fixture.template_fingerprint ||
            c.layer_index != basis.layer_index || c.capture_layout_revision.empty() ||
            c.capture_layout_revision != config.injection_capture_layout_revision ||
            c.token_selection.empty() || c.token_selection != config.injection_token_selection ||
            c.position != config.injection_capture_position ||
            (c.position != common_flydelta_capture_position::prompt_row &&
             c.position != common_flydelta_capture_position::generation_boundary) ||
            (c.position == common_flydelta_capture_position::prompt_row &&
             c.token_index != config.injection_token_index)) {
        error = "FlyDelta residual calibration is missing or incompatible"; return false;
    }
    auto sorted_norms = c.sample_norms;
    for (float norm : sorted_norms) if (!std::isfinite(norm) || norm <= 0) {
        error = "FlyDelta residual calibration samples are invalid"; return false;
    }
    std::sort(sorted_norms.begin(), sorted_norms.end());
    const size_t middle = sorted_norms.size() / 2;
    const double median = sorted_norms.size() % 2 ? sorted_norms[middle]
        : (double(sorted_norms[middle - 1]) + sorted_norms[middle]) / 2;
    if (std::abs(median - c.median_norm) > 1e-6 * median) {
        error = "FlyDelta residual calibration median differs from samples"; return false;
    }
    std::vector<std::vector<float>> directions;
    auto stencil_config = config;
    // Generate the legacy directions independently from the dose grid size.
    stencil_config.max_candidates = 64;
    if (!common_flydelta_propose_low_rank_coefficients(stencil_config, basis.vectors.size(), directions, error))
        return false;
    if (config.experimental_dose_mode == common_flydelta_coefficient_dose_mode::direction_dose) {
        for (size_t i = 0; i < basis.vectors.size(); ++i)
            for (size_t j = i + 1; j < basis.vectors.size(); ++j)
                for (float sign : {-1.0f, 1.0f}) {
                    std::vector<float> d(basis.vectors.size(), 0);
                    d[i] = 1; d[j] = sign; directions.push_back(d);
                }
    }
    proposals.push_back(std::vector<float>(basis.vectors.size(), 0));
    if (requested_fractions) requested_fractions->push_back(0);
    const auto doses = config.experimental_dose_mode == common_flydelta_coefficient_dose_mode::direction_dose
        ? config.residual_fraction_candidates : std::vector<float>{config.residual_fraction};
    // Dose-major ordering explores every direction before increasing the dose.
    for (float dose : doses) for (size_t index = 1; index < directions.size(); ++index) {
        if (proposals.size() >= config.max_candidates) return true;
        auto candidate = directions[index];
        float magnitude = 0;
        if (!common_flydelta_intervention_norm(basis, candidate, magnitude, error)) return false;
        double upper = 0;
        for (size_t i = 0; i < candidate.size(); ++i) {
            double squared = 0;
            for (float v : basis.vectors[i]) squared += double(v) * v;
            upper += std::abs(candidate[i]) * std::sqrt(squared);
        }
        // Cancellation is not a direction to amplify. Skip it explicitly.
        if (magnitude <= 1e-6 * upper) continue;
        const double factor = double(dose) * c.median_norm / magnitude;
        double coefficient_squared = 0;
        for (float & v : candidate) {
            v = static_cast<float>(v * factor); coefficient_squared += double(v) * v;
        }
        if (!std::isfinite(coefficient_squared)) {
            error = "FlyDelta calibrated coefficients overflow"; return false;
        }
        const double coefficient_norm = std::sqrt(coefficient_squared);
        if (coefficient_norm > config.max_l2_norm)
            for (float & v : candidate) v *= config.max_l2_norm / coefficient_norm;
        bool duplicate = false;
        for (const auto & existing : proposals) {
            auto difference = candidate;
            for (size_t i = 0; i < difference.size(); ++i) difference[i] -= existing[i];
            float distance = 0;
            if (!common_flydelta_intervention_norm(basis, difference, distance, error)) return false;
            float actual = 0;
            if (!common_flydelta_intervention_norm(basis, candidate, actual, error)) return false;
            if (distance <= 1e-6f * std::max(actual, 1e-12f)) { duplicate = true; break; }
        }
        if (!duplicate) {
            proposals.push_back(std::move(candidate));
            if (requested_fractions) requested_fractions->push_back(dose);
        }
    }
    if (proposals.size() == 1 && config.max_candidates > 1) {
        error = "FlyDelta calibrated search has no usable directions"; return false;
    }
    return true;
}
