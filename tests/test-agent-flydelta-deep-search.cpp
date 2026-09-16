#include "agent/adaptation/flydelta/flydelta-deep-search.h"

#include <cmath>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

static common_flydelta_experiment_fixture fixture() {
    return {1, "fixture:deep-search", "task", "model", "tokenizer",
        "template", "execution", "verifier"};
}

static common_flydelta_direction_candidate direction(
        common_flydelta_direction_kind kind, float x, float y, float alignment) {
    common_flydelta_direction_candidate value;
    value.kind = kind;
    value.layer_index = 2;
    value.values = {x, y};
    value.source_samples = 6;
    value.retained_samples = 6;
    value.median_alignment = alignment;
    value.experimental_only = true;
    return value;
}

int main() {
    std::string error;
    common_flydelta_deep_search_config config;
    config.max_rank = 2;
    config.max_directions = 3;
    config.full_generation_top_k = 2;
    config.coefficients.max_candidates = 8;
    config.coefficients.step = 0.1f;
    config.coefficients.max_l2_norm = 0.32f;

    std::vector<common_flydelta_deep_search_direction> input = {
        {direction(common_flydelta_direction_kind::raw_repair, 1.0f, 0.0f, 0.4f), true, 0.2f},
        {direction(common_flydelta_direction_kind::diagonal_whitened_mean, 0.0f, 1.0f, 0.9f), true, 0.8f},
        {direction(common_flydelta_direction_kind::token_margin_direction, 1.0f, 1.0f, 0.1f), true, 0.1f},
    };
    std::vector<common_flydelta_deep_search_direction> selected;
    CHECK(common_flydelta_select_deep_search_directions(config, input, selected, error));
    CHECK(selected.size() == 2);
    CHECK(selected.front().direction.kind ==
        common_flydelta_direction_kind::diagonal_whitened_mean);

    size_t calls = 0;
    common_flydelta_deep_search_result result;
    CHECK(common_flydelta_run_deep_search(
        fixture(), config, input,
        [&](const common_flydelta_experiment_fixture &,
                const common_flydelta_low_rank_basis & basis,
                const std::vector<float> & coefficients, bool apply_overlay,
                common_flydelta_counterfactual_trial & trial,
                common_flydelta_decision_margin & margin,
                common_flydelta_representation_diagnostics & geometry,
                bool & geometry_available, std::string &) {
            ++calls;
            trial = {};
            trial.executed = true;
            trial.verifier_known = true;
            trial.overlay_applied = apply_overlay;
            trial.evidence_ref = "evidence:deep-search";
            trial.passed = apply_overlay && coefficients.size() == 2 &&
                coefficients[0] > 0.0f;
            trial.quality = trial.passed ? 1.0f : 0.0f;
            margin = {};
            margin.available = apply_overlay;
            margin.positive_total_logprob = coefficients.empty() ? 0.0f : coefficients[0];
            margin.negative_total_logprob = 0.0f;
            margin.positive_token_count = 1;
            margin.negative_token_count = 1;
            geometry = {};
            geometry_available = apply_overlay;
            geometry.layer_index = static_cast<uint32_t>(basis.layer_index);
            geometry.cosine = 0.8f;
            geometry.progress = coefficients.empty() ? 0.0f : coefficients[0];
            geometry.leakage = 0.05f;
            geometry.shift_norm = std::fabs(geometry.progress);
            return true;
        },
        [&](const common_flydelta_experiment_fixture &,
                const common_flydelta_low_rank_basis & basis,
                const std::vector<float> & coefficients, bool apply_overlay,
                common_flydelta_counterfactual_trial & trial,
                common_flydelta_decision_margin & margin,
                common_flydelta_representation_diagnostics & geometry,
                bool & geometry_available, std::string &) {
            trial = {};
            trial.executed = true;
            trial.verifier_known = true;
            trial.overlay_applied = apply_overlay;
            trial.evidence_ref = "evidence:deep-search-full";
            trial.passed = apply_overlay && coefficients.size() == 2 && coefficients[0] > 0.0f;
            trial.quality = trial.passed ? 1.0f : 0.0f;
            margin = {};
            geometry = {};
            geometry_available = false;
            return true;
        }, result, error));
    CHECK(result.layer_index == 2);
    CHECK(result.basis.vectors.size() == 2);
    CHECK(result.coefficient_trials.size() == 6); // four diagnostic + two full top arms
    CHECK(result.coefficient_selection.selected);
    CHECK(calls == 5); // diagnostic baseline plus four bounded coefficient arms
    CHECK(result.coefficient_trials[result.coefficient_selection.trial_index].outcome ==
        common_flydelta_counterfactual_outcome::helped);
    return 0;
}
