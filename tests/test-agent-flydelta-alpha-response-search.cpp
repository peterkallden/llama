#include "agent/adaptation/flydelta/flydelta-alpha-response-search.h"

#include <cmath>
#include <vector>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

static bool scripted_alpha_runner(
        const common_flydelta_experiment_fixture &, float scale, bool apply_overlay,
        common_flydelta_counterfactual_trial & counterfactual,
        common_flydelta_decision_margin & margin,
        common_flydelta_representation_diagnostics & geometry,
        bool & geometry_available, std::string &) {
    counterfactual = {};
    counterfactual.executed = true;
    counterfactual.verifier_known = true;
    counterfactual.passed = apply_overlay && scale >= 0.2f;
    counterfactual.quality = counterfactual.passed ? 1.0f : 0.0f;
    counterfactual.overlay_applied = apply_overlay;
    counterfactual.evidence_ref = "evidence:alpha-response";
    margin = {};
    margin.available = true;
    margin.positive_token_count = 1;
    margin.negative_token_count = 1;
    margin.positive_total_logprob = apply_overlay ? -1.0f + scale * 4.0f : -1.0f;
    margin.negative_total_logprob = -2.0f;
    geometry = {};
    geometry_available = apply_overlay;
    geometry.cosine = 0.8f;
    geometry.progress = scale;
    geometry.leakage = 0.05f;
    geometry.shift_norm = scale;
    return true;
}

int main() {
    std::string error;
    common_flydelta_alpha_response_search_config config;
    config.seed_scale = 0.05f;
    config.growth_factor = 1.6f;
    config.max_scale = 0.5f;
    config.max_expansion_trials = 4;
    config.max_zoom_trials = 4;
    config.max_min_effective_trials = 2;
    config.max_expansion_non_improving = 2;
    common_flydelta_alpha_response_selection selection;
    std::vector<common_flydelta_alpha_response_trial> trials;
    size_t calls = 0;
    CHECK(common_flydelta_run_alpha_response_search(
        common_flydelta_experiment_fixture{
            1, "fixture:alpha-response", "task", "model", "tokenizer",
            "template", "context", "verifier"
        }, config,
        [&](const common_flydelta_experiment_fixture &, float scale, bool apply_overlay,
                common_flydelta_counterfactual_trial & counterfactual,
                common_flydelta_decision_margin & margin,
                common_flydelta_representation_diagnostics & geometry,
                bool & geometry_available, std::string &) {
            ++calls;
            counterfactual = {};
            counterfactual.executed = true;
            counterfactual.verifier_known = true;
            counterfactual.passed = apply_overlay && scale >= 0.2f;
            counterfactual.quality = counterfactual.passed ? 1.0f : 0.0f;
            counterfactual.overlay_applied = apply_overlay;
            counterfactual.evidence_ref = "evidence:alpha-response";
            margin = {};
            margin.available = true;
            margin.positive_token_count = 1;
            margin.negative_token_count = 1;
            margin.positive_total_logprob = apply_overlay ? -1.0f + scale * 4.0f : -1.0f;
            margin.negative_total_logprob = -2.0f;
            geometry = {};
            geometry_available = apply_overlay;
            geometry.cosine = 0.8f;
            geometry.progress = scale;
            geometry.leakage = 0.05f;
            geometry.shift_norm = scale;
            return true;
        }, trials, selection, error));
    CHECK(calls > 3);
    CHECK(selection.selected);
    CHECK(selection.minimum_effective_available);
    CHECK(selection.minimum_effective_scale >= 0.19f &&
        selection.minimum_effective_scale <= 0.21f);
    CHECK(selection.response_status == common_flydelta_alpha_response_status::helped);
    CHECK(selection.best_margin_available && selection.best_margin_delta_normalized > 0.0f);
    CHECK(std::fabs(selection.scale - selection.minimum_effective_scale) > 0.0001f ||
        selection.scale >= 0.2f);
    for (const auto & trial : trials) {
        CHECK(common_flydelta_alpha_response_trial_validate(trial, error));
        CHECK(trial.margin_available);
    }

    common_flydelta_alpha_response_search_config no_selection_config = config;
    no_selection_config.utility_epsilon = 4.0f;
    common_flydelta_alpha_response_selection no_selection;
    std::vector<common_flydelta_alpha_response_trial> no_selection_trials;
    CHECK(common_flydelta_run_alpha_response_search(
        common_flydelta_experiment_fixture{
            1, "fixture:alpha-response-no-selection", "task", "model", "tokenizer",
            "template", "context", "verifier"
        }, no_selection_config, scripted_alpha_runner,
        no_selection_trials, no_selection, error));
    CHECK(!no_selection.selected);
    CHECK(!no_selection.minimum_effective_available);
    CHECK(no_selection.response_status == common_flydelta_alpha_response_status::saturated);

    common_flydelta_alpha_response_selection failing_selection;
    std::vector<common_flydelta_alpha_response_trial> failing_trials;
    CHECK(!common_flydelta_run_alpha_response_search(
        common_flydelta_experiment_fixture{
            1, "fixture:alpha-response-failure", "task", "model", "tokenizer",
            "template", "context", "verifier"
        }, config,
        [](const common_flydelta_experiment_fixture &, float, bool,
                common_flydelta_counterfactual_trial &, common_flydelta_decision_margin &,
                common_flydelta_representation_diagnostics &, bool &, std::string & runner_error) {
            runner_error = "synthetic runner failure";
            return false;
        }, failing_trials, failing_selection, error));
    CHECK(error == "synthetic runner failure");

    common_flydelta_alpha_response_search_config early_helped_config = config;
    early_helped_config.seed_scale = 0.2f;
    early_helped_config.growth_factor = 2.0f;
    early_helped_config.max_expansion_trials = 5;
    early_helped_config.max_zoom_trials = 4;
    std::vector<common_flydelta_alpha_response_trial> early_helped_trials;
    common_flydelta_alpha_response_selection early_helped_selection;
    CHECK(common_flydelta_run_alpha_response_search(
        common_flydelta_experiment_fixture{
            1, "fixture:alpha-response-early-helped", "task", "model", "tokenizer",
            "template", "context", "verifier"
        }, early_helped_config, scripted_alpha_runner,
        early_helped_trials, early_helped_selection, error));
    CHECK(early_helped_selection.minimum_effective_available);
    CHECK(early_helped_selection.response_status == common_flydelta_alpha_response_status::helped);
    // The normal golden phase is skipped, but the bounded minimum-effective
    // bracket is still allowed to add its midpoint probes.
    CHECK(early_helped_trials.size() == 3);
    CHECK(early_helped_trials.front().outcome ==
        common_flydelta_counterfactual_outcome::helped);

    // An unsafe requested alpha must be represented explicitly: the search
    // coordinate remains the requested value while execution may retry once
    // at the bounded dose proposed by the controller.
    common_flydelta_alpha_response_search_config dose_config;
    dose_config.seed_scale = 0.05f;
    dose_config.max_scale = 0.05f;
    dose_config.max_expansion_trials = 1;
    dose_config.max_zoom_trials = 0;
    dose_config.max_min_effective_trials = 0;
    dose_config.max_expansion_non_improving = 1;
    dose_config.max_shift_norm = 1.0f;
    dose_config.dose_policy.max_shift_norm = 1.0f;
    dose_config.dose_policy.max_leakage = 1.0f;
    dose_config.dose_policy.target_absolute_fraction = 0.5f;
    dose_config.dose_policy.backoff_safety_factor = 0.8f;
    common_flydelta_alpha_response_selection dose_selection;
    std::vector<common_flydelta_alpha_response_trial> dose_trials;
    CHECK(common_flydelta_run_alpha_response_search(
        common_flydelta_experiment_fixture{
            1, "fixture:alpha-response-dose-retry", "task", "model", "tokenizer",
            "template", "context", "verifier"
        }, dose_config,
        [](const common_flydelta_experiment_fixture &, float scale, bool apply_overlay,
                common_flydelta_counterfactual_trial & counterfactual,
                common_flydelta_decision_margin & margin,
                common_flydelta_representation_diagnostics & geometry,
                bool & geometry_available, std::string &) {
            counterfactual = {};
            counterfactual.executed = true;
            counterfactual.verifier_known = true;
            counterfactual.passed = false;
            counterfactual.overlay_applied = apply_overlay;
            counterfactual.evidence_ref = "evidence:alpha-dose-retry";
            margin = {};
            margin.available = true;
            margin.positive_token_count = 1;
            margin.negative_token_count = 1;
            margin.positive_total_logprob = -1.0f;
            margin.negative_total_logprob = -2.0f;
            geometry = {};
            geometry_available = apply_overlay;
            geometry.cosine = 0.8f;
            geometry.progress = scale;
            geometry.leakage = 0.05f;
            geometry.shift_norm = scale >= 0.05f ? 2.0f : scale;
            return true;
        }, dose_trials, dose_selection, error));
    CHECK(dose_trials.size() == 1);
    CHECK(dose_trials.front().requested_scale == 0.05f);
    CHECK(dose_trials.front().scale < dose_trials.front().requested_scale);
    CHECK(dose_trials.front().dose_evaluated);
    CHECK(dose_trials.front().dose_safety_limited);
    CHECK(dose_trials.front().dose_action == common_flydelta_dose_action::accept);
    CHECK(dose_trials.front().dose_reason.find("retry_lower:") == 0);
    CHECK(common_flydelta_alpha_response_trial_validate(dose_trials.front(), error));
    return 0;
}
