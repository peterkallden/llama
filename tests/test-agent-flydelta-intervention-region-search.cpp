#include "agent/adaptation/flydelta/flydelta-intervention-region-search.h"

#include <algorithm>
#include <cmath>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

static common_flydelta_experiment_fixture fixture() {
    common_flydelta_experiment_fixture value;
    value.id = "flydelta://fixture/intervention-region";
    value.task_fingerprint = "sha256:region-task";
    value.model_profile_fingerprint = "sha256:region-model";
    value.tokenizer_fingerprint = "sha256:region-tokenizer";
    value.template_fingerprint = "sha256:region-template";
    value.execution_context_fingerprint = "sha256:region-context";
    value.verifier_revision = "verifier:region-v1";
    return value;
}

int main() {
    common_flydelta_intervention_region_search_config config;
    config.available_layers = {1, 2, 3, 4};
    config.scales = {0.02f, 0.04f, 0.08f, 0.16f};
    config.max_singleton_layers = 4;
    config.max_neighborhoods = 2;
    config.max_trials = 32;
    config.max_stalled_scales = 4;
    config.min_cosine = 0.3f;
    config.max_leakage = 1.0f;
    config.max_shift_norm = 1.0f;
    std::string error;
    CHECK(common_flydelta_intervention_region_search_config_validate(config, error));

    std::vector<common_flydelta_intervention_region_trial> trials;
    common_flydelta_intervention_region_selection selection;
    CHECK(common_flydelta_run_intervention_region_search(
        fixture(), config,
        [](const auto &, const auto * candidate, bool apply,
                auto & trial, auto & margin, auto & geometry, auto & geometry_available,
                std::string &) {
            trial = {};
            trial.executed = true;
            trial.verifier_known = true;
            trial.evidence_ref = apply ? "evidence:region-arm" : "evidence:region-baseline";
            if (!apply) {
                trial.passed = false;
                trial.quality = 0.0f;
                return true;
            }
            const bool has_layer_two = std::find(candidate->layer_indices.begin(),
                candidate->layer_indices.end(), 2U) != candidate->layer_indices.end();
            trial.passed = candidate->layer_indices.size() == 2 && has_layer_two &&
                candidate->total_scale >= 0.08f;
            trial.quality = trial.passed ? 1.0f : 0.0f;
            geometry_available = true;
            geometry.layer_index = candidate->anchor_layer_index;
            geometry.cosine = candidate->anchor_layer_index == 2 ? 0.8f : 0.4f;
            geometry.progress = candidate->anchor_layer_index == 2 &&
                candidate->total_scale >= 0.04f ? 0.2f : 0.0f;
            geometry.leakage = 0.1f;
            geometry.shift_norm = candidate->total_scale;
            margin = {};
            return true;
        }, trials, selection, error));
    CHECK(error.empty());
    CHECK(trials.size() == 24);
    CHECK(selection.selected);
    CHECK(trials[selection.trial_index].candidate.layer_indices.size() == 1 ||
          trials[selection.trial_index].candidate.layer_indices.size() == 2);
    CHECK(trials[selection.trial_index].outcome ==
        common_flydelta_counterfactual_outcome::helped);

    bool saw_layer_two_scale_ladder = false;
    bool saw_pair = false;
    for (const auto & trial : trials) {
        CHECK(trial.executed && trial.verifier_known);
        CHECK(trial.geometry_available);
        CHECK(std::isfinite(trial.geometry.leakage));
        saw_layer_two_scale_ladder = saw_layer_two_scale_ladder ||
            (trial.candidate.layer_indices == std::vector<uint32_t>{2});
        saw_pair = saw_pair || trial.candidate.layer_indices.size() == 2;
    }
    CHECK(saw_layer_two_scale_ladder);
    CHECK(saw_pair);
    return 0;
}
