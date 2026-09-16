#include "agent/adaptation/flydelta/flydelta-whirlpool-search.h"

#include <algorithm>
#include <cmath>
#include <vector>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

static common_flydelta_experiment_fixture fixture() {
    return {
        1, "fixture:whirlpool", "task", "model", "tokenizer",
        "template", "execution-context", "verifier"
    };
}

int main() {
    common_flydelta_whirlpool_search_config config;
    config.available_layers = {1, 2, 3, 4, 5, 6, 7};
    config.seed_layers = {2, 5};
    config.layer_diagnostics = {
        {1, 0.2f, 0.01f, 0.1f, 0.0f},
        {2, 0.4f, 0.05f, 0.1f, 0.0f},
        {5, 0.7f, 0.20f, 0.1f, 0.0f},
    };
    config.total_scale = 0.02f;
    config.max_rounds = 2;
    config.probes_per_round = 4;
    config.max_trials = 8;
    config.initial_radius = 2;
    std::string error;
    CHECK(common_flydelta_whirlpool_search_config_validate(config, error));

    std::vector<common_flydelta_intervention_region_trial> trials;
    common_flydelta_intervention_region_selection selection;
    CHECK(common_flydelta_run_whirlpool_search(
        fixture(), config,
        [](const auto &, const auto * candidate, auto & trial, auto & margin,
                auto & geometry, auto & geometry_available, std::string &) {
            trial = {};
            trial.executed = true;
            trial.verifier_known = true;
            trial.evidence_ref = candidate == nullptr ? "evidence:baseline" : "evidence:arm";
            if (candidate == nullptr) {
                trial.passed = false;
                trial.quality = 0.0f;
                return true;
            }
            const uint32_t layer = candidate->anchor_layer_index;
            trial.passed = layer == 5;
            trial.quality = trial.passed ? 1.0f : 0.0f;
            geometry_available = true;
            geometry.layer_index = layer;
            geometry.cosine = layer == 5 ? 0.9f : 0.4f;
            geometry.progress = layer == 5 ? 0.8f : 0.1f;
            geometry.leakage = 0.05f;
            geometry.shift_norm = candidate->total_scale;
            margin = {};
            return true;
        }, trials, selection, error));
    CHECK(error.empty());
    CHECK(!trials.empty());
    CHECK(selection.selected);
    CHECK(trials[selection.trial_index].candidate.anchor_layer_index == 5);
    CHECK(trials[selection.trial_index].outcome ==
        common_flydelta_counterfactual_outcome::helped);
    CHECK(std::any_of(trials.begin(), trials.end(), [](const auto & trial) {
        return trial.candidate.anchor_layer_index == 5;
    }));

    common_flydelta_whirlpool_search_config invalid = config;
    invalid.probes_per_round = 1;
    CHECK(!common_flydelta_whirlpool_search_config_validate(invalid, error));
    return 0;
}
