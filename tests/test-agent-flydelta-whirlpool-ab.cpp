#include "agent/adaptation/flydelta/flydelta-intervention-region-search.h"
#include "agent/adaptation/flydelta/flydelta-whirlpool-search.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

static common_flydelta_experiment_fixture fixture() {
    return {
        1, "fixture:whirlpool-ab", "task", "model", "tokenizer",
        "template", "execution-context", "verifier"
    };
}

struct scripted_landscape {
    size_t calls = 0;
    size_t first_helped_call = 0;
};

static void evaluate_layer(
        scripted_landscape & landscape,
        const common_flydelta_intervention_region_candidate * candidate,
        common_flydelta_counterfactual_trial & trial,
        common_flydelta_decision_margin & margin,
        common_flydelta_representation_diagnostics & geometry,
        bool & geometry_available) {
    ++landscape.calls;
    trial = {};
    trial.executed = true;
    trial.verifier_known = true;
    trial.evidence_ref = "evidence:whirlpool-ab";
    margin = {};
    geometry = {};
    geometry_available = candidate != nullptr;
    if (candidate == nullptr) return;

    const uint32_t layer = candidate->anchor_layer_index;
    const bool helped = layer == 5;
    trial.passed = helped;
    trial.quality = helped ? 1.0f : 0.0f;
    geometry.layer_index = layer;
    geometry.cosine = helped ? 0.9f : 0.45f;
    geometry.progress = helped ? 0.8f : 0.05f;
    geometry.leakage = 0.05f;
    geometry.shift_norm = candidate->total_scale;
    if (helped && landscape.first_helped_call == 0) {
        landscape.first_helped_call = landscape.calls;
    }
}

int main() {
    const auto test_fixture = fixture();
    const std::vector<uint32_t> layers = {1, 2, 3, 4, 5, 6, 7};

    // The fixed region search and Whirlpool receive the same four singleton
    // opportunities and the same host-verifier landscape. The only useful
    // region is layer 5, which is also the diagnostic prior for Whirlpool.
    common_flydelta_intervention_region_search_config fixed_config;
    fixed_config.available_layers = layers;
    fixed_config.singleton_layers = {1, 3, 5, 7};
    fixed_config.scales = {0.05f};
    fixed_config.max_singleton_layers = 4;
    fixed_config.max_neighborhoods = 0;
    fixed_config.max_trials = 4;
    fixed_config.max_stalled_scales = 1;
    fixed_config.min_cosine = 0.3f;
    fixed_config.max_leakage = 1.0f;
    fixed_config.max_shift_norm = 1.0f;

    scripted_landscape fixed_landscape;
    std::vector<common_flydelta_intervention_region_trial> fixed_trials;
    common_flydelta_intervention_region_selection fixed_selection;
    std::string error;
    CHECK(common_flydelta_run_intervention_region_search(
        test_fixture, fixed_config,
        [&](const auto &, const auto * candidate, bool,
                auto & trial, auto & margin, auto & geometry,
                bool & geometry_available, auto &) {
            evaluate_layer(fixed_landscape, candidate, trial, margin, geometry,
                geometry_available);
            return true;
        }, fixed_trials, fixed_selection, error));
    CHECK(error.empty());
    CHECK(fixed_trials.size() == 4);
    CHECK(fixed_landscape.calls == 5); // baseline + four fixed arms
    CHECK(fixed_landscape.first_helped_call == 4);
    CHECK(fixed_selection.selected);
    CHECK(fixed_trials[fixed_selection.trial_index].candidate.anchor_layer_index == 5);

    common_flydelta_whirlpool_search_config whirlpool_config;
    whirlpool_config.available_layers = layers;
    whirlpool_config.seed_layers = {3, 5};
    whirlpool_config.layer_diagnostics = {
        {5, 0.9f, 0.8f, 0.05f, 0.05f},
    };
    whirlpool_config.total_scale = 0.05f;
    whirlpool_config.max_rounds = 1;
    whirlpool_config.probes_per_round = 4;
    whirlpool_config.max_trials = 4;
    whirlpool_config.initial_radius = 2;
    whirlpool_config.shrink_factor = 0.5f;
    whirlpool_config.min_cosine = 0.3f;
    whirlpool_config.max_leakage = 1.0f;
    whirlpool_config.max_shift_norm = 1.0f;

    scripted_landscape whirlpool_landscape;
    std::vector<common_flydelta_intervention_region_trial> whirlpool_trials;
    common_flydelta_intervention_region_selection whirlpool_selection;
    common_flydelta_whirlpool_trace trace;
    CHECK(common_flydelta_run_whirlpool_search(
        test_fixture, whirlpool_config,
        [&](const auto &, const auto * candidate, auto & trial, auto & margin,
                auto & geometry, bool & geometry_available, auto &) {
            evaluate_layer(whirlpool_landscape, candidate, trial, margin, geometry,
                geometry_available);
            return true;
        }, whirlpool_trials, whirlpool_selection, trace, error));
    CHECK(error.empty());
    CHECK(whirlpool_trials.size() == 4);
    CHECK(whirlpool_landscape.calls == 5); // same maximum evaluation budget
    CHECK(whirlpool_landscape.first_helped_call == 2);
    CHECK(whirlpool_selection.selected);
    CHECK(whirlpool_trials[whirlpool_selection.trial_index].candidate.anchor_layer_index == 5);
    CHECK(trace.model_evaluations == whirlpool_trials.size() + 1);
    CHECK(trace.rounds.size() == 1);
    CHECK(trace.rounds.front().centre_before == 5);
    CHECK(trace.rounds.front().best_probe_layer == 5);
    CHECK(trace.rounds.front().centre_after == 5);

    // This measures the intended search property, not a claim that Whirlpool
    // always wins: with an equal maximum budget it reaches the verifier-known
    // region earlier because it recentres from the diagnostic prior.
    CHECK(whirlpool_landscape.first_helped_call < fixed_landscape.first_helped_call);
    return 0;
}
