#include "agent/adaptation/flydelta/flydelta-search-pipeline.h"

#include <cmath>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

static common_flydelta_experiment_fixture make_fixture() {
    return {
        1, "fixture:search-pipeline", "task", "model", "tokenizer",
        "template", "execution-context", "verifier"
    };
}

int main() {
    std::string error;
    common_flydelta_search_pipeline_config config;
    config.dimension = 2;
    config.layer.max_regions = 1;
    config.layer.max_singletons = 2;
    config.layer.max_neighborhoods = 2;
    config.layer.max_candidates = 4;
    config.layer.min_cosine = 0.3f;
    config.scale.initial_scale = 0.02f;
    config.scale.growth_factor = 2.0f;
    config.scale.max_scale = 0.16f;
    config.scale.max_geometric_trials = 4;
    config.scale.max_refinement_trials = 1;
    config.use_intervention_region_search = false;

    common_flydelta_search_pipeline_direction input;
    input.direction.layer_index = 2;
    input.direction.values = {1.0f, 0.0f};
    input.direction.source_samples = 2;
    input.direction.retained_samples = 2;
    input.direction.median_alignment = 0.9f;
    input.layer_diagnostics = {
        {1, 0.40f, 0.10f, 0.10f, 0.10f},
        {2, 0.90f, 0.80f, 0.10f, 0.10f},
        {3, 0.40f, 0.10f, 0.10f, 0.10f},
    };
    input.available_layers = {1, 2, 3, 4};

    size_t calls = 0;
    common_flydelta_search_pipeline_result result;
    CHECK(common_flydelta_run_search_pipeline(
        make_fixture(), config, {input},
        [&](const common_flydelta_experiment_fixture &,
                const common_flydelta_direction_candidate &,
                const common_flydelta_layer_candidate * layer,
                float scale, bool apply_overlay,
                common_flydelta_counterfactual_trial & trial,
                common_flydelta_decision_margin &,
                common_flydelta_scale_geometry & geometry,
                std::string &) {
            ++calls;
            trial = {};
            trial.executed = true;
            trial.verifier_known = true;
            trial.overlay_applied = apply_overlay;
            trial.evidence_ref = "evidence:search-pipeline";
            const bool pair = layer != nullptr && layer->layer_indices ==
                std::vector<uint32_t>{2, 3};
            trial.passed = apply_overlay && pair && scale >= 0.08f;
            trial.quality = trial.passed ? 1.0f : 0.0f;
            geometry = {};
            geometry.available = apply_overlay;
            geometry.cosine = 0.85f;
            geometry.progress = scale;
            geometry.leakage = 0.05f;
            geometry.shift_norm = scale;
            return true;
        }, result, error));

    CHECK(calls == 16); // baseline + 3 layer candidates × (baseline + 4 arms)
    CHECK(result.directions.size() == 1);
    const auto & direction = result.directions.front();
    CHECK(direction.layer_plan.singleton_candidates.size() == 1);
    CHECK(direction.layer_plan.neighborhood_candidates.size() == 2);
    CHECK(direction.layer_trials.size() == 3);
    CHECK(direction.layer_results.size() == 3);
    CHECK(result.selection.selected);
    CHECK(result.selection.layer_result_index == 2);
    CHECK(direction.layer_results[2].candidate.layer_indices ==
        std::vector<uint32_t>({2, 3}));
    CHECK(direction.layer_results[2].scale_selection.selected);
    CHECK(std::fabs(direction.layer_results[2].scale_selection.scale - 0.08f) < 0.00001f);
    CHECK(result.selection.scale == direction.layer_results[2].scale_selection.scale);

    common_learning_in_memory_lifecycle_store lifecycle;
    common_flydelta_lifecycle_event_context lifecycle_context;
    lifecycle_context.event_id = "event:search-pipeline";
    lifecycle_context.idempotency_key = "idempotency:search-pipeline";
    lifecycle_context.source_id = "source:search-pipeline";
    lifecycle_context.scope.namespace_id = "local";
    lifecycle_context.scope.project_id = "agent-tests";
    lifecycle_context.scope.session_id = "session-1";
    lifecycle_context.content_hash = "sha256:search-pipeline";
    lifecycle_context.created_at = "2026-09-15T00:00:00Z";
    CHECK(common_flydelta_append_search_pipeline_lifecycle(
        lifecycle, lifecycle_context, make_fixture(), result,
        "flydelta://sideband/search-pipeline", error));
    auto records = lifecycle.list(error);
    CHECK(error.empty() && records.size() == 12);
    CHECK(records.front().kind == common_learning_lifecycle_kind::flydelta_result);
    CHECK(records.front().payload_json.find("direction-layer-scale") != std::string::npos);
    CHECK(records.front().payload_json.find("artifact_status") != std::string::npos);

    // The normal pipeline path is the bounded layer x scale region search.
    // It retains all executed arms and still selects only host-verified HELPED.
    auto region_config = config;
    region_config.use_intervention_region_search = true;
    region_config.region_max_singleton_layers = 2;
    region_config.region_max_neighborhoods = 2;
    region_config.region_max_trials = 32;
    common_flydelta_search_pipeline_result region_result;
    calls = 0;
    CHECK(common_flydelta_run_search_pipeline(
        make_fixture(), region_config, {input},
        [&](const common_flydelta_experiment_fixture &,
                const common_flydelta_direction_candidate &,
                const common_flydelta_layer_candidate * layer,
                float scale, bool apply_overlay,
                common_flydelta_counterfactual_trial & trial,
                common_flydelta_decision_margin & margin,
                common_flydelta_scale_geometry & geometry,
                std::string &) {
            ++calls;
            trial = {};
            trial.executed = true;
            trial.verifier_known = true;
            trial.overlay_applied = apply_overlay;
            trial.evidence_ref = "evidence:search-pipeline-region";
            const bool pair = layer != nullptr && layer->layer_indices ==
                std::vector<uint32_t>{2, 3};
            trial.passed = apply_overlay && pair && scale >= 0.08f;
            trial.quality = trial.passed ? 1.0f : 0.0f;
            geometry = {};
            geometry.available = apply_overlay;
            geometry.cosine = 0.85f;
            geometry.progress = scale;
            geometry.leakage = 0.05f;
            geometry.shift_norm = scale;
            margin.available = apply_overlay;
            margin.positive_total_logprob = scale;
            margin.negative_total_logprob = 0.0f;
            margin.positive_token_count = 1;
            margin.negative_token_count = 1;
            return true;
        }, region_result, error));
    CHECK(calls == 17); // baseline + 2×4 singleton + 2×4 adjacent pairs
    CHECK(region_result.directions.size() == 1);
    CHECK(region_result.directions.front().layer_results.empty());
    CHECK(region_result.directions.front().region_trials.size() == 16);
    CHECK(region_result.directions.front().region_selection.selected);
    CHECK(region_result.selection.selected && region_result.selection.intervention_region);
    CHECK(region_result.selection.region_trial_index ==
        region_result.directions.front().region_selection.trial_index);
    CHECK(region_result.directions.front().region_trials[
        region_result.selection.region_trial_index].candidate.layer_indices ==
        std::vector<uint32_t>({2, 3}));

    common_learning_in_memory_lifecycle_store region_lifecycle;
    CHECK(common_flydelta_append_search_pipeline_lifecycle(
        region_lifecycle, lifecycle_context, make_fixture(), region_result,
        "flydelta://sideband/search-pipeline-region", error));
    auto region_records = region_lifecycle.list(error);
    CHECK(error.empty() && region_records.size() == 16);
    CHECK(region_records.front().payload_json.find("direction-layer-scale-region") !=
        std::string::npos);

    // A diagnostic-only direction still produces a bounded plan and retains
    // UNKNOWN/NEUTRAL scale trials, but cannot populate the HELPED selection.
    result = {};
    calls = 0;
    input.layer_diagnostics[1].progress = 0.0f;
    CHECK(common_flydelta_run_search_pipeline(
        make_fixture(), config, {input},
        [&](const common_flydelta_experiment_fixture &,
                const common_flydelta_direction_candidate &,
                const common_flydelta_layer_candidate *, float scale, bool apply_overlay,
                common_flydelta_counterfactual_trial & trial,
                common_flydelta_decision_margin &,
                common_flydelta_scale_geometry & geometry, std::string &) {
            ++calls;
            trial = {};
            trial.executed = true;
            trial.verifier_known = false;
            trial.overlay_applied = apply_overlay;
            geometry = {};
            geometry.available = apply_overlay;
            geometry.cosine = 0.8f;
            geometry.progress = scale;
            geometry.leakage = 0.1f;
            geometry.shift_norm = scale;
            return true;
        }, result, error));
    CHECK(!result.selection.selected);
    CHECK(!result.directions.front().layer_results.empty());
    CHECK(!result.directions.front().layer_results.front().scale_selection.selected);
    return 0;
}
