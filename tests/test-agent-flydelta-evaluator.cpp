#include "agent/adaptation/flydelta/flydelta-evaluator.h"
#include "agent/adaptation/flydelta/flydelta-bootstrap-zoom-state-store.h"
#include "agent/adaptation/flydelta/flydelta-representation-augmentation-state-store.h"

#include <cmath>
#include <utility>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

static common_flydelta_experiment_job base_job(
        common_flydelta_experiment_job_kind kind, const std::string & id) {
    common_flydelta_experiment_job job;
    job.id = id;
    job.kind = kind;
    job.seed.id = "evidence://flydelta/evaluator";
    job.seed.behavior_key = "tool_use/diagnostics/missing-argument";
    job.seed.source = common_adaptation_evidence_source::tool_repair;
    job.seed.scope.namespace_id = "local";
    job.seed.scope.project_id = "project";
    job.seed.scope.session_id = "session";
    job.seed.task_fingerprint = "sha256:task";
    job.seed.model_profile_fingerprint = "sha256:model";
    job.seed.tokenizer_fingerprint = "sha256:tokenizer";
    job.seed.template_fingerprint = "sha256:template";
    job.seed.execution_context_fingerprint = "sha256:execution-context";
    job.seed.baseline_ref = "execution:failed";
    job.seed.candidate_ref = "execution:repaired";
    job.seed.verifier_ref = "verifier:v1";
    job.seed.evidence_ref = "evidence://flydelta/evaluator";
    job.seed.transaction_ids = {"learning://failed", "learning://repaired"};
    job.learning_rate = 0.5f;
    job.decay = 1.0f;
    job.code_revision = "evaluator-test:v1";
    return job;
}

static common_flydelta_counterfactual_report report(const std::string & id) {
    common_flydelta_counterfactual_report value;
    value.experiment_id = id;
    value.fixture_id = "flydelta://fixture/evaluator";
    value.candidate_id = "flydelta://candidate/evaluator";
    value.baseline_profile_id = "base";
    value.candidate_profile_id = "overlay";
    value.baseline.executed = true;
    value.baseline.verifier_known = true;
    value.baseline.passed = false;
    value.baseline.evidence_ref = "evidence:baseline";
    value.candidate.executed = true;
    value.candidate.verifier_known = true;
    value.candidate.passed = true;
    value.candidate.overlay_applied = true;
    value.candidate.evidence_ref = "evidence:candidate";
    value.outcome = common_flydelta_counterfactual_outcome::helped;
    value.quality_delta = 1.0f;
    return value;
}

static common_flydelta_behavior_delta behavior_delta() {
    common_flydelta_behavior_delta value;
    value.id = "flydelta://behavior/evaluator";
    value.source = common_adaptation_evidence_source::tool_repair;
    value.behavior_key = "tool_use/diagnostics/missing-argument";
    value.capture_manifest_id = "flydelta://capture/evaluator";
    value.host_evidence_ref = "evidence://repair/evaluator";
    value.model_profile_fingerprint = "sha256:model";
    value.execution_context_fingerprint = "sha256:execution-context";
    value.capture_layout_revision = "layout:v1";
    value.layer_index = 4;
    value.values = {1.0f, 0.0f, 0.0f};
    return value;
}

static common_flydelta_training_example training_example(
        common_flydelta_training_split split) {
    common_flydelta_training_example value;
    value.id = "flydelta://training/evaluator";
    value.behavior_key = "tool_use/diagnostics/missing-argument";
    value.context_fingerprint = "sha256:context";
    value.basis_revision = "flydelta://basis/evaluator";
    value.evidence_ref = "evidence://training/evaluator";
    value.split = split;
    value.context.expansion_dim = 8;
    value.context.indices = {1};
    value.context.values = {1.0f};
    value.target_coefficients = {0.25f, -0.1f};
    value.outcome = common_flydelta_counterfactual_outcome::helped;
    value.confidence = 1.0f;
    return value;
}

static common_flydelta_search_pipeline_result search_pipeline_result() {
    common_flydelta_search_pipeline_result value;
    common_flydelta_search_pipeline_direction_result direction;
    direction.direction.kind = common_flydelta_direction_kind::token_margin_direction;
    direction.direction.layer_index = 2;
    direction.direction.values = {1.0f, 0.0f};
    direction.direction.source_samples = 1;
    direction.direction.retained_samples = 1;
    direction.direction.median_alignment = 1.0f;
    common_flydelta_intervention_region_trial trial;
    trial.candidate.layer_indices = {24};
    trial.candidate.anchor_layer_index = 24;
    trial.candidate.total_scale = 0.1f;
    trial.candidate.per_layer_scale = 0.1f;
    trial.executed = true;
    trial.verifier_known = true;
    trial.search_score = 0.5f;
    trial.promising = true;
    trial.safe_to_continue = true;
    trial.evidence_ref = "evidence:where";
    direction.region_trials.push_back(std::move(trial));
    value.directions.push_back(std::move(direction));
    return value;
}

int main() {
    std::string error;
    common_flydelta_evaluator_config config;
    config.basis.dimension = 3;
    config.basis.max_directions = 4;
    config.basis.source = common_adaptation_evidence_source::tool_repair;
    config.basis.behavior_key = "tool_use/diagnostics/missing-argument";
    config.basis.model_profile_fingerprint = "sha256:model";
    config.basis.capture_layout_revision = "layout:v1";
    config.basis.execution_context_fingerprint = "sha256:execution-context";
    config.direction.dimension = 3;
    config.direction.layer_index = 4;
    config.direction.min_samples = 1;
    config.direction.max_samples = 4;
    config.direction.source = common_adaptation_evidence_source::tool_repair;
    config.direction.behavior_key = "tool_use/diagnostics/missing-argument";
    config.direction.model_profile_fingerprint = "sha256:model";
    config.direction.execution_context_fingerprint = "sha256:execution-context";
    config.direction.capture_layout_revision = "layout:v1";
    config.pipeline.dimension = 2;
    config.pipeline.layer.max_regions = 1;
    config.pipeline.layer.max_singletons = 1;
    config.pipeline.layer.max_neighborhoods = 0;
    config.pipeline.layer.max_candidates = 1;
    config.memory = {8, 2, 1.0f};

    common_flydelta_evaluator_result result;
    auto counterfactual = base_job(
            common_flydelta_experiment_job_kind::counterfactual,
            "flydelta://job/counterfactual");
    counterfactual.capture_manifest_ids = {"flydelta://capture/evaluator"};
    counterfactual.alpha_search.candidates = {0.05f};
    counterfactual.alpha_search.max_candidates = 1;
    common_flydelta_evaluator_callbacks callbacks;
    callbacks.run_counterfactual = [&](const auto & job, auto & reports, std::string &) {
        reports.push_back(report(job.id));
        return true;
    };
    CHECK(common_flydelta_evaluate_job(counterfactual, config, callbacks, result, error));
    CHECK(result.processed_references == 1 && result.counterfactual_reports.size() == 1);

    auto basis = base_job(common_flydelta_experiment_job_kind::basis,
            "flydelta://job/basis");
    basis.behavior_delta_ids = {"flydelta://behavior/evaluator"};
    callbacks = {};
    callbacks.resolve_behavior_delta = [](const auto &, auto & delta, auto & credit, std::string &) {
        delta = behavior_delta();
        credit.experiment_id = "flydelta://experiment/evaluator";
        credit.candidate_id = "flydelta://candidate/evaluator";
        credit.fixture_id = "flydelta://fixture/evaluator";
        credit.outcome = common_flydelta_counterfactual_outcome::helped;
        credit.quality_delta = 1.0f;
        credit.eligible_for_learning = true;
        return true;
    };
    CHECK(common_flydelta_evaluate_job(basis, config, callbacks, result, error));
    CHECK(result.processed_references == 1 && result.basis_directions.size() == 1);

    auto direction = base_job(common_flydelta_experiment_job_kind::direction,
            "flydelta://job/direction");
    direction.behavior_delta_ids = {"flydelta://behavior/evaluator"};
    CHECK(common_flydelta_evaluate_job(direction, config, callbacks, result, error));
    CHECK(result.processed_references == 1 && result.direction_candidates.size() == 3);
    CHECK(result.evidence_depth.depth == common_flydelta_search_depth::bootstrap);
    CHECK(result.search_budget.depth == common_flydelta_search_depth::bootstrap);
    CHECK(result.search_budget.max_region_trials == 4);
    CHECK(!result.search_budget.build_aggregate_directions &&
        !result.search_budget.allow_tfo_lite);

    // Queue execution can explicitly opt into experimental direction search
    // without weakening the strict learning-mode job above.
    auto experimental_direction = direction;
    experimental_direction.id = "flydelta://job/experimental-direction";
    config.direction.mode = common_flydelta_direction_search_mode::experimental;
    callbacks.resolve_behavior_delta = [](const auto &, auto & delta, auto & credit, std::string &) {
        delta = behavior_delta();
        delta.id = "flydelta://behavior/experimental";
        credit.experiment_id = "flydelta://experiment/experimental";
        credit.candidate_id = "flydelta://candidate/experimental";
        credit.fixture_id = "flydelta://fixture/experimental";
        credit.outcome = common_flydelta_counterfactual_outcome::unknown;
        credit.quality_delta = 0.0f;
        credit.eligible_for_learning = false;
        return true;
    };
    CHECK(common_flydelta_evaluate_job(
        experimental_direction, config, callbacks, result, error));
    CHECK(result.processed_references == 1 && result.direction_candidates.size() == 3);
    CHECK(result.direction_candidates.front().experimental_only);
    CHECK(result.direction_candidates[1].experimental_only &&
        result.direction_candidates[2].experimental_only);

    auto pipeline = base_job(common_flydelta_experiment_job_kind::search_pipeline,
            "flydelta://job/search-pipeline");
    pipeline.capture_manifest_ids = {"flydelta://capture/evaluator"};
    pipeline.behavior_delta_ids = {"flydelta://behavior/evaluator"};
    pipeline.alpha_search.candidates = {0.02f};
    pipeline.alpha_search.max_candidates = 1;
    callbacks = {};
    callbacks.resolve_behavior_delta = [](const auto &, auto & delta, auto & credit, std::string &) {
        delta = behavior_delta();
        credit.experiment_id = "flydelta://experiment/search";
        credit.candidate_id = "flydelta://candidate/search";
        credit.fixture_id = "flydelta://fixture/search";
        credit.outcome = common_flydelta_counterfactual_outcome::unknown;
        credit.quality_delta = 0.0f;
        credit.eligible_for_learning = false;
        return true;
    };
    callbacks.run_search_pipeline = [](const auto &, auto & value, std::string &) {
        value = search_pipeline_result();
        return true;
    };
    CHECK(common_flydelta_evaluate_job(pipeline, config, callbacks, result, error));
    CHECK(result.processed_references == 1 && result.search_pipeline_results.size() == 1);
    CHECK(result.search_continuations.size() == 1);
    CHECK(result.search_continuations.front().region.anchor_layer_index == 24);
    CHECK(result.has_experiment_plan &&
        result.experiment_plan.phase == common_flydelta_experiment_phase::bootstrap &&
        result.experiment_plan.depth == common_flydelta_search_depth::bootstrap &&
        !result.experiment_plan.run_tfo_lite);

    auto resumable_pipeline = pipeline;
    resumable_pipeline.id = "flydelta://job/search-pipeline-resume";
    resumable_pipeline.bootstrap_zoom_state_ref = "flydelta://state/bootstrap-1";
    callbacks = {};
    bool resumed = false;
    bool persisted = false;
    callbacks.resolve_bootstrap_zoom_state = [&](const auto & ref, auto & state, std::string &) {
        if (ref != "flydelta://state/bootstrap-1") return false;
        state.state_ref = ref;
        state.behavior_key = resumable_pipeline.seed.behavior_key;
        state.model_profile_fingerprint = resumable_pipeline.seed.model_profile_fingerprint;
        state.capture_layout_revision = "layout:v1";
        state.anchor_layer = 24;
        state.selected_scale = 0.1f;
        state.extra_model_trials = 3;
        state.next_candidate_index = 2;
        return true;
    };
    callbacks.run_search_pipeline_with_state = [&](const auto &, const auto * state,
            auto & value, auto & next_state, std::string &) {
        resumed = state != nullptr && state->next_candidate_index == 2;
        value = search_pipeline_result();
        next_state.state_ref.clear();
        next_state.behavior_key = resumable_pipeline.seed.behavior_key;
        next_state.model_profile_fingerprint = resumable_pipeline.seed.model_profile_fingerprint;
        next_state.capture_layout_revision = "layout:v1";
        next_state.anchor_layer = 24;
        next_state.selected_scale = 0.15f;
        next_state.best_margin_delta = 0.2f;
        next_state.best_search_score = 0.7f;
        next_state.extra_model_trials = 4;
        next_state.next_candidate_index = 4;
        return true;
    };
    callbacks.persist_bootstrap_zoom_state = [&](const auto & state, auto & ref, std::string &) {
        persisted = state.next_candidate_index == 4;
        ref = "flydelta://state/bootstrap-2";
        return true;
    };
    CHECK(common_flydelta_evaluate_job(resumable_pipeline, config, callbacks, result, error));
    CHECK(resumed && persisted && result.has_bootstrap_zoom_state &&
        result.bootstrap_zoom_state_ref == "flydelta://state/bootstrap-2" &&
        result.bootstrap_zoom_state.state_ref == "flydelta://state/bootstrap-2" &&
        result.bootstrap_zoom_state.next_candidate_index == 4);

    // A configured post-Bootstrap callback must not hijack the initial
    // Bootstrap/Whirlpool evaluation. It is selected only after the host has
    // attached the separate opaque post-Bootstrap state reference.
    auto initial_with_both_runners = pipeline;
    initial_with_both_runners.id = "flydelta://job/search-pipeline-initial-runner-order";
    callbacks = {};
    callbacks.resolve_behavior_delta = [](const auto &, auto & delta, auto & credit, std::string &) {
        delta = behavior_delta();
        credit.experiment_id = "flydelta://experiment/search-order";
        credit.candidate_id = "flydelta://candidate/search-order";
        credit.fixture_id = "flydelta://fixture/search-order";
        credit.outcome = common_flydelta_counterfactual_outcome::unknown;
        credit.eligible_for_learning = false;
        return true;
    };
    bool initial_runner_called = false;
    bool post_bootstrap_runner_called = false;
    callbacks.run_search_pipeline = [&](const auto &, auto & value, std::string &) {
        initial_runner_called = true;
        value = search_pipeline_result();
        return true;
    };
    callbacks.run_search_pipeline_with_search_state =
        [&](const auto &, const auto &, auto &, auto &, std::string &) {
            post_bootstrap_runner_called = true;
            return false;
        };
    CHECK(common_flydelta_evaluate_job(
        initial_with_both_runners, config, callbacks, result, error));
    CHECK(initial_runner_called && !post_bootstrap_runner_called);

    auto generic_resumable_pipeline = pipeline;
    generic_resumable_pipeline.id = "flydelta://job/search-pipeline-generic-resume";
    generic_resumable_pipeline.search_state_ref = "flydelta://state/search-1";
    callbacks = {};
    bool generic_resumed = false;
    callbacks.run_search_pipeline_with_search_state = [&](const auto & job,
            const auto & state_ref, auto & value, auto & next_state_ref, std::string &) {
        generic_resumed = job.search_state_ref == state_ref &&
            state_ref == "flydelta://state/search-1";
        value = search_pipeline_result();
        next_state_ref = "flydelta://state/search-2";
        return true;
    };
    bool typed_bootstrap_called_for_post_state = false;
    callbacks.run_search_pipeline_with_state = [&](const auto &, const auto *, auto &, auto &, std::string &) {
        typed_bootstrap_called_for_post_state = true;
        return false;
    };
    CHECK(common_flydelta_evaluate_job(
        generic_resumable_pipeline, config, callbacks, result, error));
    CHECK(generic_resumed && !typed_bootstrap_called_for_post_state &&
        result.search_state_ref == "flydelta://state/search-2" &&
        !result.has_experiment_plan &&
        result.search_pipeline_results.size() == 1);

    // The production-facing adapter persists only immutable state metadata in
    // the host lifecycle store. It can be attached without replacing the
    // host's model runner callback.
    common_learning_in_memory_lifecycle_store lifecycle_store;
    common_flydelta_bootstrap_zoom_lifecycle_context lifecycle_context;
    lifecycle_context.project_id = "project";
    lifecycle_context.session_id = "session";
    lifecycle_context.source_id = "host://agent";
    lifecycle_context.created_at = "2026-09-16T00:00:00Z";
    common_flydelta_evaluator_callbacks lifecycle_callbacks;
    CHECK(common_flydelta_configure_bootstrap_zoom_lifecycle_callbacks(
        lifecycle_store, lifecycle_context, lifecycle_callbacks, error));
    common_flydelta_bootstrap_zoom_state persisted_state;
    persisted_state.behavior_key = pipeline.seed.behavior_key;
    persisted_state.model_profile_fingerprint = pipeline.seed.model_profile_fingerprint;
    persisted_state.capture_layout_revision = "layout:v1";
    persisted_state.anchor_layer = 24;
    persisted_state.selected_scale = 0.1f;
    persisted_state.local_layers = {23, 24, 25};
    common_flydelta_bootstrap_zoom_candidate selected_arm;
    selected_arm.phase = common_flydelta_bootstrap_zoom_phase::profile_zoom;
    selected_arm.layer_indices = {24, 25};
    selected_arm.layer_weights = {0.70710678f, 0.70710678f};
    selected_arm.total_scale = 0.1f;
    common_flydelta_bootstrap_zoom_trial selected_trial;
    selected_trial.candidate = selected_arm;
    selected_trial.host_verified = true;
    selected_trial.margin_available = true;
    selected_trial.margin_delta = 0.2f;
    selected_trial.diagnostics_available = true;
    selected_trial.diagnostics = {1, 25, 0.8f, 0.2f, 0.1f, 0.2f};
    persisted_state.completed_trials = {selected_trial};
    CHECK(common_flydelta_select_bootstrap_zoom_trial(
        persisted_state.completed_trials, persisted_state.selection, error));
    std::string lifecycle_ref;
    CHECK(lifecycle_callbacks.persist_bootstrap_zoom_state(
        persisted_state, lifecycle_ref, error));
    CHECK(!lifecycle_ref.empty());
    common_flydelta_bootstrap_zoom_state resolved_state;
    CHECK(lifecycle_callbacks.resolve_bootstrap_zoom_state(
        lifecycle_ref, resolved_state, error));
    CHECK(resolved_state.state_ref == lifecycle_ref);
    CHECK(resolved_state.anchor_layer == 24);
    CHECK(resolved_state.local_layers == std::vector<uint32_t>({23, 24, 25}));
    CHECK(resolved_state.selection.selected &&
        resolved_state.completed_trials.size() == 1 &&
        resolved_state.completed_trials.front().candidate.layer_indices ==
            std::vector<uint32_t>({24, 25}));

    // A post-Bootstrap augmentation state uses the same opaque search-state
    // transport, but its typed state remains separately validated and
    // resumable. It must not fall through to the generic search callback.
    common_flydelta_representation_augmentation_state augmentation_state;
    augmentation_state.state_ref = "flydelta://state/representation-augmentation/evaluator";
    augmentation_state.model_fingerprint = pipeline.seed.model_profile_fingerprint;
    augmentation_state.behavior_key = pipeline.seed.behavior_key;
    augmentation_state.direction_family_id = "tool-choice";
    augmentation_state.parent_surface_revision = 1;
    augmentation_state.parent_search_state_ref = "flydelta://state/search/plateau";
    augmentation_state.parent_evidence_rank = 1.0f;
    augmentation_state.evidence_rank = 1.0f;
    augmentation_state.search_rank = 1;
    augmentation_state.selected_region = {24};
    augmentation_state.target_fixture_ref = "fixture://augmentation";
    augmentation_state.remaining_budget = 4;
    augmentation_state.surface_revision = 1;
    CHECK(common_flydelta_representation_augmentation_state_validate(
        augmentation_state, config.representation_augmentation, error));
    auto augmentation_job = pipeline;
    augmentation_job.id = "flydelta://job/search-pipeline-augmentation";
    augmentation_job.search_state_ref = augmentation_state.state_ref;
    callbacks = {};
    bool augmentation_runner_called = false;
    callbacks.resolve_representation_augmentation_state =
        [&](const auto & state_ref, auto & value, std::string &) {
            if (state_ref != augmentation_state.state_ref) return false;
            value = augmentation_state;
            return true;
        };
    callbacks.run_representation_augmentation_with_state =
        [&](const auto &, const auto * resume, auto & value, auto & next, std::string &) {
            augmentation_runner_called = resume != nullptr;
            value = search_pipeline_result();
            next = augmentation_state;
            next.phase = common_flydelta_representation_augmentation_phase::run_controls;
            next.surface_revision = 2;
            next.next_action = "run_controls";
            return true;
        };
    callbacks.persist_representation_augmentation_state =
        [](const auto & value, auto & ref, std::string &) {
            ref = value.state_ref;
            return true;
        };
    callbacks.run_search_pipeline_with_search_state =
        [](const auto &, const auto &, auto &, auto &, std::string &) { return false; };
    CHECK(common_flydelta_evaluate_job(augmentation_job, config, callbacks, result, error));
    CHECK(augmentation_runner_called && result.has_representation_augmentation_state &&
        result.representation_augmentation_state_ref == augmentation_state.state_ref &&
        result.representation_augmentation_state.surface_revision == 2 &&
        result.search_state_ref == augmentation_state.state_ref);

    auto memory = base_job(common_flydelta_experiment_job_kind::delta_memory,
            "flydelta://job/memory");
    memory.training_example_ids = {"flydelta://training/evaluator"};
    callbacks = {};
    callbacks.resolve_training_example = [](const auto &, auto & example, std::string &) {
        example = training_example(common_flydelta_training_split::train);
        return true;
    };
    CHECK(common_flydelta_evaluate_job(memory, config, callbacks, result, error));
    CHECK(result.processed_references == 1 && !result.delta_memory_weights.empty());

    callbacks.resolve_training_example = [](const auto &, auto & example, std::string &) {
        example = training_example(common_flydelta_training_split::holdout);
        return true;
    };
    CHECK(!common_flydelta_evaluate_job(memory, config, callbacks, result, error));
    CHECK(error.find("validation or holdout") != std::string::npos);
    return 0;
}
