#include "agent/adaptation/flydelta/flydelta-deep-search.h"
#include "agent/adaptation/flydelta/experiment/flydelta-outcome-guidance.h"
#include "agent/adaptation/flydelta/experiment/flydelta-coefficient-execution-internal.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#define CHECK(condition) do { if (!(condition)) { std::cerr << "failed at line " << __LINE__ << ": " << #condition << '\n'; return 1; } } while (false)

int main() {
    std::string error;
    common_flydelta_experiment_fixture fixture;
    fixture.id = "fixture:progress-anchor";
    fixture.task_fingerprint = "task";
    fixture.model_profile_fingerprint = "model";
    fixture.tokenizer_fingerprint = "tokenizer";
    fixture.template_fingerprint = "template";
    fixture.execution_context_fingerprint = "execution";
    fixture.verifier_revision = "verifier";
    common_flydelta_direction_candidate direction;
    direction.kind = common_flydelta_direction_kind::raw_repair;
    direction.layer_index = 2;
    direction.values = {1.0f, 0.0f};
    direction.source_samples = direction.retained_samples = 2;
    direction.median_alignment = 0.4f;
    direction.experimental_only = true;
    auto second = direction;
    second.kind = common_flydelta_direction_kind::diagonal_whitened_mean;
    second.values = {0.0f, 1.0f};
    const std::vector<common_flydelta_deep_search_direction> directions = {
        {direction, true, 0.8f}, {second, true, 0.6f}};
    common_flydelta_deep_search_config config;
    config.max_rank = 2; config.max_directions = 2; config.full_generation_top_k = 2;
    config.coefficients.max_candidates = 8; config.coefficients.step = 0.1f;
    config.coefficients.max_l2_norm = 0.32f;
    size_t wave = 1, arm = 0;
    const common_flydelta_coefficient_search_runner diagnostic =
        [](const auto &, const auto &, const std::vector<float> & coefficients, bool overlay,
           auto & trial, auto & margin, auto & geometry, bool & has_geometry, std::string &) {
            trial = {}; trial.executed = true; trial.overlay_applied = overlay;
            margin = {}; margin.available = overlay;
            margin.positive_total_logprob = coefficients.empty() ? 0.0f : coefficients[0];
            margin.positive_token_count = 1; margin.negative_token_count = 1;
            geometry = {}; has_geometry = false; return true;
        };
    const common_flydelta_coefficient_search_runner full =
        [&](const auto &, const auto &, const std::vector<float> &, bool overlay,
            auto & trial, auto & margin, auto & geometry, bool & has_geometry, std::string &) {
            trial = {}; margin = {}; geometry = {}; has_geometry = false;
            trial.executed = trial.host_evaluated = trial.verifier_known = true;
            trial.overlay_applied = overlay;
            trial.arm_id = overlay ? "arm:" + std::to_string(++arm) : "baseline";
            trial.evidence_ref = "evidence:" + trial.arm_id;
            auto & observation = trial.semantic_progress_observation;
            observation.available = observation.regression_coverage_complete = true;
            auto & objective = observation.objective;
            objective.objective_id = "feature"; objective.objective_revision = "goals:v1";
            objective.verifier_ref = "oracle:v1";
            objective.assessment_known = objective.regression_known = true;
            objective.satisfied_goals = !overlay ? std::vector<std::string>{"A"} : wave == 1
                ? std::vector<std::string>{"A", "B"} : std::vector<std::string>{"A", "B", "C"};
            objective.remaining_goals = !overlay ? std::vector<std::string>{"B", "C", "D"} : wave == 1
                ? std::vector<std::string>{"C", "D"} : std::vector<std::string>{"D"};
            return true;
        };
    const auto batch = [](const common_flydelta_coefficient_search_runner & scalar) {
        return [scalar](const auto & f, const auto & basis, const auto & coefficients,
                auto & trials, auto & margins, auto & geometries, auto & available, std::string & e) {
            trials.clear(); margins.clear(); geometries.clear(); available.clear();
            for (const auto & c : coefficients) {
                common_flydelta_counterfactual_trial trial;
                common_flydelta_decision_margin margin;
                common_flydelta_representation_diagnostics geometry;
                bool has_geometry = false;
                if (!scalar(f, basis, c, true, trial, margin, geometry, has_geometry, e)) return false;
                trials.push_back(std::move(trial)); margins.push_back(std::move(margin));
                geometries.push_back(std::move(geometry)); available.push_back(has_geometry);
            }
            return true;
        };
    };
    common_flydelta_deep_search_result initial;
    CHECK(common_flydelta_run_deep_search_batched(fixture, config, directions,
        diagnostic, batch(diagnostic), full, batch(full), initial, error));
    CHECK(initial.provisional_selection.selected && !initial.coefficient_selection.selected);
    const auto anchor = initial.coefficient_trials.at(initial.provisional_selection.trial_index);
    common_flydelta_progress_evidence progress;
    CHECK(common_flydelta_coefficient_trial_progress_evidence(anchor, progress, error));
    CHECK(common_flydelta_progress_evidence_classify(progress) == common_flydelta_progress_transition::improved);
    config.resume_anchor = true;
    config.anchor_ref = "resource:anchor:1";
    config.anchor_basis = initial.basis;
    config.anchor_observation = anchor.semantic_progress_observation;
    config.coefficients.anchor_basis_identity = common_flydelta_coefficient_basis_identity(initial.basis);
    config.coefficients.anchor_coefficients = anchor.coefficients;
    std::vector<float> physical;
    CHECK(common_flydelta_reconstruct_intervention(config.anchor_basis,
        config.coefficients.anchor_basis_identity, config.coefficients.anchor_coefficients, physical, error));
    CHECK(!physical.empty());
    common_flydelta_outcome_guidance_context context;
    context.scope_fingerprint = "scope"; context.model_profile_fingerprint = "model";
    context.tokenizer_fingerprint = "tokenizer"; context.template_fingerprint = "template";
    context.baseline_deployment_fingerprint = "base"; context.verifier_revision = "oracle:v1";
    context.semantic_domain = "feature"; context.strategy = "deep"; context.estimator = "host-goals";
    context.local_task_fingerprint = "task"; context.local_lineage_ref = "lineage";
    context.progress_baseline = progress.baseline;
    context.local_parent = anchor.semantic_progress_observation.objective;
    context.local_parent_ref = config.anchor_ref;
    auto candidate_config = config.coefficients;
    candidate_config.guidance_basis_identity = common_flydelta_coefficient_detail::guidance_basis_identity(config.anchor_basis);
    candidate_config.guidance_layer_index = config.anchor_basis.layer_index;
    const auto anchor_candidate = common_flydelta_coefficient_detail::guidance_candidate(
        anchor.coefficients, config.anchor_basis, candidate_config, &anchor.margin_comparison);
    common_flydelta_outcome_guidance_observation observed;
    observed.stable_observation_id = anchor.arm_id; observed.fixture_group_id = "group";
    observed.basis_identity = anchor_candidate.basis_identity;
    observed.direction_identity = anchor_candidate.direction_identity; observed.dose_identity = anchor_candidate.dose_identity;
    observed.context = context; observed.rank = anchor_candidate.rank;
    observed.relative_layer = anchor_candidate.relative_layer; observed.relative_dose = anchor_candidate.relative_dose;
    observed.mixed_coefficients = anchor_candidate.mixed_coefficients;
    observed.diagnostic_margin_sign = anchor_candidate.diagnostic_margin_sign;
    observed.comparison_completed = observed.overlay_applied = observed.verifier_known = true;
    observed.task_fingerprint = context.local_task_fingerprint; observed.lineage_ref = context.local_lineage_ref;
    observed.progress = progress;
    common_flydelta_outcome_guidance_config guidance;
    guidance.mode = common_flydelta_outcome_guidance_mode::experimental;
    bool frontier = false, matching_parent = false, experimental_applied = false, known_progress = false;
    bool posterior_selected = false;
    std::vector<std::string> selected;
    const common_flydelta_outcome_guidance_adviser adviser =
        [&](const auto & request, auto & decision, std::string & e) {
            auto current = context;
            current.progress_baseline = request.progress_baseline;
            current.local_parent = request.progress_parent;
            current.local_parent_ref = request.progress_parent_ref;
            if (!common_flydelta_outcome_guidance_build_decision(guidance, current, {observed},
                    request.candidates, request.selection_limit, 7, decision, e,
                    common_flydelta_outcome_guidance_policy::posterior_sampling,
                    common_flydelta_guidance_policy_start(0, 2, request.policy_sequence_start))) return false;
            decision.shadow_mode = guidance.mode == common_flydelta_outcome_guidance_mode::shadow;
            if (request.stage == "deep_full_generation_frontier") {
                frontier = true;
                posterior_selected = std::find(decision.selection_reasons.begin(),
                    decision.selection_reasons.begin() + request.selection_limit, "posterior_sample") !=
                    decision.selection_reasons.begin() + request.selection_limit;
                matching_parent = request.progress_parent && request.progress_parent_ref == config.anchor_ref;
                experimental_applied = !decision.shadow_mode && decision.fallback_reason.empty();
                known_progress = std::any_of(decision.predictions.begin(), decision.predictions.end(),
                    [](const auto & prediction) { return prediction.effective_progress_known > 0; });
                for (size_t i = 0; i < request.selection_limit; ++i)
                    selected.push_back(request.candidates.at(decision.selected_order.at(i)).arm_id);
            }
            return true;
        };
    wave = 2;
    common_flydelta_deep_search_result continued;
    CHECK(common_flydelta_run_deep_search_batched(fixture, config, directions,
        diagnostic, batch(diagnostic), full, batch(full), continued, error, adviser));
    CHECK(frontier);
    CHECK(matching_parent);
    CHECK(experimental_applied);
    CHECK(posterior_selected);
    CHECK(known_progress);
    CHECK(!selected.empty());
    CHECK(continued.provisional_selection.selected);
    const auto & next = continued.coefficient_trials.at(continued.provisional_selection.trial_index);
    common_flydelta_progress_evidence next_progress;
    CHECK(common_flydelta_coefficient_trial_progress_evidence(next, next_progress, error));
    CHECK(next_progress.parent && next_progress.parent_ref == config.anchor_ref);
    CHECK(common_flydelta_progress_evidence_classify(next_progress) == common_flydelta_progress_transition::improved);
    for (const auto & id : selected) {
        CHECK(std::any_of(continued.coefficient_trials.begin(), continued.coefficient_trials.end(),
            [&](const auto & trial) { return trial.mutation_kind == "full_generation_top_arm" &&
                trial.executed && trial.overlay_applied && trial.guidance_arm_id == id; }));
    }
    std::cout << "progress anchor and experimental adviser continuation verified with deterministic fixtures\n";
    return 0;
}
