#include "../agent-daemon-flydelta-internal.h"
#include "agent/adaptation/flydelta/flydelta-search-budget.h"

#include "agent/agent-residual-patch.h"
#include "agent/adaptation/flydelta/experiment/flydelta-guidance-qualification.h"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>

#include "state.h"

namespace agent_daemon_flydelta_internal {

bool daemon_flydelta_run_bootstrap_zoom_slice(
        std::shared_ptr<daemon_flydelta_resource_provider> provider,
        const common_flydelta_experiment_job & job,
        const common_flydelta_bootstrap_zoom_state & resume,
        common_flydelta_search_pipeline_result & output,
        common_flydelta_bootstrap_zoom_state & next,
        std::string & error) {
    error.clear();
    provider = daemon_flydelta_provider_for_scope(provider, job.seed.scope);
    common_flydelta_experiment_fixture fixture;
    if (!daemon_flydelta_fixture_from_job(job, fixture, error)) return false;
    std::vector<common_flydelta_basis_direction> directions;
    if (!daemon_flydelta_parse_directions(*provider, job.seed.candidate_ref, directions, error)) return false;

    common_flydelta_bootstrap_zoom_config zoom_config;
    std::vector<common_flydelta_bootstrap_zoom_candidate> candidates;
    if (resume.phase == common_flydelta_bootstrap_zoom_phase::alpha_zoom) {
        if (!common_flydelta_propose_bootstrap_alpha_zoom(
                resume.anchor_layer, resume.selected_scale, zoom_config, candidates, error)) return false;
    } else if (resume.phase == common_flydelta_bootstrap_zoom_phase::profile_zoom) {
        if (!common_flydelta_propose_bootstrap_profile_zoom(
                resume.local_layers, resume.anchor_layer, resume.selected_scale,
                zoom_config, candidates, error)) return false;
    } else {
        // Sign control is the final deterministic BootstrapZoom slice.  The
        // next orchestration decision, not this runner, decides whether a
        // later rank-one adaptive-alpha slice is warranted.
        common_flydelta_bootstrap_zoom_candidate candidate;
        candidate.phase = common_flydelta_bootstrap_zoom_phase::sign_control;
        candidate.layer_indices = {resume.anchor_layer};
        candidate.layer_weights = {-1.0f};
        candidate.total_scale = resume.selected_scale;
        candidate.opposite_sign_control = true;
        candidates.push_back(std::move(candidate));
    }
    if (candidates.empty()) {
        error = "FlyDelta BootstrapZoom produced no bounded candidates";
        return false;
    }
    const size_t remaining = resume.extra_model_trials >= zoom_config.max_extra_model_trials
        ? 0 : zoom_config.max_extra_model_trials - resume.extra_model_trials;
    if (remaining == 0) {
        error = "FlyDelta BootstrapZoom budget is exhausted";
        return false;
    }
    if (candidates.size() > remaining) candidates.resize(remaining);

    common_flydelta_arm_batch_request batch;
    batch.batch_id = job.id + ":bootstrap-zoom:" + std::to_string(resume.next_candidate_index);
    batch.wave_id = common_flydelta_bootstrap_zoom_phase_name(resume.phase);
    common_flydelta_arm_request baseline;
    baseline.job_id = job.id;
    baseline.wave_id = batch.wave_id;
    baseline.proposal_index = 0;
    baseline.arm_id = batch.batch_id + ":baseline";
    baseline.context_ref = job.seed.baseline_ref;
    baseline.fixture_ref = fixture.id;
    baseline.intervention_ref = job.seed.candidate_ref;
    baseline.batch_compatibility_key = baseline.context_ref + "\n" + baseline.fixture_ref;
    baseline.fresh_context = true;
    baseline.request_capture = true;
    baseline.request_teacher_forced_margin = true;
    baseline.request_generation = true;
    baseline.request_host_verification = true;
    baseline.max_generated_tokens = 0;
    for (const auto & direction : directions) baseline.layer_indices.push_back(
        static_cast<uint32_t>(direction.layer_index));
    baseline.coefficients.assign(baseline.layer_indices.size(), 0.0f);
    if (!common_flydelta_arm_request_validate(baseline, error)) return false;
    batch.arms.push_back(baseline);
    for (size_t index = 0; index < candidates.size(); ++index) {
        const auto & candidate = candidates[index];
        common_flydelta_arm_request arm;
        arm.job_id = job.id;
        arm.wave_id = batch.wave_id;
        arm.proposal_index = index + 1;
        arm.arm_id = batch.batch_id + ":" + std::to_string(index);
        arm.context_ref = job.seed.baseline_ref;
        arm.fixture_ref = fixture.id;
        arm.intervention_ref = job.seed.candidate_ref;
        arm.batch_compatibility_key = baseline.batch_compatibility_key;
        arm.apply_overlay = true;
        arm.fresh_context = true;
        arm.alpha = candidate.total_scale;
        arm.layer_indices = candidate.layer_indices;
        arm.coefficients = candidate.layer_weights;
        arm.request_capture = true;
        arm.request_teacher_forced_margin = true;
        arm.request_generation = true;
        arm.request_host_verification = true;
        arm.max_generated_tokens = 0;
        if (!common_flydelta_arm_request_validate(arm, error)) return false;
        batch.arms.push_back(std::move(arm));
    }
    common_flydelta_arm_batch_request diagnostic_batch = batch;
    for (auto & arm : diagnostic_batch.arms) {
        arm.request_generation = false;
        arm.request_host_verification = false;
        arm.max_generated_tokens = 0;
    }
    common_flydelta_arm_batch_result diagnostic_results;
    if (!daemon_flydelta_execute_batch(provider, diagnostic_batch, diagnostic_results, error) ||
            diagnostic_results.arms.size() != diagnostic_batch.arms.size()) {
        if (error.empty()) error = "FlyDelta BootstrapZoom diagnostic batch returned incomplete arms";
        return false;
    }

    // Diagnostics preserve the existing candidate order and ranking signal.
    // Only the bounded diagnostic frontier is re-run with generation and
    // host verification; this is an execution optimization, not a policy
    // or candidate-generation change.
    std::vector<size_t> frontier_indices;
    frontier_indices.reserve(candidates.size());
    for (size_t index = 0; index < candidates.size(); ++index) {
        frontier_indices.push_back(index);
    }
    std::stable_sort(frontier_indices.begin(), frontier_indices.end(),
        [&](const size_t left, const size_t right) {
            const auto & lhs = diagnostic_results.arms[left + 1];
            const auto & rhs = diagnostic_results.arms[right + 1];
            const float lhs_score = lhs.margin_available ? lhs.margin.normalized_delta() : 0.0f;
            const float rhs_score = rhs.margin_available ? rhs.margin.normalized_delta() : 0.0f;
            return lhs_score > rhs_score;
        });
    constexpr size_t full_generation_top_k = 2;
    if (frontier_indices.size() > full_generation_top_k) {
        frontier_indices.resize(full_generation_top_k);
    }
    common_flydelta_arm_batch_request full_batch;
    full_batch.batch_id = batch.batch_id + ":frontier";
    full_batch.wave_id = batch.wave_id;
    full_batch.arms.push_back(batch.arms.front());
    for (const size_t index : frontier_indices) full_batch.arms.push_back(batch.arms[index + 1]);
    common_flydelta_arm_batch_result full_results;
    if (!daemon_flydelta_execute_batch(provider, full_batch, full_results, error) ||
            full_results.arms.size() != full_batch.arms.size()) {
        if (error.empty()) error = "FlyDelta BootstrapZoom frontier batch returned incomplete arms";
        return false;
    }

    next = resume;
    next.state_ref.clear();
    std::vector<common_flydelta_bootstrap_zoom_trial> new_trials;
    common_flydelta_search_pipeline_direction_result direction_result;
    direction_result.direction.kind = common_flydelta_direction_kind::raw_repair;
    direction_result.direction.layer_index = directions.front().layer_index;
    direction_result.direction.values = directions.front().values;
    direction_result.direction.origin = "runtime_resource";
    direction_result.direction.extraction_id = job.seed.candidate_ref;
    direction_result.direction.source_samples = 1;
    direction_result.direction.retained_samples = 1;
    direction_result.direction.experimental_only = true;
    const auto & baseline_result = full_results.arms.front();
    if (!baseline_result.executed || baseline_result.arm_id != full_batch.arms.front().arm_id) {
        error = "FlyDelta BootstrapZoom baseline arm is incomplete";
        return false;
    }
    for (size_t index = 0; index < candidates.size(); ++index) {
        const auto & arm = diagnostic_batch.arms[index + 1];
        const auto frontier = std::find(frontier_indices.begin(), frontier_indices.end(), index);
        const bool full_execution = frontier != frontier_indices.end();
        const auto & arm_result = full_execution
            ? full_results.arms[1 + static_cast<size_t>(frontier - frontier_indices.begin())]
            : diagnostic_results.arms[index + 1];
        if (!arm_result.executed || arm_result.arm_id != batch.arms[index + 1].arm_id) {
            error = "FlyDelta BootstrapZoom candidate arm is incomplete";
            return false;
        }
        common_flydelta_representation_diagnostics diagnostics;
        if (!daemon_flydelta_diagnostics_for_arm(provider, job, diagnostic_batch.arms.front().arm_id, arm,
                candidates[index].layer_indices.front(), diagnostics, error)) return false;
        common_flydelta_bootstrap_zoom_trial trial;
        trial.candidate = candidates[index];
        trial.outcome = full_execution ? daemon_flydelta_classify_arm_pair(baseline_result, arm_result)
                                      : common_flydelta_counterfactual_outcome::unknown;
        trial.host_evaluated = full_execution && arm_result.host_evaluated;
        trial.verifier_known = full_execution && arm_result.verifier_known;
        trial.margin_available = arm_result.margin.available &&
            (!full_execution || baseline_result.margin.available);
        trial.margin_delta = trial.margin_available
            ? (full_execution
                ? arm_result.margin.normalized_delta() - baseline_result.margin.normalized_delta()
                : arm_result.margin.normalized_delta()) : 0.0f;
        if (full_execution) {
            trial.progress = common_flydelta_compare_semantic_progress(
                baseline_result.semantic_progress_observation,
                arm_result.semantic_progress_observation,
                daemon_flydelta_arm_semantic_passed(baseline_result),
                daemon_flydelta_arm_semantic_passed(arm_result));
            trial.progress_available = trial.progress.comparable;
        }
        trial.diagnostics_available = diagnostics.layer_index != 0;
        trial.diagnostics = diagnostics;
        if (!common_flydelta_bootstrap_zoom_trial_validate(trial, error)) return false;
        new_trials.push_back(trial);

        common_flydelta_intervention_region_trial region_trial;
        region_trial.candidate.layer_indices = candidates[index].layer_indices;
        region_trial.candidate.anchor_layer_index = candidates[index].layer_indices.front();
        region_trial.candidate.total_scale = candidates[index].total_scale;
        region_trial.candidate.per_layer_scale = candidates[index].total_scale /
            std::sqrt(static_cast<float>(candidates[index].layer_indices.size()));
        region_trial.requested_total_scale = candidates[index].total_scale;
        region_trial.executed_total_scale = candidates[index].total_scale;
        region_trial.outcome = trial.outcome;
        region_trial.quality_delta = full_execution
            ? arm_result.quality - baseline_result.quality : 0.0f;
        region_trial.margin = arm_result.margin;
        region_trial.margin_comparison.available = trial.margin_available;
        region_trial.margin_comparison.baseline = baseline_result.margin;
        region_trial.margin_comparison.candidate = arm_result.margin;
        region_trial.executed = arm_result.executed;
        region_trial.host_evaluated = full_execution && arm_result.host_evaluated;
        region_trial.verifier_known = full_execution && arm_result.verifier_known;
        if (full_execution) daemon_flydelta_record_arm_observations(region_trial, arm_result, &baseline_result);
        region_trial.geometry_available = trial.diagnostics_available;
        region_trial.geometry = diagnostics;
        region_trial.safe_to_continue = trial.outcome != common_flydelta_counterfactual_outcome::harmed &&
            arm_result.host_outcome != common_flydelta_counterfactual_outcome::harmed &&
            trial.diagnostics_available &&
            diagnostics.leakage <= 1.0f && diagnostics.shift_norm <= 1.0f;
        region_trial.promising = trial.margin_available && trial.margin_delta > 0.0f;
        region_trial.search_score = trial.margin_delta;
        region_trial.observed_decision_summary = arm_result.observed_decision_summary;
        region_trial.expected_decision_summary = arm_result.expected_decision_summary;
        region_trial.verifier_reason = arm_result.verifier_reason;
        region_trial.evidence_ref = full_execution ? arm_result.generation_ref : arm_result.capture_ref;
        direction_result.region_trials.push_back(std::move(region_trial));
    }
    next.completed_trials.insert(next.completed_trials.end(), new_trials.begin(), new_trials.end());
    next.extra_model_trials += new_trials.size();
    next.next_candidate_index += new_trials.size();
    // A diagnostics-first slice may legitimately reject every newly probed
    // arm. Preserve the safety gate, but report that bounded branch as
    // no-useful-utility rather than turning it into a worker failure. This is
    // the same terminal semantics used by the ordinary search pipeline.
    bool has_safe_trial = false;
    for (const auto & trial : next.completed_trials) {
        if (trial.outcome == common_flydelta_counterfactual_outcome::harmed) continue;
        const bool geometry_safe = !trial.diagnostics_available ||
            (trial.diagnostics.cosine >= 0.3f && trial.diagnostics.progress > 0.0f &&
             trial.diagnostics.leakage <= 1.0f && trial.diagnostics.shift_norm <= 1.0f);
        if (geometry_safe) {
            has_safe_trial = true;
            break;
        }
    }
    if (has_safe_trial && !common_flydelta_select_bootstrap_zoom_trial(
            next.completed_trials, next.selection, error)) return false;
    if (next.selection.selected) {
        const auto & selected = next.completed_trials[next.selection.trial_index];
        next.selected_scale = selected.candidate.total_scale;
        next.best_margin_delta = selected.margin_delta;
        next.best_search_score = next.selection.search_score;
    }
    if (next.progress_available && !common_flydelta_semantic_progress_is_safe_anchor(next.best_progress)) {
        next.progress_available = false;
        next.best_progress = {};
        next.best_experimental_candidate_ref.clear();
        next.best_progress_trial_available = false;
        next.best_progress_trial_index = 0;
    }
    for (size_t index = 0; index < next.completed_trials.size(); ++index) {
        const auto & trial = next.completed_trials[index];
        if (!trial.progress_available || !common_flydelta_semantic_progress_is_safe_anchor(trial.progress)) {
            continue;
        }
        if (!next.progress_available ||
                trial.progress.candidate_score > next.best_progress.candidate_score) {
            next.progress_available = true;
            next.best_progress = trial.progress;
            next.best_experimental_candidate_ref = job.seed.candidate_ref;
            next.best_progress_trial_available = true;
            next.best_progress_trial_index = index;
        } else if (!next.best_progress_trial_available &&
                trial.progress.candidate_score == next.best_progress.candidate_score &&
                trial.progress.baseline_score == next.best_progress.baseline_score &&
                trial.progress.outcome == next.best_progress.outcome &&
                trial.progress.improved_dimensions == next.best_progress.improved_dimensions &&
                trial.progress.residual_dimensions == next.best_progress.residual_dimensions) {
            // Reconcile older states only when their recorded progress matches
            // an actually persisted trial; never infer an arm from its score.
            next.best_progress_trial_available = true;
            next.best_progress_trial_index = index;
        }
    }
    if (next.progress_available) ++next.progress_iteration;
    if (resume.phase == common_flydelta_bootstrap_zoom_phase::alpha_zoom) {
        next.phase = common_flydelta_bootstrap_zoom_phase::profile_zoom;
    } else if (resume.phase == common_flydelta_bootstrap_zoom_phase::profile_zoom &&
            zoom_config.include_opposite_sign_control) {
        next.phase = common_flydelta_bootstrap_zoom_phase::sign_control;
    } else {
        next.phase = common_flydelta_bootstrap_zoom_phase::adaptive_alpha;
    }
    if (next.extra_model_trials >= zoom_config.max_extra_model_trials) {
        next.phase = common_flydelta_bootstrap_zoom_phase::adaptive_alpha;
    }
    if (next.phase == common_flydelta_bootstrap_zoom_phase::adaptive_alpha) {
        next.refinement_kind = common_flydelta_bootstrap_refinement_kind::adaptive_alpha;
    }
    output = {};
    for (size_t index = 0; index < direction_result.region_trials.size(); ++index) {
        const auto & trial = direction_result.region_trials[index];
        if (!trial.safe_to_continue || !trial.promising) continue;
        const bool helped = trial.outcome == common_flydelta_counterfactual_outcome::helped && trial.verifier_known;
        const bool selected_helped = output.selection.selected &&
            direction_result.region_trials[output.selection.region_trial_index].outcome ==
                common_flydelta_counterfactual_outcome::helped &&
            direction_result.region_trials[output.selection.region_trial_index].verifier_known;
        if (!output.selection.selected || (helped && !selected_helped) ||
                (helped == selected_helped && trial.search_score > output.selection.score)) {
            output.selection.selected = true;
            output.selection.intervention_region = true;
            output.selection.region_trial_index = index;
            output.selection.scale = trial.executed_total_scale;
            output.selection.score = trial.search_score;
        }
    }
    direction_result.region_selection.selected = output.selection.selected;
    direction_result.region_selection.trial_index = output.selection.region_trial_index;
    direction_result.region_selection.score = output.selection.score;
    output.directions.push_back(std::move(direction_result));
    // A diagnostics batch may contain trials without producing a safe,
    // promising selection. Preserve the terminal no-useful-utility status so
    // the continuation seam can stop the branch safely instead of treating
    // the absence of a selected arm as a callback failure.
    output.search_status = output.selection.selected
        ? common_flydelta_search_status::candidate_available
        : common_flydelta_search_status::no_useful_utility;
    return true;
}

bool daemon_flydelta_run_adaptive_alpha_slice(
        std::shared_ptr<daemon_flydelta_resource_provider> provider,
        const common_flydelta_experiment_job & job,
        const common_flydelta_bootstrap_zoom_state & resume,
        common_flydelta_search_pipeline_result & output,
        common_flydelta_bootstrap_zoom_state & next,
        std::string & error) {
    error.clear();
    provider = daemon_flydelta_provider_for_scope(provider, job.seed.scope);
    common_flydelta_experiment_fixture fixture;
    if (!daemon_flydelta_fixture_from_job(job, fixture, error)) return false;
    std::vector<common_flydelta_basis_direction> directions;
    if (!daemon_flydelta_parse_directions(*provider, job.seed.candidate_ref, directions, error)) {
        return false;
    }
    const auto direction = std::find_if(directions.begin(), directions.end(), [&](const auto & value) {
        return value.layer_index == static_cast<int32_t>(resume.anchor_layer);
    });
    if (direction == directions.end()) {
        error = "FlyDelta AdaptiveAlpha anchor direction is absent from the intervention resource";
        return false;
    }
    common_flydelta_alpha_response_search_config config;
    const float prior_scale = resume.alpha_response_available && resume.alpha_response.selected
        ? resume.alpha_response.scale : resume.selected_scale;
    config.seed_scale = std::max(0.0001f, std::min(config.max_scale, prior_scale));
    std::string baseline_arm_id;
    common_flydelta_decision_margin baseline_margin;
    size_t proposal_index = 0;
    const auto runner = [provider, &job, &fixture, layer = resume.anchor_layer,
            &baseline_arm_id, &baseline_margin, &proposal_index, config](const common_flydelta_experiment_fixture &,
            const float scale, const bool apply_overlay,
            common_flydelta_counterfactual_trial & trial,
            common_flydelta_decision_margin & margin,
            common_flydelta_representation_diagnostics & geometry,
            bool & geometry_available, std::string & runner_error) mutable {
        std::string arm_id;
        const size_t arm_proposal = proposal_index++;
        if (!daemon_flydelta_run_rank1_alpha_arm(provider, job, fixture, layer, scale,
                apply_overlay, arm_proposal, !apply_overlay, trial, margin, geometry,
                geometry_available, arm_id, runner_error)) return false;
        if (!apply_overlay) {
            baseline_arm_id = std::move(arm_id);
            baseline_margin = margin;
            return true;
        }
        if (baseline_arm_id.empty()) {
            runner_error = "FlyDelta AdaptiveAlpha candidate ran before its baseline";
            return false;
        }
        common_flydelta_arm_request diagnostic_arm;
        diagnostic_arm.arm_id = arm_id;
        diagnostic_arm.context_ref = job.seed.baseline_ref;
        diagnostic_arm.intervention_ref = job.seed.candidate_ref;
        if (!daemon_flydelta_diagnostics_for_arm(provider, job, baseline_arm_id,
                diagnostic_arm, layer, geometry, runner_error)) return false;
        geometry_available = geometry.layer_index != 0;
        const bool safe_geometry = !geometry_available ||
            (geometry.cosine >= config.min_cosine &&
             geometry.leakage <= config.max_leakage &&
             geometry.shift_norm <= config.max_shift_norm);
        if (!safe_geometry) return true;
        if (!daemon_flydelta_run_rank1_alpha_arm(provider, job, fixture, layer, scale,
                true, arm_proposal, true, trial, margin, geometry,
                geometry_available, arm_id, runner_error)) return false;
        if (!daemon_flydelta_diagnostics_for_arm(provider, job, baseline_arm_id,
                diagnostic_arm, layer, geometry, runner_error)) return false;
        geometry_available = geometry.layer_index != 0;
        return true;
    };
    std::vector<common_flydelta_alpha_response_trial> trials;
    common_flydelta_alpha_response_selection selection;
    if (!common_flydelta_run_alpha_response_search(
            fixture, config, runner, trials, selection, error)) return false;

    next = resume;
    next.state_ref.clear();
    next.phase = common_flydelta_bootstrap_zoom_phase::adaptive_alpha;
    next.refinement_kind = common_flydelta_bootstrap_refinement_kind::adaptive_alpha;
    next.alpha_response_available = true;
    next.alpha_response = selection;
    if (selection.selected) {
        next.selected_scale = selection.scale;
        next.best_search_score = selection.utility;
        if (selection.best_margin_available) {
            next.best_margin_delta = selection.best_margin_delta_normalized;
        }
    }

    common_flydelta_search_pipeline_direction_result direction_result;
    direction_result.direction.kind = common_flydelta_direction_kind::raw_repair;
    direction_result.direction.layer_index = direction->layer_index;
    direction_result.direction.values = direction->values;
    direction_result.direction.origin = "runtime_adaptive_alpha";
    direction_result.direction.extraction_id = job.seed.candidate_ref;
    direction_result.direction.source_samples = 1;
    direction_result.direction.retained_samples = 1;
    direction_result.direction.experimental_only = true;
    for (const auto & alpha_trial : trials) {
        common_flydelta_intervention_region_trial region_trial;
        region_trial.candidate.layer_indices = {resume.anchor_layer};
        region_trial.candidate.anchor_layer_index = resume.anchor_layer;
        region_trial.candidate.total_scale = alpha_trial.scale;
        region_trial.candidate.per_layer_scale = alpha_trial.scale;
        region_trial.requested_total_scale = alpha_trial.requested_scale;
        region_trial.executed_total_scale = alpha_trial.scale;
        region_trial.outcome = alpha_trial.outcome;
        region_trial.quality_delta = alpha_trial.counterfactual.quality;
        region_trial.margin = alpha_trial.margin;
        region_trial.margin_comparison.available = alpha_trial.margin_available;
        region_trial.margin_comparison.baseline = baseline_margin;
        region_trial.margin_comparison.candidate = alpha_trial.margin;
        region_trial.executed = alpha_trial.counterfactual.executed;
        region_trial.host_evaluated = alpha_trial.counterfactual.host_evaluated;
        region_trial.verifier_known = alpha_trial.counterfactual.verifier_known;
        if (alpha_trial.counterfactual.host_evaluated) region_trial.candidate_observation = alpha_trial.counterfactual;
        region_trial.geometry_available = alpha_trial.geometry_available;
        region_trial.geometry = alpha_trial.geometry;
        region_trial.observed_decision_summary = alpha_trial.counterfactual.observed_decision_summary;
        region_trial.expected_decision_summary = alpha_trial.counterfactual.expected_decision_summary;
        region_trial.verifier_reason = alpha_trial.counterfactual.verifier_reason;
        region_trial.safe_to_continue = alpha_trial.outcome !=
            common_flydelta_counterfactual_outcome::harmed &&
            alpha_trial.safe_to_continue;
        region_trial.promising = alpha_trial.safe_to_continue &&
            alpha_trial.utility > config.utility_epsilon;
        region_trial.search_score = alpha_trial.utility;
        region_trial.evidence_ref = alpha_trial.counterfactual.evidence_ref;
        if (!common_flydelta_intervention_region_trial_validate(region_trial, error)) return false;
        direction_result.region_trials.push_back(std::move(region_trial));
    }
    output = {};
    output.directions.push_back(std::move(direction_result));
    output.alpha_response_available = true;
    output.alpha_response = selection;
    output.search_status = selection.selected
        ? common_flydelta_search_status::candidate_available
        : common_flydelta_search_status::no_useful_utility;
    return true;
}

// Orthogonal remains a same-phase experimental escape.  The common layer
// derives the profile-space residual from persisted BootstrapZoom trials; the
// daemon owns only the fresh baseline/control wave that validates it.
bool daemon_flydelta_run_orthogonal_slice(
        std::shared_ptr<daemon_flydelta_resource_provider> provider,
        const common_flydelta_experiment_job & job,
        const common_flydelta_bootstrap_zoom_state & resume,
        common_flydelta_search_pipeline_result & output,
        common_flydelta_bootstrap_zoom_state & next,
        std::string & error) {
    error.clear();
    provider = daemon_flydelta_provider_for_scope(provider, job.seed.scope);
    common_flydelta_experiment_fixture fixture;
    if (!daemon_flydelta_fixture_from_job(job, fixture, error)) return false;
    common_flydelta_orthogonal_search_config config;
    common_flydelta_orthogonal_search_input input;
    if (!common_flydelta_prepare_orthogonal_search_input(config, resume, input, error)) {
        return false;
    }
    common_flydelta_orthogonal_search_result orthogonal;
    if (!common_flydelta_build_orthogonal_search_direction(
            config, input.rank1_intervention, input.arms, orthogonal, error)) return false;
    if (!orthogonal.available) {
        output = {};
        output.search_status = common_flydelta_search_status::no_useful_utility;
        next = resume;
        next.state_ref.clear();
        return true;
    }
    std::vector<common_flydelta_basis_direction> available;
    if (!daemon_flydelta_parse_directions(*provider, job.seed.candidate_ref, available, error)) {
        return false;
    }
    const auto normalize = [](std::vector<float> values) {
        double energy = 0.0;
        for (const float value : values) energy += static_cast<double>(value) * value;
        if (energy > 0.0) {
            const float inverse = static_cast<float>(1.0 / std::sqrt(energy));
            for (float & value : values) value *= inverse;
        }
        return values;
    };
    struct orthogonal_control {
        std::string label;
        std::vector<float> profile;
        bool opposite = false;
    };
    std::vector<orthogonal_control> controls;
    controls.push_back({"rank1", input.rank1_intervention, false});
    controls.push_back({"orthogonal", orthogonal.direction, false});
    std::vector<float> sum(input.rank1_intervention.size(), 0.0f);
    std::vector<float> difference(input.rank1_intervention.size(), 0.0f);
    for (size_t index = 0; index < sum.size(); ++index) {
        sum[index] = input.rank1_intervention[index] + orthogonal.direction[index];
        difference[index] = input.rank1_intervention[index] - orthogonal.direction[index];
    }
    controls.push_back({"sum", normalize(std::move(sum)), false});
    controls.push_back({"difference", normalize(std::move(difference)), true});
    if (controls.size() != common_flydelta_orthogonal_control_count) {
        error = "FlyDelta orthogonal control count is inconsistent";
        return false;
    }

    common_flydelta_arm_batch_request batch;
    batch.batch_id = job.id + ":orthogonal";
    batch.wave_id = "orthogonal-controls";
    common_flydelta_arm_request baseline;
    baseline.job_id = job.id;
    baseline.wave_id = batch.wave_id;
    baseline.proposal_index = 0;
    baseline.context_ref = job.seed.baseline_ref;
    baseline.fixture_ref = fixture.id;
    baseline.intervention_ref = job.seed.candidate_ref;
    baseline.batch_compatibility_key = baseline.context_ref + "\n" + baseline.fixture_ref;
    baseline.layer_indices = input.local_layers;
    baseline.coefficients.assign(input.local_layers.size(), 0.0f);
    baseline.fresh_context = true;
    baseline.request_capture = true;
    baseline.request_teacher_forced_margin = true;
    baseline.max_capture_bytes = 4U * 1024U * 1024U;
    baseline.max_generated_tokens = 0;
    baseline.arm_id = "flydelta://runtime/" + job.id + "/orthogonal/baseline";
    if (!common_flydelta_arm_request_validate(baseline, error)) return false;
    batch.arms.push_back(baseline);

    std::vector<common_flydelta_bootstrap_zoom_candidate> candidates;
    candidates.reserve(controls.size());
    for (size_t control_index = 0; control_index < controls.size(); ++control_index) {
        const auto & control = controls[control_index];
        common_flydelta_bootstrap_zoom_candidate candidate;
        candidate.phase = control.opposite
            ? common_flydelta_bootstrap_zoom_phase::sign_control
            : common_flydelta_bootstrap_zoom_phase::profile_zoom;
        candidate.total_scale = std::max(0.0001f, resume.selected_scale);
        candidate.opposite_sign_control = control.opposite;
        for (size_t layer = 0; layer < input.local_layers.size(); ++layer) {
            if (std::fabs(control.profile[layer]) <= 0.000001f) continue;
            const auto direction = std::find_if(available.begin(), available.end(),
                [&](const auto & value) {
                    return value.layer_index == static_cast<int32_t>(input.local_layers[layer]);
                });
            if (direction == available.end()) {
                error = "FlyDelta orthogonal control lacks a compatible source direction";
                return false;
            }
            candidate.layer_indices.push_back(input.local_layers[layer]);
            candidate.layer_weights.push_back(control.profile[layer]);
        }
        candidate.layer_weights = normalize(std::move(candidate.layer_weights));
        if (!common_flydelta_bootstrap_zoom_candidate_validate(candidate, error)) return false;
        common_flydelta_arm_request arm;
        arm.job_id = job.id;
        arm.wave_id = batch.wave_id;
        arm.proposal_index = control_index + 1;
        arm.context_ref = job.seed.baseline_ref;
        arm.fixture_ref = fixture.id;
        arm.intervention_ref = job.seed.candidate_ref;
        arm.batch_compatibility_key = arm.context_ref + "\n" + arm.fixture_ref;
        arm.layer_indices = candidate.layer_indices;
        arm.coefficients = candidate.layer_weights;
        arm.alpha = candidate.total_scale;
        arm.apply_overlay = true;
        arm.fresh_context = true;
        arm.request_capture = true;
        arm.request_teacher_forced_margin = true;
        arm.request_generation = false;
        arm.request_host_verification = false;
        arm.max_capture_bytes = 4U * 1024U * 1024U;
        arm.max_generated_tokens = 0;
        const std::string identity = job.id + "\n" + batch.wave_id + "\n" + control.label;
        arm.arm_id = "flydelta://runtime/" + job.id + "/orthogonal/" +
            hash_sha256_hex(identity.data(), identity.size()).substr(0, 32);
        if (!common_flydelta_arm_request_validate(arm, error)) return false;
        candidates.push_back(std::move(candidate));
        batch.arms.push_back(std::move(arm));
    }
    common_flydelta_arm_batch_result diagnostic_results;
    if (!daemon_flydelta_execute_batch(provider, batch, diagnostic_results, error) ||
            diagnostic_results.arms.size() != batch.arms.size()) {
        if (error.empty()) error = "FlyDelta orthogonal diagnostic batch returned incomplete arms";
        return false;
    }

    std::vector<size_t> frontier_indices;
    for (size_t index = 0; index < candidates.size(); ++index) frontier_indices.push_back(index);
    std::stable_sort(frontier_indices.begin(), frontier_indices.end(),
        [&](const size_t left, const size_t right) {
            const auto & lhs = diagnostic_results.arms[left + 1];
            const auto & rhs = diagnostic_results.arms[right + 1];
            const float lhs_score = lhs.margin_available ? lhs.margin.normalized_delta() : 0.0f;
            const float rhs_score = rhs.margin_available ? rhs.margin.normalized_delta() : 0.0f;
            return lhs_score > rhs_score;
        });
    constexpr size_t full_generation_top_k = common_flydelta_orthogonal_full_generation_top_k;
    if (frontier_indices.size() > full_generation_top_k) frontier_indices.resize(full_generation_top_k);
    common_flydelta_arm_batch_request full_batch;
    full_batch.batch_id = batch.batch_id + ":frontier";
    full_batch.wave_id = batch.wave_id;
    auto full_baseline = batch.arms.front();
    full_baseline.request_generation = true;
    full_baseline.request_host_verification = true;
    full_baseline.max_generated_tokens = 0;
    full_batch.arms.push_back(std::move(full_baseline));
    for (const size_t index : frontier_indices) {
        auto arm = batch.arms[index + 1];
        arm.request_generation = true;
        arm.request_host_verification = true;
        arm.max_generated_tokens = 0;
        full_batch.arms.push_back(std::move(arm));
    }
    common_flydelta_arm_batch_result full_results;
    if (!daemon_flydelta_execute_batch(provider, full_batch, full_results, error) ||
            full_results.arms.size() != full_batch.arms.size()) {
        if (error.empty()) error = "FlyDelta orthogonal frontier batch returned incomplete arms";
        return false;
    }
    const auto & baseline_result = full_results.arms.front();
    if (!baseline_result.executed || baseline_result.arm_id != baseline.arm_id) {
        error = "FlyDelta orthogonal baseline arm is invalid";
        return false;
    }
    std::vector<common_flydelta_bootstrap_zoom_trial> trials;
    trials.reserve(candidates.size());
    const auto anchor_direction = std::find_if(available.begin(), available.end(),
        [&](const auto & value) {
            return value.layer_index == static_cast<int32_t>(resume.anchor_layer);
        });
    if (anchor_direction == available.end()) {
        error = "FlyDelta orthogonal surface lacks the rank-one anchor direction";
        return false;
    }
    common_flydelta_search_pipeline_direction_result direction_result;
    // A direction candidate is an embedding-space vector.  Orthogonal here
    // is a local layer-profile coordinate, so retain the compatible anchor
    // vector and record the new experimental surface exclusively in origin
    // and the persisted surface trials rather than fabricating a mismatched
    // embedding-space direction.
    direction_result.direction.kind = common_flydelta_direction_kind::raw_repair;
    direction_result.direction.layer_index = anchor_direction->layer_index;
    direction_result.direction.values = anchor_direction->values;
    direction_result.direction.origin = "runtime_orthogonal_profile";
    direction_result.direction.extraction_id = job.seed.candidate_ref;
    direction_result.direction.source_samples = orthogonal.source_arm_count;
    direction_result.direction.retained_samples = orthogonal.source_arm_count;
    direction_result.direction.experimental_only = true;
    for (size_t index = 0; index < candidates.size(); ++index) {
        const auto frontier = std::find(frontier_indices.begin(), frontier_indices.end(), index);
        const bool full_execution = frontier != frontier_indices.end();
        const auto & arm = full_execution
            ? full_results.arms[1 + static_cast<size_t>(frontier - frontier_indices.begin())]
            : diagnostic_results.arms[index + 1];
        if (!arm.executed || arm.arm_id != batch.arms[index + 1].arm_id) {
            error = "FlyDelta orthogonal control arm is invalid";
            return false;
        }
        common_flydelta_bootstrap_zoom_trial trial;
        trial.candidate = candidates[index];
        trial.outcome = full_execution ? daemon_flydelta_classify_arm_pair(baseline_result, arm)
                                      : common_flydelta_counterfactual_outcome::unknown;
        trial.host_evaluated = full_execution && arm.host_evaluated;
        trial.verifier_known = full_execution && arm.verifier_known;
        trial.margin_available = arm.margin.available;
        trial.margin_delta = trial.margin_available
            ? (full_execution
                ? arm.margin.normalized_delta() - baseline_result.margin.normalized_delta()
                : arm.margin.normalized_delta()) : 0.0f;
        if (!daemon_flydelta_diagnostics_for_arm(provider, job, baseline.arm_id,
                batch.arms[index + 1], trial.candidate.layer_indices.front(),
                trial.diagnostics, error)) return false;
        trial.diagnostics_available = trial.diagnostics.schema_version == 1 &&
            trial.diagnostics.layer_index != 0;
        if (!common_flydelta_bootstrap_zoom_trial_validate(trial, error)) return false;
        trials.push_back(trial);
        common_flydelta_intervention_region_trial region_trial;
        region_trial.candidate.layer_indices = trial.candidate.layer_indices;
        region_trial.candidate.anchor_layer_index = resume.anchor_layer;
        region_trial.candidate.total_scale = trial.candidate.total_scale;
        region_trial.candidate.per_layer_scale = trial.candidate.total_scale /
            std::sqrt(static_cast<float>(std::max<size_t>(1, trial.candidate.layer_indices.size())));
        region_trial.requested_total_scale = trial.candidate.total_scale;
        region_trial.executed_total_scale = trial.candidate.total_scale;
        region_trial.outcome = trial.outcome;
        region_trial.quality_delta = full_execution
            ? arm.quality - baseline_result.quality : 0.0f;
        region_trial.margin = arm.margin;
        region_trial.executed = arm.executed;
        region_trial.host_evaluated = full_execution && arm.host_evaluated;
        region_trial.verifier_known = full_execution && arm.verifier_known;
        if (full_execution) daemon_flydelta_record_arm_observations(region_trial, arm, &baseline_result);
        region_trial.geometry_available = trial.diagnostics_available;
        region_trial.geometry = trial.diagnostics;
        region_trial.observed_decision_summary = arm.observed_decision_summary;
        region_trial.expected_decision_summary = arm.expected_decision_summary;
        region_trial.verifier_reason = arm.verifier_reason;
        region_trial.safe_to_continue = arm.host_outcome !=
            common_flydelta_counterfactual_outcome::harmed &&
            trial.diagnostics_available &&
            trial.diagnostics.cosine >= config.minimum_cosine &&
            trial.diagnostics.progress > 0.0f &&
            trial.diagnostics.leakage <= config.maximum_leakage &&
            trial.diagnostics.shift_norm <= config.maximum_shift_norm;
        region_trial.promising = region_trial.safe_to_continue &&
            trial.margin_available && trial.margin_delta > 0.0f;
        region_trial.search_score = trial.margin_available ? trial.margin_delta : 0.0f;
        region_trial.evidence_ref = full_execution ? arm.generation_ref : arm.capture_ref;
        if (!common_flydelta_intervention_region_trial_validate(region_trial, error)) return false;
        direction_result.region_trials.push_back(std::move(region_trial));
    }
    common_flydelta_bootstrap_zoom_selection selection;
    bool has_safe_trial = false;
    for (const auto & trial : trials) {
        if (trial.outcome == common_flydelta_counterfactual_outcome::harmed) continue;
        const bool geometry_safe = !trial.diagnostics_available ||
            (trial.diagnostics.cosine >= config.minimum_cosine &&
             trial.diagnostics.progress > 0.0f &&
             trial.diagnostics.leakage <= config.maximum_leakage &&
             trial.diagnostics.shift_norm <= config.maximum_shift_norm);
        if (geometry_safe) {
            has_safe_trial = true;
            break;
        }
    }
    if (has_safe_trial && !common_flydelta_select_bootstrap_zoom_trial(
            trials, selection, error)) return false;
    direction_result.region_selection.selected = selection.selected;
    direction_result.region_selection.trial_index = selection.trial_index;
    direction_result.region_selection.score = selection.search_score;
    output = {};
    output.directions.push_back(std::move(direction_result));
    output.search_status = selection.selected
        ? common_flydelta_search_status::candidate_available
        : common_flydelta_search_status::no_useful_utility;
    output.selection.selected = selection.selected;
    output.selection.intervention_region = selection.selected;
    output.selection.direction_index = 0;
    output.selection.region_trial_index = selection.trial_index;
    output.selection.score = selection.search_score;
    next = resume;
    next.state_ref.clear();
    next.parent_surface_revision = resume.surface_revision;
    next.surface_revision = resume.surface_revision + 1;
    next.search_rank = 2;
    next.surface_origin = "orthogonal_profile";
    next.parent_surface_ref = resume.state_ref;
    next.surface_trials = std::move(trials);
    return common_flydelta_bootstrap_zoom_state_validate(next, error);
}


} // namespace agent_daemon_flydelta_internal
