#include "agent/adaptation/flydelta/flydelta-worker.h"
#include "agent/adaptation/flydelta/flydelta-evaluator.h"

#include <algorithm>
#include <cmath>

#include <nlohmann/json.hpp>

namespace {

using json = nlohmann::ordered_json;

bool finite(const float value) {
    return std::isfinite(value);
}

const char * phase_name(const common_flydelta_experiment_job_kind kind) {
    switch (kind) {
        case common_flydelta_experiment_job_kind::counterfactual: return "counterfactual";
        case common_flydelta_experiment_job_kind::direction: return "direction";
        case common_flydelta_experiment_job_kind::basis: return "basis";
        case common_flydelta_experiment_job_kind::delta_memory: return "delta_memory";
        case common_flydelta_experiment_job_kind::search_pipeline: return "search_pipeline";
        case common_flydelta_experiment_job_kind::donor_capture: return "donor_capture";
    }
    return "unknown";
}

void copy_geometry(const common_flydelta_representation_diagnostics & source,
        common_flydelta_trace_arm & target) {
    target.geometry_available = true;
    target.cosine = source.cosine;
    target.progress = source.progress;
    target.leakage = source.leakage;
    target.shift_norm = source.shift_norm;
}

void copy_margin(const common_flydelta_margin_comparison & source,
        common_flydelta_trace_arm & target) {
    if (!source.available) return;
    target.margin_available = true;
    target.margin_total = source.candidate.total_delta();
    target.margin_normalized = source.candidate.normalized_delta();
    target.margin_delta_total = source.total_delta();
    target.margin_delta_normalized = source.normalized_delta();
}

void append_region_arm(const common_flydelta_intervention_region_trial & trial,
        const size_t direction_index, const size_t trial_index,
        common_flydelta_trace & trace) {
    common_flydelta_trace_arm arm;
    arm.phase = "region";
    arm.arm_id = "region:" + std::to_string(direction_index) + ":" +
        std::to_string(trial_index);
    arm.layer_indices = trial.candidate.layer_indices;
    arm.scale = trial.candidate.total_scale;
    arm.requested_scale = trial.requested_total_scale > 0.0f
        ? trial.requested_total_scale : trial.candidate.total_scale;
    arm.executed_scale = trial.executed_total_scale > 0.0f
        ? trial.executed_total_scale : trial.candidate.total_scale;
    arm.dose_evaluated = trial.dose_evaluated;
    arm.relative_dose = trial.relative_dose;
    arm.dose_comparable = trial.dose_comparable;
    arm.dose_safety_limited = trial.dose_safety_limited;
    arm.dose_action = trial.dose_evaluated
        ? common_flydelta_dose_action_name(trial.dose_action) : "not_evaluated";
    arm.dose_reason = trial.dose_reason;
    copy_margin(trial.margin_comparison, arm);
    if (!arm.margin_available && trial.margin.available) {
        arm.margin_available = true;
        arm.margin_total = trial.margin.total_delta();
        arm.margin_normalized = trial.margin.normalized_delta();
    }
    if (trial.geometry_available) copy_geometry(trial.geometry, arm);
    arm.search_score = trial.search_score;
    arm.promising = trial.promising;
    arm.safe_to_continue = trial.safe_to_continue;
    arm.host_evaluated = trial.executed;
    arm.verifier_known = trial.verifier_known;
    arm.host_outcome = trial.outcome;
    arm.candidate_passed = trial.outcome == common_flydelta_counterfactual_outcome::helped;
    arm.evidence_ref = trial.evidence_ref;
    trace.arms.push_back(std::move(arm));
}

void append_counterfactual_arm(const common_flydelta_counterfactual_report & report,
        const common_flydelta_experiment_job & job, common_flydelta_trace & trace) {
    common_flydelta_trace_arm arm;
    arm.phase = phase_name(job.kind);
    arm.arm_id = report.candidate_id;
    arm.scale = job.alpha_search.candidates.empty() ? 0.0f :
        job.alpha_search.candidates.front();
    arm.host_evaluated = report.candidate.executed;
    arm.verifier_known = report.candidate.verifier_known;
    arm.candidate_passed = report.candidate.passed;
    arm.host_outcome = report.outcome;
    arm.evidence_ref = report.candidate.evidence_ref;
    arm.has_baseline = true;
    arm.baseline_executed = report.baseline.executed;
    arm.baseline_verifier_known = report.baseline.verifier_known;
    arm.baseline_passed = report.baseline.passed;
    trace.arms.push_back(std::move(arm));
}

void append_derived_trace(const common_flydelta_experiment_job & job,
        const common_flydelta_experiment_worker_result & result,
        common_flydelta_trace & trace) {
    trace.job_id = job.id;
    trace.phase = phase_name(job.kind);
    trace.behavior_key = job.seed.behavior_key;
    trace.fixture_baseline_ref = job.seed.baseline_ref;
    trace.surface_parent_best_ref = job.seed.candidate_ref;
    trace.evidence_rank = result.evidence_depth.stable_rank;
    trace.search_rank = result.basis_directions.size();
    trace.region_budget = result.search_budget.max_region_trials;
    trace.coefficient_budget = result.search_budget.max_coefficient_trials;
    trace.tfo_lite_allowed = result.search_budget.allow_tfo_lite;
    if (!result.search_pipeline_results.empty()) {
        trace.search_status = common_flydelta_search_status_name(
            result.search_pipeline_results.back().search_status);
    }
    if (result.has_experiment_plan) {
        trace.bootstrap_refinement = common_flydelta_bootstrap_refinement_kind_name(
            result.experiment_plan.bootstrap_refinement);
    }
    if (result.has_bootstrap_zoom_state &&
            result.bootstrap_zoom_state.alpha_response_available) {
        trace.alpha_response_available = true;
        trace.alpha_response_status =
            result.bootstrap_zoom_state.alpha_response.response_status;
        trace.alpha_range_not_exhausted =
            result.bootstrap_zoom_state.alpha_response.range_not_exhausted;
        trace.alpha_last_scale = result.bootstrap_zoom_state.alpha_response.last_scale;
        trace.alpha_last_requested_scale = result.bootstrap_zoom_state.alpha_response.last_requested_scale;
        trace.alpha_last_executed_scale = result.bootstrap_zoom_state.alpha_response.last_executed_scale;
        trace.alpha_last_relative_dose = result.bootstrap_zoom_state.alpha_response.last_relative_dose;
        trace.alpha_last_dose_action = common_flydelta_dose_action_name(
            result.bootstrap_zoom_state.alpha_response.last_dose_action);
        trace.alpha_last_dose_evaluated = result.bootstrap_zoom_state.alpha_response.last_dose_evaluated;
        trace.alpha_last_dose_safety_limited = result.bootstrap_zoom_state.alpha_response.last_dose_safety_limited;
        trace.alpha_utility_slope = result.bootstrap_zoom_state.alpha_response.utility_slope;
        trace.alpha_best_margin_delta_normalized =
            result.bootstrap_zoom_state.alpha_response.best_margin_delta_normalized;
    }
    trace.has_next_action = result.has_next_action;
    trace.next_action = result.next_action;
    trace.next_action_reason = result.next_action_reason;

    if (result.counterfactual_reports.size() <= 256) {
        for (const auto & report : result.counterfactual_reports) {
            append_counterfactual_arm(report, job, trace);
        }
    }
    size_t direction_index = 0;
    for (const auto & pipeline : result.search_pipeline_results) {
        if (pipeline.alpha_response_available) {
            trace.alpha_response_available = true;
            trace.alpha_response_status = pipeline.alpha_response.response_status;
            trace.alpha_range_not_exhausted = pipeline.alpha_response.range_not_exhausted;
            trace.alpha_last_scale = pipeline.alpha_response.last_scale;
            trace.alpha_last_requested_scale = pipeline.alpha_response.last_requested_scale;
            trace.alpha_last_executed_scale = pipeline.alpha_response.last_executed_scale;
            trace.alpha_last_relative_dose = pipeline.alpha_response.last_relative_dose;
            trace.alpha_last_dose_action = common_flydelta_dose_action_name(
                pipeline.alpha_response.last_dose_action);
            trace.alpha_last_dose_evaluated = pipeline.alpha_response.last_dose_evaluated;
            trace.alpha_last_dose_safety_limited = pipeline.alpha_response.last_dose_safety_limited;
            trace.alpha_utility_slope = pipeline.alpha_response.utility_slope;
            trace.alpha_best_margin_delta_normalized =
                pipeline.alpha_response.best_margin_delta_normalized;
        }
        for (const auto & direction : pipeline.directions) {
            trace.whirlpool.push_back(direction.whirlpool_trace);
            trace.model_evaluations += direction.whirlpool_trace.model_evaluations;
            for (size_t trial_index = 0; trial_index < direction.region_trials.size();
                    ++trial_index) {
                if (trace.arms.size() >= 256) break;
                append_region_arm(direction.region_trials[trial_index], direction_index,
                    trial_index, trace);
            }
            ++direction_index;
        }
    }
    if (trace.model_evaluations == 0) trace.model_evaluations = trace.arms.size();
}

json trace_arm_json(const common_flydelta_trace_arm & arm) {
    return {
        {"phase", arm.phase}, {"arm_id", arm.arm_id},
        {"layer_indices", arm.layer_indices}, {"scale", arm.scale},
        {"requested_scale", arm.requested_scale},
        {"executed_scale", arm.executed_scale},
        {"dose_evaluated", arm.dose_evaluated},
        {"relative_dose", arm.relative_dose},
        {"dose_comparable", arm.dose_comparable},
        {"dose_safety_limited", arm.dose_safety_limited},
        {"dose_action", arm.dose_action},
        {"dose_reason", arm.dose_reason},
        {"coefficients", arm.coefficients},
        {"margin_available", arm.margin_available},
        {"margin_total", arm.margin_total},
        {"margin_normalized", arm.margin_normalized},
        {"margin_delta_total", arm.margin_delta_total},
        {"margin_delta_normalized", arm.margin_delta_normalized},
        {"geometry_available", arm.geometry_available},
        {"cosine", arm.cosine}, {"progress", arm.progress},
        {"leakage", arm.leakage}, {"shift_norm", arm.shift_norm},
        {"search_score", arm.search_score}, {"promising", arm.promising},
        {"safe_to_continue", arm.safe_to_continue},
        {"host_evaluated", arm.host_evaluated},
        {"verifier_known", arm.verifier_known},
        {"candidate_passed", arm.candidate_passed},
        {"has_baseline", arm.has_baseline},
        {"baseline_executed", arm.baseline_executed},
        {"baseline_verifier_known", arm.baseline_verifier_known},
        {"baseline_passed", arm.baseline_passed},
        {"host_outcome", common_flydelta_counterfactual_outcome_name(arm.host_outcome)},
        {"evidence_ref", arm.evidence_ref}
    };
}

bool validate_result(
        const common_flydelta_claimed_experiment_job & claimed,
        const common_flydelta_experiment_worker_result & result,
        std::string & error) {
    error.clear();
    if (result.safe_summary.size() > 4U * 1024U) {
        error = "FlyDelta worker result summary exceeds bound";
        return false;
    }
    if (result.has_bootstrap_zoom_state &&
            !common_flydelta_bootstrap_zoom_state_validate(result.bootstrap_zoom_state, error)) {
        return false;
    }
    if (result.bootstrap_zoom_state_ref.size() > 512) {
        error = "FlyDelta worker BootstrapZoom state reference is invalid";
        return false;
    }
    if (result.search_state_ref.size() > 512) {
        error = "FlyDelta worker search state reference is invalid";
        return false;
    }
    if (result.next_action_reason.size() > 512) {
        error = "FlyDelta worker next-action reason is invalid";
        return false;
    }
    if (!common_flydelta_trace_validate(result.trace, error)) return false;
    for (const auto & manifest : result.capture_manifests) {
        if (!common_flydelta_capture_manifest_validate(
                manifest, 4U * 1024U * 1024U, error)) return false;
    }
    if (result.has_representation_augmentation_state &&
            !common_flydelta_representation_augmentation_state_validate(
                result.representation_augmentation_state,
                common_flydelta_representation_augmentation_config{}, error)) {
        return false;
    }
    if (result.representation_augmentation_state_ref.size() > 512) {
        error = "FlyDelta worker representation augmentation state reference is invalid";
        return false;
    }
    for (const auto & counterfactual : result.counterfactual_reports) {
        if (!common_flydelta_counterfactual_report_validate(counterfactual, error)) return false;
        if (counterfactual.experiment_id != claimed.job.id) {
            error = "FlyDelta worker result belongs to another experiment job";
            return false;
        }
    }
    for (const auto & direction : result.direction_candidates) {
        if (!common_flydelta_direction_candidate_validate(
                direction, direction.values.size(), error)) return false;
    }
    if (!result.direction_candidates.empty()) {
        if (!common_flydelta_search_budget_validate(result.search_budget, error)) return false;
        if (result.evidence_depth.compatible_samples !=
                result.aggregation.evidence_eligible_samples) {
            error = "FlyDelta worker evidence depth and aggregation counts differ";
            return false;
        }
    }
    if (claimed.job.kind == common_flydelta_experiment_job_kind::counterfactual &&
            result.counterfactual_reports.empty()) {
        error = "FlyDelta counterfactual worker result requires a report";
            return false;
    }
    if (claimed.job.kind == common_flydelta_experiment_job_kind::search_pipeline) {
        if (result.search_pipeline_results.empty()) {
            error = "FlyDelta search pipeline worker result requires a result";
            return false;
        }
        if (result.search_continuations.empty()) {
            if (result.search_pipeline_results.size() != 1 ||
                    result.search_pipeline_results.front().selection.selected ||
                    !common_flydelta_search_status_is_terminal_without_candidate(
                        result.search_pipeline_results.front().search_status)) {
                error = "FlyDelta empty search continuation is not a terminal result";
                return false;
            }
        } else if (result.search_continuations.size() != result.search_pipeline_results.size()) {
            error = "FlyDelta search pipeline result and continuation counts differ";
            return false;
        }
    }
    if (claimed.job.kind == common_flydelta_experiment_job_kind::direction &&
            result.direction_candidates.empty()) {
        error = "FlyDelta direction worker result requires a candidate";
        return false;
    }
    if (claimed.job.kind == common_flydelta_experiment_job_kind::basis &&
            result.basis_directions.empty()) {
        error = "FlyDelta basis worker result requires a direction";
        return false;
    }
    if (claimed.job.kind == common_flydelta_experiment_job_kind::delta_memory &&
            result.delta_memory_weights.empty()) {
        error = "FlyDelta DeltaMemory worker result requires weights";
        return false;
    }
    if (claimed.job.kind == common_flydelta_experiment_job_kind::donor_capture &&
            (result.capture_manifests.empty() ||
             result.capture_manifests.size() != claimed.job.capture_candidate_ids.size())) {
        error = "FlyDelta donor capture worker result requires a manifest";
        return false;
    }
    return true;
}

} // namespace

bool common_flydelta_experiment_worker_run_once(
        const std::filesystem::path & queue_root,
        const common_flydelta_experiment_queue_limits & limits,
        const common_flydelta_experiment_worker_callback & callback,
        common_flydelta_experiment_worker_report & report,
        std::string & error) {
    error.clear();
    report = {};
    common_flydelta_claimed_experiment_job claimed;
    if (!common_flydelta_experiment_queue_claim_next(queue_root, limits, claimed, error)) return false;
    if (claimed.queue_key.empty()) {
        report.state = common_flydelta_experiment_queue_state::pending;
        return true;
    }
    report.job_id = claimed.job.id;
    report.completed_job = claimed.job;
    common_flydelta_experiment_worker_result result;
    std::string callback_error;
    bool succeeded = callback && callback(claimed.job, result, callback_error);
    if (succeeded) append_derived_trace(claimed.job, result, result.trace);
    if (succeeded && !validate_result(claimed, result, callback_error)) succeeded = false;
    if (!succeeded && callback_error.empty()) {
        callback_error = callback
            ? "FlyDelta experiment callback or result validation failed"
            : "FlyDelta experiment worker has no evaluator callback";
    }
    if (!succeeded && error.empty() && !callback_error.empty()) error = callback_error;

    const auto state = succeeded
        ? common_flydelta_experiment_queue_state::succeeded
        : common_flydelta_experiment_queue_state::failed;
    const auto safe_summary = succeeded
        ? result.safe_summary
        : "FlyDelta experiment callback or result validation failed";
    std::string completion_error;
    if (!common_flydelta_experiment_queue_complete(
            queue_root, claimed, state, safe_summary, limits, completion_error)) {
        error = completion_error;
        return false;
    }
    if (!succeeded) error = callback_error;
    report.state = state;
    report.safe_summary = safe_summary;
    report.trace = std::move(result.trace);
    report.trace_json = common_flydelta_trace_to_json(report.trace);
    report.report_count = result.capture_manifests.size() +
        result.counterfactual_reports.size() + result.direction_candidates.size() +
        result.search_pipeline_results.size();
    report.capture_manifests = std::move(result.capture_manifests);
    report.evidence_depth = result.evidence_depth;
    report.search_budget = result.search_budget;
    report.has_experiment_plan = result.has_experiment_plan;
    report.experiment_plan = std::move(result.experiment_plan);
    report.has_bootstrap_zoom_state = result.has_bootstrap_zoom_state;
    report.bootstrap_zoom_state = std::move(result.bootstrap_zoom_state);
    report.bootstrap_zoom_state_ref = std::move(result.bootstrap_zoom_state_ref);
    report.search_state_ref = std::move(result.search_state_ref);
    report.has_next_action = result.has_next_action;
    report.next_action = result.next_action;
    report.utility_decision = result.utility_decision;
    report.next_action_reason = std::move(result.next_action_reason);
    report.has_representation_augmentation_state = result.has_representation_augmentation_state;
    report.representation_augmentation_state = std::move(result.representation_augmentation_state);
    report.representation_augmentation_state_ref =
        std::move(result.representation_augmentation_state_ref);
    return true;
}

bool common_flydelta_trace_validate(
        const common_flydelta_trace & trace, std::string & error) {
    error.clear();
    if (trace.schema_version != 1 || trace.job_id.size() > 512 ||
            trace.phase.size() > 64 || trace.behavior_key.size() > 256 ||
            trace.fixture_baseline_ref.size() > 512 ||
            trace.surface_parent_best_ref.size() > 512 ||
            trace.next_action_reason.size() > 512 || trace.arms.size() > 256 ||
            trace.whirlpool.size() > 32 || trace.bootstrap_refinement.size() > 64 ||
            !finite(trace.alpha_last_scale) || !finite(trace.alpha_utility_slope) ||
            !finite(trace.alpha_best_margin_delta_normalized)) {
        error = "FlyDelta trace exceeds its bounds";
        return false;
    }
    for (const auto & arm : trace.arms) {
        if (arm.phase.size() > 64 || arm.arm_id.size() > 512 ||
                arm.evidence_ref.size() > 512 || arm.layer_indices.size() > 64 ||
                arm.coefficients.size() > 64 || !finite(arm.scale) ||
                !finite(arm.margin_total) || !finite(arm.margin_normalized) ||
                !finite(arm.margin_delta_total) || !finite(arm.margin_delta_normalized) ||
                !finite(arm.cosine) || !finite(arm.progress) || !finite(arm.leakage) ||
                !finite(arm.shift_norm) || !finite(arm.search_score)) {
            error = "FlyDelta trace arm is invalid";
            return false;
        }
    }
    return true;
}

std::string common_flydelta_trace_to_json(const common_flydelta_trace & trace) {
    json arms = json::array();
    for (const auto & arm : trace.arms) arms.push_back(trace_arm_json(arm));
    json whirlpool = json::array();
    for (const auto & value : trace.whirlpool) {
        json rounds = json::array();
        for (const auto & round : value.rounds) {
            rounds.push_back({
                {"round", round.round}, {"centre_before", round.centre_before},
                {"radius_before", round.radius_before},
                {"probed_layers", round.probed_layers},
                {"best_probe_layer", round.best_probe_layer},
                {"best_probe_score", round.best_probe_score},
                {"centre_after", round.centre_after},
                {"radius_after", round.radius_after}
            });
        }
        whirlpool.push_back({
            {"schema_version", value.schema_version},
            {"model_evaluations", value.model_evaluations},
            {"best_trial_index", value.best_trial_index},
            {"best_search_score", value.best_search_score},
            {"final_centre", value.final_centre},
            {"final_radius", value.final_radius},
            {"rounds", std::move(rounds)}
        });
    }
    const json payload = {
        {"kind", "flydelta_trace"}, {"schema_version", trace.schema_version},
        {"job_id", trace.job_id}, {"phase", trace.phase},
        {"behavior_key", trace.behavior_key},
        {"fixture_baseline_ref", trace.fixture_baseline_ref},
        {"surface_parent_best_ref", trace.surface_parent_best_ref},
        {"evidence_rank", trace.evidence_rank}, {"search_rank", trace.search_rank},
        {"model_evaluations", trace.model_evaluations},
        {"region_budget", trace.region_budget},
        {"coefficient_budget", trace.coefficient_budget},
        {"tfo_lite_allowed", trace.tfo_lite_allowed},
        {"search_status", trace.search_status},
        {"bootstrap_refinement", trace.bootstrap_refinement},
        {"alpha_response_available", trace.alpha_response_available},
        {"alpha_response_status", common_flydelta_alpha_response_status_name(
            trace.alpha_response_status)},
        {"alpha_range_not_exhausted", trace.alpha_range_not_exhausted},
        {"alpha_last_scale", trace.alpha_last_scale},
        {"alpha_last_requested_scale", trace.alpha_last_requested_scale},
        {"alpha_last_executed_scale", trace.alpha_last_executed_scale},
        {"alpha_last_relative_dose", trace.alpha_last_relative_dose},
        {"alpha_last_dose_action", trace.alpha_last_dose_action},
        {"alpha_last_dose_evaluated", trace.alpha_last_dose_evaluated},
        {"alpha_last_dose_safety_limited", trace.alpha_last_dose_safety_limited},
        {"alpha_utility_slope", trace.alpha_utility_slope},
        {"alpha_best_margin_delta_normalized",
            trace.alpha_best_margin_delta_normalized},
        {"has_next_action", trace.has_next_action},
        {"next_action", common_flydelta_next_action_name(trace.next_action)},
        {"next_action_reason", trace.next_action_reason},
        {"arms", std::move(arms)}, {"whirlpool", std::move(whirlpool)}
    };
    return payload.dump();
}

bool common_flydelta_worker_result_from_evaluator(
        const common_flydelta_evaluator_result & source,
        common_flydelta_experiment_worker_result & target,
        std::string & error) {
    error.clear();
    target = {};
    target.capture_manifests = source.capture_manifests;
    target.counterfactual_reports = source.counterfactual_reports;
    target.direction_candidates = source.direction_candidates;
    target.basis_directions = source.basis_directions;
    target.search_pipeline_results = source.search_pipeline_results;
    target.search_continuations = source.search_continuations;
    target.delta_memory_weights = source.delta_memory_weights;
    target.aggregation = source.aggregation;
    target.evidence_depth = source.evidence_depth;
    target.search_budget = source.search_budget;
    target.has_experiment_plan = source.has_experiment_plan;
    target.experiment_plan = source.experiment_plan;
    target.has_bootstrap_zoom_state = source.has_bootstrap_zoom_state;
    target.bootstrap_zoom_state = source.bootstrap_zoom_state;
    target.bootstrap_zoom_state_ref = source.bootstrap_zoom_state_ref;
    target.search_state_ref = source.search_state_ref;
    target.has_next_action = source.has_next_action;
    target.next_action = source.next_action;
    target.utility_decision = source.utility_decision;
    target.next_action_reason = source.next_action_reason;
    target.has_representation_augmentation_state =
        source.has_representation_augmentation_state;
    target.representation_augmentation_state = source.representation_augmentation_state;
    target.representation_augmentation_state_ref = source.representation_augmentation_state_ref;
    target.safe_summary = "FlyDelta evaluator processed " +
        std::to_string(source.processed_references) + " reference(s)";
    return true;
}

bool common_flydelta_experiment_worker_run_evaluator_once(
        const std::filesystem::path & queue_root,
        const common_flydelta_experiment_queue_limits & limits,
        const common_flydelta_evaluator_config & config,
        const common_flydelta_evaluator_callbacks & callbacks,
        common_flydelta_experiment_worker_report & report,
        std::string & error) {
    return common_flydelta_experiment_worker_run_once(
        queue_root, limits,
        [&](const common_flydelta_experiment_job & job,
                common_flydelta_experiment_worker_result & worker_result,
                std::string & callback_error) {
            common_flydelta_evaluator_result evaluator_result;
            if (!common_flydelta_evaluate_job(
                job, config, callbacks, evaluator_result, callback_error)) return false;
            return common_flydelta_worker_result_from_evaluator(
                evaluator_result, worker_result, callback_error);
        }, report, error);
}
