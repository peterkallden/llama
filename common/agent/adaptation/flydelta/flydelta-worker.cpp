#include "agent/adaptation/flydelta/flydelta-worker.h"
#include "agent/adaptation/flydelta/flydelta-trace-internal.h"
#include "agent/adaptation/flydelta/flydelta-evaluator.h"

#include <algorithm>
#include <cmath>


namespace {

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
    if (result.graft_direction_ref.size() > 512) {
        error = "FlyDelta worker graft direction reference is invalid";
        return false;
    }
    if (result.graft_direction_refs.size() > 8) {
        error = "FlyDelta worker graft direction portfolio exceeds its bound";
        return false;
    }
    if (result.has_selected_candidate &&
            !common_flydelta_selected_candidate_descriptor_validate(
                result.selected_candidate, error)) return false;
    for (const auto & direction_ref : result.graft_direction_refs) {
        if (direction_ref.empty() || direction_ref.size() > 512) {
            error = "FlyDelta worker graft direction portfolio contains an invalid reference";
            return false;
        }
    }
    if (result.has_paired_intervention &&
            !common_flydelta_paired_intervention_proposal_validate(
                result.paired_intervention, error)) return false;
    for (const auto & counterfactual : result.counterfactual_reports) {
        if (!common_flydelta_counterfactual_report_validate(counterfactual, error)) return false;
        const bool belongs_to_evaluation = claimed.job.kind ==
            common_flydelta_experiment_job_kind::evaluation &&
            counterfactual.experiment_id.rfind(claimed.job.id + ":fixture:", 0) == 0;
        if (counterfactual.experiment_id != claimed.job.id && !belongs_to_evaluation) {
            error = "FlyDelta worker result belongs to another experiment job";
            return false;
        }
    }
    if (result.has_evaluation_report) {
        if (!common_flydelta_evaluation_report_validate(result.evaluation_report, error) ||
                result.evaluation_report.candidate_id != claimed.job.evaluation_candidate_id) {
            if (error.empty()) error = "FlyDelta evaluation report identity is invalid";
            return false;
        }
        for (const auto & fixture : result.evaluation_fixture_results) {
            if (!common_flydelta_evaluation_fixture_result_validate(fixture, error) ||
                    fixture.candidate_id != claimed.job.evaluation_candidate_id) return false;
        }
    }
    if (result.has_causal_diagnostic_report &&
            !common_flydelta_causal_diagnostic_report_validate(
                result.causal_diagnostic_report, error)) {
        return false;
    }
    for (const auto & direction : result.direction_candidates) {
        if (!common_flydelta_direction_candidate_validate(
                direction, direction.values.size(), error)) return false;
    }
    for (const auto & candidate : result.concept_candidates) {
        if (!common_flydelta_concept_candidate_validate(
                candidate, candidate.values.size(), error)) return false;
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
    if (claimed.job.kind == common_flydelta_experiment_job_kind::evaluation &&
            (!result.has_evaluation_report || result.evaluation_fixture_results.empty())) {
        error = "FlyDelta evaluation worker result requires a report and fixture results";
        return false;
    }
    if (claimed.job.kind == common_flydelta_experiment_job_kind::causal_diagnostic &&
            !result.has_causal_diagnostic_report) {
        error = "FlyDelta causal diagnostic worker result requires a report";
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
    if (claimed.job.kind == common_flydelta_experiment_job_kind::concept_synthesis &&
            (result.concept_candidates.empty() || result.direction_candidates.empty() ||
             result.graft_direction_ref.empty() || result.graft_direction_refs.empty() ||
             result.graft_direction_refs.front() != result.graft_direction_ref)) {
        error = "FlyDelta concept synthesis worker result requires candidates and a graft reference";
        return false;
    }
    if (claimed.job.kind == common_flydelta_experiment_job_kind::concept_capture &&
            result.concept_trajectory_refs.empty()) {
        error = "FlyDelta concept capture worker result requires trajectory references";
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
        std::string & error,
        const common_flydelta_worker_commit_callback & commit,
        const common_flydelta_worker_claim_callback & claim) {
    error.clear();
    report = {};
    const auto queue_owner = common_flydelta_acquire_queue_owner(queue_root, error);
    if (!queue_owner) return false;
    common_flydelta_claimed_experiment_job claimed;
    if (claim) {
        if (!claim(claimed, error)) return false;
    } else if (!common_flydelta_experiment_queue_claim_next(queue_root, limits, claimed, error)) return false;
    if (claimed.queue_key.empty()) {
        report.state = common_flydelta_experiment_queue_state::pending;
        return true;
    }
    report.job_id = claimed.job.id;
    report.completed_job = claimed.job;
    common_flydelta_experiment_worker_result result;
    std::string callback_error;
    bool succeeded = callback && callback(claimed.job, result, callback_error);
    if (succeeded) common_flydelta_trace_detail::append_worker_result(claimed.job, result, result.trace);
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
    if (!succeeded) error = callback_error;
    report.state = state;
    report.safe_summary = safe_summary;
    report.trace = std::move(result.trace);
    report.trace_json = common_flydelta_trace_to_json(report.trace);
    report.report_count = result.capture_manifests.size() +
        result.counterfactual_reports.size() + result.direction_candidates.size() +
        result.concept_candidates.size() +
        result.concept_trajectory_refs.size() +
        result.search_pipeline_results.size() +
        result.evaluation_fixture_results.size() +
        (result.has_causal_diagnostic_report
            ? result.causal_diagnostic_report.arms.size() : 0U);
    report.capture_manifests = std::move(result.capture_manifests);
    report.concept_candidates = std::move(result.concept_candidates);
    report.concept_trajectory_refs = std::move(result.concept_trajectory_refs);
    report.counterfactual_reports = std::move(result.counterfactual_reports);
    report.has_causal_diagnostic_report = result.has_causal_diagnostic_report;
    report.causal_diagnostic_report = std::move(result.causal_diagnostic_report);
    report.has_evaluation_report = result.has_evaluation_report;
    report.evaluation_report = std::move(result.evaluation_report);
    report.evaluation_fixture_results = std::move(result.evaluation_fixture_results);
    report.evidence_depth = result.evidence_depth;
    report.search_budget = result.search_budget;
    report.has_experiment_plan = result.has_experiment_plan;
    report.experiment_plan = std::move(result.experiment_plan);
    report.has_bootstrap_zoom_state = result.has_bootstrap_zoom_state;
    report.bootstrap_zoom_state = std::move(result.bootstrap_zoom_state);
    report.bootstrap_zoom_state_ref = std::move(result.bootstrap_zoom_state_ref);
    report.search_state_ref = std::move(result.search_state_ref);
    report.has_selected_candidate = result.has_selected_candidate;
    report.selected_candidate = result.selected_candidate;
    report.has_next_action = result.has_next_action;
    report.next_action = result.next_action;
    report.utility_decision = result.utility_decision;
    report.next_action_reason = std::move(result.next_action_reason);
    report.graft_direction_ref = std::move(result.graft_direction_ref);
    report.graft_direction_refs = std::move(result.graft_direction_refs);
    report.has_paired_intervention = result.has_paired_intervention;
    report.paired_intervention = std::move(result.paired_intervention);
    report.has_representation_augmentation_state = result.has_representation_augmentation_state;
    report.representation_augmentation_state = std::move(result.representation_augmentation_state);
    report.representation_augmentation_state_ref =
        std::move(result.representation_augmentation_state_ref);
    // Preserve the validated result and finish host effects before the
    // terminal acknowledgement. A failed commit leaves the job recoverable.
    if (succeeded && commit && !commit(claimed.job, report, error)) return false;
    std::string completion_error;
    if (!common_flydelta_experiment_queue_complete(
            queue_root, claimed, state, safe_summary, limits, completion_error)) {
        error = completion_error;
        return false;
    }
    return true;
}

bool common_flydelta_worker_result_from_evaluator(
        const common_flydelta_evaluator_result & source,
        common_flydelta_experiment_worker_result & target,
        std::string & error) {
    error.clear();
    target = {};
    target.capture_manifests = source.capture_manifests;
    target.concept_candidates = source.concept_candidates;
    target.concept_trajectory_refs = source.concept_trajectory_refs;
    target.graft_direction_ref = source.graft_direction_ref;
    target.graft_direction_refs = source.graft_direction_refs;
    target.has_paired_intervention = source.has_paired_intervention;
    target.paired_intervention = source.paired_intervention;
    target.counterfactual_reports = source.counterfactual_reports;
    target.has_causal_diagnostic_report = source.has_causal_diagnostic_report;
    target.causal_diagnostic_report = source.causal_diagnostic_report;
    target.has_evaluation_report = source.has_evaluation_report;
    target.evaluation_report = source.evaluation_report;
    target.evaluation_fixture_results = source.evaluation_fixture_results;
    target.direction_candidates = source.direction_candidates;
    target.basis_directions = source.basis_directions;
    target.search_pipeline_results = source.search_pipeline_results;
    target.has_selected_candidate = source.has_selected_candidate;
    target.selected_candidate = source.selected_candidate;
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
