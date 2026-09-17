#include "agent/adaptation/flydelta/flydelta-worker.h"
#include "agent/adaptation/flydelta/flydelta-evaluator.h"

#include <algorithm>

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
        if (result.evidence_depth.compatible_samples != result.aggregation.compatible_samples) {
            error = "FlyDelta worker evidence depth and aggregation counts differ";
            return false;
        }
    }
    if (claimed.job.kind == common_flydelta_experiment_job_kind::counterfactual &&
            result.counterfactual_reports.empty()) {
        error = "FlyDelta counterfactual worker result requires a report";
            return false;
    }
    if (claimed.job.kind == common_flydelta_experiment_job_kind::search_pipeline &&
            (result.search_pipeline_results.empty() || result.search_continuations.empty())) {
        error = "FlyDelta search pipeline worker result requires a result and continuation";
        return false;
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
    common_flydelta_experiment_worker_result result;
    std::string callback_error;
    bool succeeded = callback && callback(claimed.job, result, callback_error);
    if (succeeded && !validate_result(claimed, result, callback_error)) succeeded = false;

    const auto state = succeeded
        ? common_flydelta_experiment_queue_state::succeeded
        : common_flydelta_experiment_queue_state::failed;
    const auto safe_summary = succeeded
        ? result.safe_summary
        : "FlyDelta experiment callback or result validation failed";
    if (!common_flydelta_experiment_queue_complete(
            queue_root, claimed, state, safe_summary, limits, error)) return false;
    report.state = state;
    report.safe_summary = safe_summary;
    report.report_count = result.counterfactual_reports.size() +
        result.direction_candidates.size() + result.search_pipeline_results.size();
    report.evidence_depth = result.evidence_depth;
    report.search_budget = result.search_budget;
    report.has_experiment_plan = result.has_experiment_plan;
    report.experiment_plan = std::move(result.experiment_plan);
    report.has_bootstrap_zoom_state = result.has_bootstrap_zoom_state;
    report.bootstrap_zoom_state = std::move(result.bootstrap_zoom_state);
    report.bootstrap_zoom_state_ref = std::move(result.bootstrap_zoom_state_ref);
    report.search_state_ref = std::move(result.search_state_ref);
    report.has_representation_augmentation_state = result.has_representation_augmentation_state;
    report.representation_augmentation_state = std::move(result.representation_augmentation_state);
    report.representation_augmentation_state_ref =
        std::move(result.representation_augmentation_state_ref);
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
            worker_result.counterfactual_reports =
                std::move(evaluator_result.counterfactual_reports);
            worker_result.direction_candidates =
                std::move(evaluator_result.direction_candidates);
            worker_result.basis_directions =
                std::move(evaluator_result.basis_directions);
            worker_result.search_pipeline_results =
                std::move(evaluator_result.search_pipeline_results);
            worker_result.search_continuations =
                std::move(evaluator_result.search_continuations);
            worker_result.delta_memory_weights =
                std::move(evaluator_result.delta_memory_weights);
            worker_result.aggregation = std::move(evaluator_result.aggregation);
            worker_result.evidence_depth = evaluator_result.evidence_depth;
            worker_result.search_budget = evaluator_result.search_budget;
            worker_result.has_experiment_plan = evaluator_result.has_experiment_plan;
            worker_result.experiment_plan = std::move(evaluator_result.experiment_plan);
            worker_result.has_bootstrap_zoom_state = evaluator_result.has_bootstrap_zoom_state;
            worker_result.bootstrap_zoom_state = std::move(evaluator_result.bootstrap_zoom_state);
            worker_result.bootstrap_zoom_state_ref = std::move(
                evaluator_result.bootstrap_zoom_state_ref);
            worker_result.search_state_ref = std::move(evaluator_result.search_state_ref);
            worker_result.has_representation_augmentation_state =
                evaluator_result.has_representation_augmentation_state;
            worker_result.representation_augmentation_state = std::move(
                evaluator_result.representation_augmentation_state);
            worker_result.representation_augmentation_state_ref = std::move(
                evaluator_result.representation_augmentation_state_ref);
            worker_result.safe_summary = "FlyDelta evaluator processed " +
                std::to_string(evaluator_result.processed_references) + " reference(s)";
            return true;
        }, report, error);
}
