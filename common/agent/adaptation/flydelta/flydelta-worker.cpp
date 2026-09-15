#include "agent/adaptation/flydelta/flydelta-worker.h"

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
    if (claimed.job.kind == common_flydelta_experiment_job_kind::counterfactual &&
            result.counterfactual_reports.empty()) {
        error = "FlyDelta counterfactual worker result requires a report";
        return false;
    }
    if (claimed.job.kind == common_flydelta_experiment_job_kind::direction &&
            result.direction_candidates.empty()) {
        error = "FlyDelta direction worker result requires a candidate";
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
        result.direction_candidates.size();
    return true;
}
