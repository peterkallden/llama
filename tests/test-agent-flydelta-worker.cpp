#include "agent/adaptation/flydelta/flydelta-worker.h"

#include <filesystem>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

static common_flydelta_experiment_job job(const std::string & id) {
    common_flydelta_experiment_job value;
    value.id = id;
    value.kind = common_flydelta_experiment_job_kind::counterfactual;
    value.seed.id = "evidence://repair/worker/flydelta";
    value.seed.behavior_key = "tool_use/diagnostics/missing-argument";
    value.seed.source = common_adaptation_evidence_source::tool_repair;
    value.seed.scope.namespace_id = "local";
    value.seed.scope.project_id = "project";
    value.seed.scope.session_id = "session";
    value.seed.task_fingerprint = "sha256:task";
    value.seed.model_profile_fingerprint = "sha256:model";
    value.seed.tokenizer_fingerprint = "sha256:tokenizer";
    value.seed.template_fingerprint = "sha256:template";
    value.seed.execution_context_fingerprint = "sha256:execution-context";
    value.seed.baseline_ref = "execution:failed";
    value.seed.candidate_ref = "execution:repaired";
    value.seed.verifier_ref = "verifier:v1";
    value.seed.evidence_ref = "evidence://repair/worker";
    value.seed.transaction_ids = {"learning://failed", "learning://repaired"};
    value.capture_manifest_ids = {"flydelta://capture/worker"};
    value.alpha_search.candidates = {0.05f};
    value.alpha_search.max_candidates = 1;
    value.code_revision = "worker-test:v1";
    return value;
}

static common_flydelta_counterfactual_report report(const std::string & job_id) {
    common_flydelta_counterfactual_report value;
    value.experiment_id = job_id;
    value.fixture_id = "flydelta://fixture/worker";
    value.candidate_id = "flydelta://candidate/worker";
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

static common_flydelta_direction_candidate direction_candidate() {
    common_flydelta_direction_candidate value;
    value.kind = common_flydelta_direction_kind::token_margin_direction;
    value.layer_index = 2;
    value.values = {1.0f, 0.0f};
    value.source_samples = 1;
    value.retained_samples = 1;
    value.median_alignment = 1.0f;
    return value;
}

int main() {
    std::string error;
    const auto root = std::filesystem::temp_directory_path() /
        "llama-agent-flydelta-worker-test";
    std::error_code ignored;
    std::filesystem::remove_all(root, ignored);
    const auto first = job("flydelta://job/worker-1");
    CHECK(common_flydelta_experiment_queue_enqueue(root, first, {}, error));

    common_flydelta_experiment_worker_report worker_report;
    bool callback_received_expected_job = false;
    CHECK(common_flydelta_experiment_worker_run_once(root, {},
        [&](const auto & claimed, auto & result, std::string &) {
            callback_received_expected_job = claimed.id == first.id;
            result.safe_summary = "counterfactual completed";
            result.counterfactual_reports.push_back(report(claimed.id));
            return true;
        }, worker_report, error));
    CHECK(callback_received_expected_job);
    CHECK(worker_report.state == common_flydelta_experiment_queue_state::succeeded);
    CHECK(worker_report.report_count == 1);

    auto direction_job = job("flydelta://job/worker-direction");
    direction_job.kind = common_flydelta_experiment_job_kind::direction;
    direction_job.capture_manifest_ids.clear();
    direction_job.behavior_delta_ids = {"flydelta://delta/worker"};
    CHECK(common_flydelta_experiment_queue_enqueue(root, direction_job, {}, error));
    CHECK(common_flydelta_experiment_worker_run_once(root, {},
        [&](const auto &, auto & result, std::string &) {
            result.safe_summary = "direction search completed";
            result.direction_candidates.push_back(direction_candidate());
            return true;
        }, worker_report, error));
    CHECK(worker_report.state == common_flydelta_experiment_queue_state::succeeded);
    CHECK(worker_report.report_count == 1);

    const auto second = job("flydelta://job/worker-2");
    CHECK(common_flydelta_experiment_queue_enqueue(root, second, {}, error));
    CHECK(common_flydelta_experiment_worker_run_once(root, {},
        [](const auto &, auto &, std::string &) { return true; }, worker_report, error));
    CHECK(worker_report.state == common_flydelta_experiment_queue_state::failed);

    CHECK(common_flydelta_experiment_worker_run_once(root, {},
        [](const auto &, auto &, std::string &) { return false; }, worker_report, error));
    CHECK(worker_report.state == common_flydelta_experiment_queue_state::pending);
    std::filesystem::remove_all(root, ignored);
    return 0;
}
