#include "agent/adaptation/flydelta/flydelta-worker.h"
#include "agent/adaptation/flydelta/flydelta-evaluator.h"

#include <filesystem>
#include <iostream>
#include <utility>

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

static common_flydelta_behavior_delta behavior_delta() {
    common_flydelta_behavior_delta value;
    value.id = "flydelta://behavior/worker";
    value.source = common_adaptation_evidence_source::tool_repair;
    value.behavior_key = "tool_use/diagnostics/missing-argument";
    value.capture_manifest_id = "flydelta://capture/worker";
    value.host_evidence_ref = "evidence://repair/worker";
    value.model_profile_fingerprint = "sha256:model";
    value.execution_context_fingerprint = "sha256:execution-context";
    value.capture_layout_revision = "layout:v1";
    value.layer_index = 2;
    value.values = {1.0f, 0.0f};
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
            result.aggregation.compatible_samples = 1;
            result.evidence_depth.compatible_samples = 1;
            result.evidence_depth.depth = common_flydelta_search_depth::bootstrap;
            result.search_budget = common_flydelta_search_budget_for_depth(
                common_flydelta_search_depth::bootstrap);
            result.has_bootstrap_zoom_state = true;
            result.bootstrap_zoom_state.state_ref = "flydelta://state/worker-bootstrap";
            result.bootstrap_zoom_state.behavior_key = direction_job.seed.behavior_key;
            result.bootstrap_zoom_state.model_profile_fingerprint =
                direction_job.seed.model_profile_fingerprint;
            result.bootstrap_zoom_state.capture_layout_revision = "layout:v1";
            result.bootstrap_zoom_state.anchor_layer = 2;
            result.bootstrap_zoom_state.selected_scale = 0.05f;
            result.bootstrap_zoom_state.extra_model_trials = 3;
            result.bootstrap_zoom_state.next_candidate_index = 2;
            result.bootstrap_zoom_state_ref = result.bootstrap_zoom_state.state_ref;
            return true;
        }, worker_report, error));
    CHECK(worker_report.state == common_flydelta_experiment_queue_state::succeeded);
    CHECK(worker_report.report_count == 1);
    CHECK(worker_report.evidence_depth.depth == common_flydelta_search_depth::bootstrap);
    CHECK(worker_report.search_budget.max_region_trials == 4);
    CHECK(worker_report.has_bootstrap_zoom_state &&
        worker_report.bootstrap_zoom_state_ref == "flydelta://state/worker-bootstrap" &&
        worker_report.bootstrap_zoom_state.next_candidate_index == 2);

    const auto second = job("flydelta://job/worker-2");
    CHECK(common_flydelta_experiment_queue_enqueue(root, second, {}, error));
    CHECK(common_flydelta_experiment_worker_run_once(root, {},
        [](const auto &, auto &, std::string &) { return true; }, worker_report, error));
    CHECK(worker_report.state == common_flydelta_experiment_queue_state::failed);

    CHECK(common_flydelta_experiment_worker_run_once(root, {},
        [](const auto &, auto &, std::string &) { return false; }, worker_report, error));
    CHECK(worker_report.state == common_flydelta_experiment_queue_state::pending);

    common_flydelta_evaluator_config evaluator_config;
    evaluator_config.pipeline.dimension = 2;
    evaluator_config.pipeline.layer.min_cosine = 0.0f;
    evaluator_config.pipeline.scale.max_geometric_trials = 1;
    evaluator_config.pipeline.region_max_stalled_scales = 1;
    evaluator_config.direction.dimension = 2;
    evaluator_config.direction.layer_index = 2;
    evaluator_config.direction.min_samples = 1;
    evaluator_config.direction.max_samples = 4;
    evaluator_config.direction.mode = common_flydelta_direction_search_mode::experimental;
    evaluator_config.direction.source = common_adaptation_evidence_source::tool_repair;
    evaluator_config.direction.behavior_key = "tool_use/diagnostics/missing-argument";
    evaluator_config.direction.model_profile_fingerprint = "sha256:model";
    evaluator_config.direction.execution_context_fingerprint = "sha256:execution-context";
    evaluator_config.direction.capture_layout_revision = "layout:v1";
    common_flydelta_evaluator_callbacks evaluator_callbacks;
    evaluator_callbacks.run_counterfactual = [](const auto &, auto &, auto &) { return false; };
    evaluator_callbacks.resolve_behavior_delta = [](const auto &, auto & delta, auto & credit, auto &) {
        delta = {};
        delta.layer_index = 2;
        delta.values = {1.0f, 0.0f};
        credit = {};
        return false;
    };
    evaluator_callbacks.run_search_pipeline = [](const auto &, auto & value, auto &) {
        value = {};
        common_flydelta_search_pipeline_direction_result direction;
        direction.direction.layer_index = 2;
        direction.direction.values = {1.0f, 0.0f};
        direction.direction.source_samples = 1;
        direction.direction.retained_samples = 1;
        direction.direction.median_alignment = 1.0f;
        common_flydelta_layer_candidate layer;
        layer.layer_indices = {2};
        layer.anchor_layer_index = 2;
        layer.diagnostic_score = 1.0f;
        layer.total_scale = 0.05f;
        layer.per_layer_scale = 0.05f;
        direction.layer_plan.singleton_candidates.push_back(std::move(layer));
        common_flydelta_intervention_region_trial region;
        region.candidate.layer_indices = {2};
        region.candidate.anchor_layer_index = 2;
        region.candidate.total_scale = 0.05f;
        region.candidate.per_layer_scale = 0.05f;
        region.executed = true;
        region.verifier_known = true;
        region.search_score = 0.5f;
        region.promising = true;
        region.safe_to_continue = true;
        region.evidence_ref = "evidence:worker-where";
        direction.region_trials.push_back(std::move(region));
        value.directions.push_back(std::move(direction));
        return true;
    };
    evaluator_callbacks.resolve_training_example = [](const auto &, auto &, auto &) { return false; };
    common_flydelta_experiment_worker_report evaluator_report;
    auto pipeline_job = job("flydelta://job/worker-evaluator");
    pipeline_job.kind = common_flydelta_experiment_job_kind::search_pipeline;
    pipeline_job.capture_manifest_ids = {"flydelta://capture/evaluator"};
    pipeline_job.behavior_delta_ids = {"flydelta://delta/evaluator"};
    common_flydelta_evaluator_result direct_evaluator_result;
    const bool direct_evaluator_succeeded = common_flydelta_evaluate_job(
        pipeline_job, evaluator_config, evaluator_callbacks, direct_evaluator_result, error);
    if (!direct_evaluator_succeeded) std::cerr << "direct evaluator error: " << error << "\n";
    CHECK(direct_evaluator_succeeded);
    CHECK(common_flydelta_experiment_queue_enqueue(root, pipeline_job, {}, error));
    const bool evaluator_succeeded = common_flydelta_experiment_worker_run_evaluator_once(
        root, {}, evaluator_config, evaluator_callbacks, evaluator_report, error);
    if (!evaluator_succeeded) std::cerr << "evaluator worker error: " << error << "\n";
    CHECK(evaluator_succeeded);
    CHECK(evaluator_report.state == common_flydelta_experiment_queue_state::succeeded);
    CHECK(evaluator_report.report_count == 1);

    evaluator_callbacks = {};
    evaluator_callbacks.resolve_behavior_delta = [](const auto &, auto & delta, auto & credit, auto &) {
        delta = behavior_delta();
        credit.experiment_id = "flydelta://experiment/worker";
        credit.candidate_id = "flydelta://candidate/worker";
        credit.fixture_id = "flydelta://fixture/worker";
        credit.outcome = common_flydelta_counterfactual_outcome::unknown;
        credit.quality_delta = 0.0f;
        credit.eligible_for_learning = false;
        return true;
    };
    auto evaluator_direction_job = job("flydelta://job/worker-evaluator-direction");
    evaluator_direction_job.kind = common_flydelta_experiment_job_kind::direction;
    evaluator_direction_job.capture_manifest_ids.clear();
    evaluator_direction_job.behavior_delta_ids = {"flydelta://behavior/worker"};
    CHECK(common_flydelta_experiment_queue_enqueue(root, evaluator_direction_job, {}, error));
    CHECK(common_flydelta_experiment_worker_run_evaluator_once(
        root, {}, evaluator_config, evaluator_callbacks, evaluator_report, error));
    CHECK(evaluator_report.state == common_flydelta_experiment_queue_state::succeeded);
    CHECK(evaluator_report.report_count == 3);
    CHECK(evaluator_report.evidence_depth.depth == common_flydelta_search_depth::bootstrap);
    CHECK(evaluator_report.evidence_depth.compatible_samples == 1);
    CHECK(evaluator_report.search_budget.max_region_trials == 4);
    std::filesystem::remove_all(root, ignored);
    return 0;
}
