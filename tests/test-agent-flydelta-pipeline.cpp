#include "agent/adaptation/flydelta/flydelta-collection.h"
#include "agent/adaptation/flydelta/flydelta-promotion.h"
#include "agent/adaptation/flydelta/flydelta-worker.h"

#include <filesystem>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

static common_adaptation_evidence make_evidence() {
    common_adaptation_evidence value;
    value.id = "evidence://repair/pipeline-1";
    value.source = common_adaptation_evidence_source::tool_repair;
    value.behavior_key = "tool_use/diagnostics/missing-argument";
    value.scope.namespace_id = "local";
    value.scope.project_id = "project";
    value.scope.session_id = "session";
    value.task_fingerprint = "sha256:task";
    value.baseline_ref = "execution:failed";
    value.candidate_ref = "execution:repaired";
    value.verifier_ref = "verifier:pipeline-v1";
    value.transaction_ids = {"learning://failed", "learning://repaired"};
    value.cause = common_learning_cause::model_behavior;
    value.host_verified = true;
    return value;
}

static common_flydelta_experiment_collection_request make_request() {
    common_flydelta_experiment_collection_request value;
    value.enabled = true;
    value.kind = common_flydelta_experiment_job_kind::counterfactual;
    value.evidence = make_evidence();
    value.behavior_key = "tool_use/diagnostics/missing-argument";
    value.model_profile_fingerprint = "sha256:model";
    value.tokenizer_fingerprint = "sha256:tokenizer";
    value.template_fingerprint = "sha256:template";
    value.execution_context_fingerprint = "sha256:execution-context";
    value.capture_manifest_ids = {"flydelta://capture/pipeline-1"};
    value.alpha_search.candidates = {0.05f};
    value.alpha_search.max_candidates = 1;
    value.code_revision = "pipeline-test:v1";
    return value;
}

static common_flydelta_counterfactual_report make_report(
        const std::string & job_id,
        common_flydelta_counterfactual_outcome outcome,
        const std::string & id) {
    common_flydelta_counterfactual_report value;
    value.experiment_id = job_id;
    value.fixture_id = "flydelta://fixture/" + id;
    value.candidate_id = "flydelta://sideband/pipeline-v1";
    value.baseline_profile_id = "base";
    value.candidate_profile_id = "overlay";
    value.baseline.executed = true;
    value.baseline.verifier_known = outcome != common_flydelta_counterfactual_outcome::unknown;
    value.baseline.passed = outcome == common_flydelta_counterfactual_outcome::neutral;
    value.baseline.quality = value.baseline.passed ? 1.0f : 0.0f;
    value.baseline.evidence_ref = value.baseline.verifier_known ? "evidence:base" : "";
    value.candidate.executed = true;
    value.candidate.verifier_known = value.baseline.verifier_known;
    value.candidate.passed = outcome == common_flydelta_counterfactual_outcome::helped ||
        outcome == common_flydelta_counterfactual_outcome::neutral;
    value.candidate.quality = value.candidate.passed ? 1.0f : 0.0f;
    value.candidate.overlay_applied = true;
    value.candidate.evidence_ref = value.candidate.verifier_known ? "evidence:candidate" : "";
    value.outcome = outcome;
    value.quality_delta = value.candidate.quality - value.baseline.quality;
    return value;
}

static common_flydelta_sideband_manifest make_manifest() {
    common_flydelta_sideband_manifest value;
    value.id = "flydelta://sideband/pipeline-v1";
    value.artifact_path = "sidebands/pipeline-v1.flyd";
    value.artifact_hash = "sha256:pipeline-artifact";
    value.compatibility.base_model_fingerprint = "sha256:base";
    value.compatibility.tokenizer_fingerprint = "sha256:tokenizer";
    value.compatibility.template_fingerprint = "sha256:template";
    value.compatibility.architecture = "qwen2";
    value.compatibility.inference_layout_revision = "layout:v1";
    value.model_n_embd = 4;
    value.model_n_layers = 3;
    value.il_end = 2;
    return value;
}

int main() {
    std::string error;
    const auto root = std::filesystem::temp_directory_path() /
        "llama-agent-flydelta-pipeline-test";
    std::error_code ignored;
    std::filesystem::remove_all(root, ignored);

    common_flydelta_experiment_collection_result collection_result;
    CHECK(common_flydelta_collect_experiment_job(
        root, {}, make_request(), collection_result, error));
    CHECK(collection_result == common_flydelta_experiment_collection_result::enqueued);

    std::vector<common_flydelta_counterfactual_report> reports;
    common_flydelta_experiment_worker_report worker_report;
    CHECK(common_flydelta_experiment_worker_run_once(root, {},
        [&](const auto & job, auto & result, std::string &) {
            reports = {
                make_report(job.id, common_flydelta_counterfactual_outcome::helped, "1"),
                make_report(job.id, common_flydelta_counterfactual_outcome::helped, "2"),
                make_report(job.id, common_flydelta_counterfactual_outcome::neutral, "3"),
                make_report(job.id, common_flydelta_counterfactual_outcome::unknown, "4"),
            };
            result.safe_summary = "pipeline counterfactuals completed";
            result.counterfactual_reports = reports;
            return true;
        }, worker_report, error));
    CHECK(worker_report.state == common_flydelta_experiment_queue_state::succeeded);
    CHECK(reports.size() == 4);

    common_flydelta_promotion_policy policy;
    policy.min_trials = 4;
    policy.min_known_trials = 3;
    policy.min_helped_trials = 2;
    policy.min_help_confidence = 0.60f;
    common_flydelta_promotion_summary summary;
    CHECK(common_flydelta_promotion_summary_from_reports(
        "flydelta://promotion/pipeline-v1", reports, policy, summary, error));
    CHECK(summary.status == common_flydelta_candidate_status::eligible);

    auto evaluation = common_flydelta_evaluation_report{};
    evaluation.revision_id = "flydelta-evaluation:pipeline-v1";
    evaluation.candidate_id = "flydelta://sideband/pipeline-v1";
    evaluation.baseline_profile_id = "base";
    evaluation.candidate_profile_id = "overlay";
    evaluation.test_suite_revision = "agent-suite:pipeline-v1";
    evaluation.intended_behavior_passed = true;
    evaluation.retention_passed = true;
    evaluation.agent_regression_passed = true;
    evaluation.evaluated_turns = 4;
    evaluation.baseline_successes = 2;
    evaluation.candidate_successes = 4;
    evaluation.status = "passed";
    common_flydelta_sideband_registry registry;
    CHECK(common_flydelta_promote_verified_sideband(
        registry, make_manifest(), summary, evaluation, policy, true, error));
    CHECK(registry.list().at("flydelta://sideband/pipeline-v1").status ==
        common_flydelta_sideband_status::canary);

    std::filesystem::remove_all(root, ignored);
    return 0;
}
