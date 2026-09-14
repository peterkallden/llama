#include "agent/adaptation/flydelta/flydelta-sideband-controller.h"

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

static common_flydelta_counterfactual_report report(const std::string & id) {
    common_flydelta_counterfactual_report value;
    value.experiment_id = "flydelta://experiment/" + id;
    value.fixture_id = "flydelta://fixture/" + id;
    value.candidate_id = "flydelta://sideband/controller-v1";
    value.baseline_profile_id = "base";
    value.candidate_profile_id = "overlay";
    value.baseline.executed = true;
    value.baseline.verifier_known = true;
    value.baseline.passed = false;
    value.baseline.evidence_ref = "evidence:base-" + id;
    value.candidate.executed = true;
    value.candidate.verifier_known = true;
    value.candidate.passed = true;
    value.candidate.overlay_applied = true;
    value.candidate.evidence_ref = "evidence:candidate-" + id;
    value.outcome = common_flydelta_counterfactual_outcome::helped;
    value.quality_delta = 1.0f;
    return value;
}

static common_flydelta_sideband_manifest manifest() {
    common_flydelta_sideband_manifest value;
    value.id = "flydelta://sideband/controller-v1";
    value.artifact_path = "sidebands/controller-v1.flyd";
    value.artifact_hash = "sha256:controller-artifact";
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
    common_flydelta_promotion_policy policy;
    policy.min_trials = 3;
    policy.min_known_trials = 3;
    policy.min_helped_trials = 3;
    policy.min_help_confidence = 1.0f;
    policy.max_unknown_ratio = 0.0f;
    std::vector<common_flydelta_counterfactual_report> reports = {
        report("1"), report("2"), report("3")};
    common_flydelta_promotion_summary summary;
    CHECK(common_flydelta_promotion_summary_from_reports(
        "flydelta://promotion/controller", reports, policy, summary, error));
    CHECK(summary.status == common_flydelta_candidate_status::eligible);

    common_flydelta_evaluation_report evaluation;
    evaluation.revision_id = "flydelta-evaluation:controller-v1";
    evaluation.candidate_id = manifest().id;
    evaluation.baseline_profile_id = "base";
    evaluation.candidate_profile_id = "overlay";
    evaluation.test_suite_revision = "agent-suite:v1";
    evaluation.intended_behavior_passed = true;
    evaluation.retention_passed = true;
    evaluation.agent_regression_passed = true;
    evaluation.evaluated_turns = 3;
    evaluation.baseline_successes = 0;
    evaluation.candidate_successes = 3;
    evaluation.status = "passed";

    common_flydelta_sideband_registry registry;
    common_flydelta_sideband_controller controller(registry);
    const auto candidate = manifest();
    CHECK(!controller.promote_to_canary(
        candidate, summary, evaluation, policy, false, error));
    CHECK(registry.list().empty());
    CHECK(controller.promote_to_canary(
        candidate, summary, evaluation, policy, true, error));
    CHECK(registry.list().at(candidate.id).status == common_flydelta_sideband_status::canary);
    CHECK(!controller.activate(candidate.id, false, error));
    CHECK(controller.activate(candidate.id, true, error));
    CHECK(registry.list().at(candidate.id).status == common_flydelta_sideband_status::active);
    CHECK(controller.retire(candidate.id, error));
    CHECK(registry.list().at(candidate.id).status == common_flydelta_sideband_status::retired);
    return 0;
}
