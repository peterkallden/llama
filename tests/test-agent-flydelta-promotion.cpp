#include "agent/adaptation/flydelta/flydelta-promotion.h"

#include <string>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

static common_flydelta_counterfactual_report report(
        common_flydelta_counterfactual_outcome outcome,
        const std::string & id) {
    common_flydelta_counterfactual_report value;
    value.experiment_id = "flydelta://experiment/" + id;
    value.fixture_id = "flydelta://fixture/" + id;
    value.candidate_id = "flydelta://candidate/1";
    value.baseline_profile_id = "base";
    value.candidate_profile_id = "overlay";
    value.baseline.executed = true;
    value.baseline.verifier_known = outcome != common_flydelta_counterfactual_outcome::unknown;
    value.baseline.passed = outcome == common_flydelta_counterfactual_outcome::harmed ||
        outcome == common_flydelta_counterfactual_outcome::neutral;
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

int main() {
    std::string error;
    common_flydelta_promotion_policy policy;
    policy.min_trials = 4;
    policy.min_known_trials = 3;
    policy.min_helped_trials = 2;
    policy.min_help_confidence = 0.60f;
    policy.max_unknown_ratio = 0.25f;
    std::vector<common_flydelta_counterfactual_report> reports = {
        report(common_flydelta_counterfactual_outcome::helped, "1"),
        report(common_flydelta_counterfactual_outcome::helped, "2"),
        report(common_flydelta_counterfactual_outcome::neutral, "3"),
        report(common_flydelta_counterfactual_outcome::unknown, "4"),
    };
    common_flydelta_promotion_summary summary;
    CHECK(common_flydelta_promotion_summary_from_reports(
        "flydelta://promotion/1", reports, policy, summary, error));
    CHECK(summary.status == common_flydelta_candidate_status::eligible);
    CHECK(summary.helped_trials == 2 && summary.unknown_trials == 1);
    CHECK(summary.help_confidence > 0.6f);

    reports.push_back(report(common_flydelta_counterfactual_outcome::harmed, "5"));
    CHECK(common_flydelta_promotion_summary_from_reports(
        "flydelta://promotion/2", reports, policy, summary, error));
    CHECK(summary.status == common_flydelta_candidate_status::observed);
    reports.pop_back();
    reports[0].candidate_id = "flydelta://candidate/other";
    CHECK(!common_flydelta_promotion_summary_from_reports(
        "flydelta://promotion/3", reports, policy, summary, error));

    // Promotion requires both an independently passed evaluation and an
    // explicit host approval, then stops at canary rather than active.
    common_flydelta_promotion_summary eligible = {};
    reports[0].candidate_id = "flydelta://sideband/promotion-v1";
    reports[1].candidate_id = reports[0].candidate_id;
    reports[2].candidate_id = reports[0].candidate_id;
    reports[3].candidate_id = reports[0].candidate_id;
    CHECK(common_flydelta_promotion_summary_from_reports(
        "flydelta://promotion/verified", reports, policy, eligible, error));
    common_flydelta_sideband_manifest manifest;
    manifest.id = reports[0].candidate_id;
    manifest.artifact_path = "sidebands/promotion-v1.flyd";
    manifest.artifact_hash = "sha256:promotion-artifact";
    manifest.compatibility.base_model_fingerprint = "sha256:base";
    manifest.compatibility.tokenizer_fingerprint = "sha256:tokenizer";
    manifest.compatibility.template_fingerprint = "sha256:template";
    manifest.compatibility.architecture = "qwen2";
    manifest.compatibility.inference_layout_revision = "layout:v1";
    manifest.model_n_embd = 4;
    manifest.model_n_layers = 3;
    manifest.il_end = 2;
    common_flydelta_evaluation_report evaluation;
    evaluation.revision_id = "flydelta-evaluation:v1";
    evaluation.candidate_id = manifest.id;
    evaluation.baseline_profile_id = "base";
    evaluation.candidate_profile_id = "overlay";
    evaluation.test_suite_revision = "agent-suite:v1";
    evaluation.intended_behavior_passed = true;
    evaluation.retention_passed = true;
    evaluation.agent_regression_passed = true;
    evaluation.evaluated_turns = 4;
    evaluation.baseline_successes = 2;
    evaluation.candidate_successes = 4;
    evaluation.status = "passed";
    common_flydelta_sideband_registry registry;
    CHECK(!common_flydelta_promote_verified_sideband(
        registry, manifest, eligible, evaluation, policy, false, error));
    CHECK(registry.list().empty());
    CHECK(common_flydelta_promote_verified_sideband(
        registry, manifest, eligible, evaluation, policy, true, error));
    CHECK(registry.list().at(manifest.id).status == common_flydelta_sideband_status::canary);
    CHECK(!registry.list().at(manifest.id).evaluation_revision.empty());
    CHECK(registry.activate(manifest.id, error));
    CHECK(registry.list().at(manifest.id).status == common_flydelta_sideband_status::active);

    return 0;
}
