#include "agent/adaptation/flydelta/flydelta-sideband-review-store.h"

#include <string>
#include <utility>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

static common_flydelta_sideband_manifest manifest() {
    common_flydelta_sideband_manifest value;
    value.id = "flydelta://sideband/review-store-v1";
    value.status = common_flydelta_sideband_status::experimental;
    value.artifact_path = "sidebands/review-store-v1.flyd";
    value.artifact_hash = "sha256:review-store-artifact";
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

static common_flydelta_sideband_review review(
        const std::string & id, common_flydelta_review_action action,
        common_flydelta_sideband_manifest value) {
    common_flydelta_sideband_review result;
    result.event_id = id;
    result.actor_id = "operator:test";
    result.action = action;
    result.manifest = std::move(value);
    if (action == common_flydelta_review_action::promote_to_candidate ||
            action == common_flydelta_review_action::stage_canary) {
        result.evaluation_revision = "evaluation:review-store-v1";
    }
    if (action == common_flydelta_review_action::revoke) result.reason = "test rollback";
    return result;
}

static common_flydelta_promotion_summary summary() {
    common_flydelta_promotion_summary value;
    value.id = "promotion:review-store-v1";
    value.candidate_id = "flydelta://sideband/review-store-v1";
    value.baseline_profile_id = "profile:base";
    value.candidate_profile_id = "profile:candidate";
    value.total_trials = 3;
    value.known_trials = 3;
    value.helped_trials = 3;
    value.help_confidence = 1.0f;
    value.status = common_flydelta_candidate_status::eligible;
    return value;
}

static common_flydelta_evaluation_report evaluation() {
    common_flydelta_evaluation_report value;
    value.revision_id = "evaluation:review-store-v1";
    value.candidate_id = "flydelta://sideband/review-store-v1";
    value.baseline_profile_id = "profile:base";
    value.candidate_profile_id = "profile:candidate";
    value.test_suite_revision = "suite:review-store-v1";
    value.intended_behavior_passed = true;
    value.retention_passed = true;
    value.agent_regression_passed = true;
    value.evaluated_turns = 3;
    value.candidate_successes = 3;
    value.status = "passed";
    return value;
}

int main() {
    std::string error;
    common_learning_in_memory_lifecycle_store journal;
    common_flydelta_sideband_review_store store(journal);
    common_flydelta_sideband_registry registry;
    auto value = manifest();

    CHECK(store.apply_and_append(registry,
        review("review-1", common_flydelta_review_action::admit_experimental, value), true, error));
    CHECK(store.apply_and_append(registry,
        review("review-2", common_flydelta_review_action::promote_to_candidate, value), true, error));
    value.status = common_flydelta_sideband_status::candidate;
    auto canary_review = review("review-3", common_flydelta_review_action::stage_canary, value);
    canary_review.has_promotion_evidence = true;
    canary_review.promotion_summary = summary();
    canary_review.promotion_policy.min_trials = 3;
    canary_review.promotion_policy.min_known_trials = 3;
    canary_review.promotion_policy.min_helped_trials = 3;
    canary_review.promotion_policy.min_help_confidence = 1.0f;
    canary_review.promotion_policy.max_unknown_ratio = 0.0f;
    canary_review.evaluation = evaluation();
    CHECK(store.apply_and_append(registry, canary_review, true, error));
    value.status = common_flydelta_sideband_status::canary;
    CHECK(store.apply_and_append(registry,
        review("review-4", common_flydelta_review_action::activate, value), true, error));
    CHECK(registry.list().at(value.id).status == common_flydelta_sideband_status::active);
    CHECK(store.list(error).size() == 4);

    common_flydelta_sideband_registry restored;
    CHECK(store.replay(restored, error));
    CHECK(restored.list().at(value.id).status == common_flydelta_sideband_status::active);
    return 0;
}
