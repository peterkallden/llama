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
        result.evaluation_revision = "evaluation:" + result.manifest.id;
    }
    if (action == common_flydelta_review_action::revoke) result.reason = "test rollback";
    return result;
}

static common_flydelta_promotion_summary summary_for(const std::string & candidate_id) {
    common_flydelta_promotion_summary value;
    value.id = "promotion:" + candidate_id;
    value.candidate_id = candidate_id;
    value.baseline_profile_id = "profile:base";
    value.candidate_profile_id = "profile:candidate";
    value.total_trials = 3;
    value.known_trials = 3;
    value.helped_trials = 3;
    value.help_confidence = 1.0f;
    value.status = common_flydelta_candidate_status::eligible;
    return value;
}

static common_flydelta_evaluation_report evaluation_for(const std::string & candidate_id) {
    common_flydelta_evaluation_report value;
    value.revision_id = "evaluation:" + candidate_id;
    value.candidate_id = candidate_id;
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

static common_flydelta_promotion_summary summary() {
    return summary_for("flydelta://sideband/review-store-v1");
}

static common_flydelta_evaluation_report evaluation() {
    return evaluation_for("flydelta://sideband/review-store-v1");
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

    auto value2 = manifest();
    value2.id = "flydelta://sideband/review-store-v2";
    value2.artifact_path = "sidebands/review-store-v2.flyd";
    value2.artifact_hash = "sha256:review-store-artifact-v2";
    CHECK(store.apply_and_append(registry,
        review("review-5", common_flydelta_review_action::admit_experimental, value2), true, error));
    CHECK(store.apply_and_append(registry,
        review("review-6", common_flydelta_review_action::promote_to_candidate, value2), true, error));
    value2.status = common_flydelta_sideband_status::candidate;
    auto canary_review2 = review("review-7", common_flydelta_review_action::stage_canary, value2);
    canary_review2.evaluation_revision = "evaluation:" + value2.id;
    canary_review2.has_promotion_evidence = true;
    canary_review2.promotion_summary = summary_for(value2.id);
    canary_review2.promotion_policy.min_trials = 3;
    canary_review2.promotion_policy.min_known_trials = 3;
    canary_review2.promotion_policy.min_helped_trials = 3;
    canary_review2.promotion_policy.min_help_confidence = 1.0f;
    canary_review2.promotion_policy.max_unknown_ratio = 0.0f;
    canary_review2.evaluation = evaluation_for(value2.id);
    CHECK(store.apply_and_append(registry, canary_review2, true, error));
    value2.status = common_flydelta_sideband_status::canary;
    auto activate_review2 = review("review-8", common_flydelta_review_action::activate, value2);
    activate_review2.binding_key = "flydelta://binding/review-store";
    CHECK(store.apply_and_append(registry, activate_review2, true, error));
    common_flydelta_activation_binding binding;
    CHECK(registry.binding("flydelta://binding/review-store", binding, error));
    CHECK(binding.selected_revision_id == value2.id);

    value.status = common_flydelta_sideband_status::active;
    value.evaluation_passed = true;
    value.evaluation_revision = "evaluation:" + value.id;
    auto rollback_review = review("review-9", common_flydelta_review_action::rollback, value);
    rollback_review.binding_key = "flydelta://binding/review-store";
    rollback_review.expected_current_revision_id = value2.id;
    CHECK(store.apply_and_append(registry, rollback_review, true, error));
    CHECK(registry.binding("flydelta://binding/review-store", binding, error));
    CHECK(binding.selected_revision_id == value.id && binding.previous_revision_id == value2.id);
    CHECK(store.list(error).size() == 9);

    common_flydelta_sideband_registry restored;
    CHECK(store.replay(restored, error));
    CHECK(restored.list().at(value.id).status == common_flydelta_sideband_status::active);
    CHECK(restored.list().at(value2.id).status == common_flydelta_sideband_status::active);
    CHECK(restored.binding("flydelta://binding/review-store", binding, error));
    CHECK(binding.selected_revision_id == value.id);
    return 0;
}
