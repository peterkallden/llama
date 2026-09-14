#include "agent/adaptation/flydelta/flydelta-evidence.h"

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

static common_learning_transaction transaction(
        const std::string & id,
        common_learning_signal_type signal_type) {
    common_learning_transaction value;
    value.id = id;
    value.created_at = "2026-09-13T00:00:00Z";
    value.observation.id = id;
    value.observation.scope.namespace_id = "local";
    value.observation.scope.project_id = "project";
    value.observation.scope.session_id = "session";
    value.observation.source_turn_id = id + ":turn";
    value.observation.source_plan_id = id + ":plan";
    value.observation.signals.push_back({signal_type, value.observation.source_plan_id,
        "step", "data.inspect", id + ":evidence", "verified", "diagnostics", "native"});
    value.observation.evidence_ids = {id + ":evidence"};
    value.observation.cause = common_learning_cause::model_behavior;
    value.observation.verification = common_learning_verification::host_verified;
    value.observation.idempotency_key = id + ":key";
    value.observation.content_hash = "sha256:" + id;
    value.observation.collection_allowed = true;
    return value;
}

int main() {
    std::string error;
    auto failed = transaction("learning://transaction/failed", common_learning_signal_type::tool_failure);
    auto repaired = transaction("learning://transaction/repaired", common_learning_signal_type::successful_recovery);
    common_flydelta_repair_transition transition;
    CHECK(common_flydelta_repair_transition_from_transactions(
        failed, repaired, "sha256:task", "execution:failed", "execution:repaired",
        "verifier:v1", transition, error));

    common_flydelta_contrast_set contrast_set;
    CHECK(common_flydelta_contrast_set_from_transitions(
        "flydelta://contrast/1", "missing-required-tool-argument", {transition}, 8,
        contrast_set, error));
    CHECK(contrast_set.positive_transaction_ids[0] == repaired.id);
    CHECK(contrast_set.negative_transaction_ids[0] == failed.id);

    common_flydelta_experiment_fixture fixture;
    fixture.id = "flydelta://fixture/1";
    fixture.task_fingerprint = "sha256:task";
    fixture.model_profile_fingerprint = "sha256:model";
    fixture.tokenizer_fingerprint = "sha256:tokenizer";
    fixture.template_fingerprint = "sha256:template";
    fixture.execution_context_fingerprint = "sha256:execution-context";
    fixture.verifier_revision = "verifier:v1";
    common_flydelta_counterfactual_report report;
    report.experiment_id = "flydelta://experiment/1";
    report.fixture_id = fixture.id;
    report.candidate_id = "flydelta://candidate/1";
    report.baseline_profile_id = "base";
    report.candidate_profile_id = "overlay";
    report.baseline.executed = true;
    report.baseline.verifier_known = true;
    report.baseline.passed = false;
    report.baseline.evidence_ref = "evidence:failed";
    report.candidate.executed = true;
    report.candidate.verifier_known = true;
    report.candidate.passed = true;
    report.candidate.overlay_applied = true;
    report.candidate.quality = 1.0f;
    report.candidate.evidence_ref = "evidence:passed";
    report.outcome = common_flydelta_counterfactual_outcome::helped;
    report.quality_delta = 1.0f;
    common_flydelta_intervention_credit credit;
    CHECK(common_flydelta_intervention_credit_from_report(report, credit, error));
    CHECK(credit.eligible_for_learning);

    report.outcome = common_flydelta_counterfactual_outcome::unknown;
    credit.outcome = common_flydelta_counterfactual_outcome::unknown;
    credit.eligible_for_learning = true;
    CHECK(!common_flydelta_intervention_credit_validate(credit, error));

    common_adaptation_evidence source;
    source.id = "evidence://tool-repair/1";
    source.source = common_adaptation_evidence_source::tool_repair;
    source.scope.namespace_id = "local";
    source.scope.project_id = "project";
    source.scope.session_id = "session";
    source.task_fingerprint = "sha256:task";
    source.baseline_ref = "execution:failed";
    source.candidate_ref = "execution:repaired";
    source.verifier_ref = "verifier:v1";
    source.transaction_ids = {failed.id, repaired.id};
    source.host_verified = true;
    common_flydelta_experiment_seed seed;
    CHECK(common_flydelta_experiment_seed_from_evidence(
        source, "tool_use/diagnostics/missing-argument", "sha256:model",
        "sha256:tokenizer", "sha256:template", "sha256:tool-resource-context",
        common_flydelta_training_split::holdout, seed, error));
    CHECK(seed.split == common_flydelta_training_split::holdout);
    common_flydelta_experiment_fixture seed_fixture;
    CHECK(common_flydelta_experiment_fixture_from_seed(seed, seed_fixture, error));
    CHECK(seed_fixture.verifier_revision == source.verifier_ref);
    source.host_verified = false;
    CHECK(!common_flydelta_experiment_seed_from_evidence(
        source, "tool_use/diagnostics/missing-argument", "sha256:model",
        "sha256:tokenizer", "sha256:template", "sha256:tool-resource-context",
        common_flydelta_training_split::train, seed, error));
    return 0;
}
