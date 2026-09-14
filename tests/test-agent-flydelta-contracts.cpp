#include "agent/adaptation/flydelta/flydelta-contracts.h"

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

static common_learning_transaction transaction(
        const std::string & id,
        common_learning_signal_type signal_type,
        common_learning_cause cause = common_learning_cause::model_behavior) {
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
    value.observation.cause = cause;
    value.observation.verification = common_learning_verification::host_verified;
    value.observation.idempotency_key = id + ":key";
    value.observation.content_hash = "sha256:" + id;
    value.observation.collection_allowed = true;
    return value;
}

int main() {
    std::string error;
    common_flydelta_capture_manifest manifest;
    manifest.id = "flydelta://capture/1";
    manifest.observation_id = "learning://observation/1";
    manifest.behavior_key = "tool_use/diagnostics/missing-argument";
    manifest.model_profile_fingerprint = "sha256:model";
    manifest.template_fingerprint = "sha256:template";
    manifest.positive_execution_ref = "execution:positive";
    manifest.negative_execution_ref = "execution:negative";
    manifest.capture_layout_revision = "l_out:v1";
    manifest.evidence_hash = "sha256:evidence";
    manifest.redaction_attested = true;
    manifest.captured_bytes = 128;
    CHECK(common_flydelta_capture_manifest_validate(manifest, 1024, error));
    common_flydelta_capture_manifest parsed_manifest;
    CHECK(common_flydelta_capture_manifest_from_json(
        common_flydelta_capture_manifest_to_json(manifest), 1024, parsed_manifest, error));
    manifest.redaction_attested = false;
    CHECK(!common_flydelta_capture_manifest_validate(manifest, 1024, error));

    common_flydelta_static_overlay overlay;
    overlay.enabled = true;
    overlay.artifact_id = "flydelta://artifact/1";
    overlay.n_embd = 4;
    overlay.il_start = 1;
    overlay.il_end = 2;
    overlay.data.assign(8, 0.25f);
    CHECK(common_flydelta_static_overlay_validate(overlay, 4, 3, 1024, error));
    overlay.data[0] = 0.0f / 0.0f;
    CHECK(!common_flydelta_static_overlay_validate(overlay, 4, 3, 1024, error));
    overlay.data[0] = 0.25f;
    overlay.scale = 2.0f;
    CHECK(!common_flydelta_static_overlay_validate(overlay, 4, 3, 1024, error));

    common_flydelta_candidate_policy policy;
    policy.min_observations = 3;
    policy.min_verified_observations = 2;
    std::vector<common_learning_transaction> transactions = {
        transaction("learning://transaction/1", common_learning_signal_type::successful_recovery),
        transaction("learning://transaction/2", common_learning_signal_type::successful_recovery),
        transaction("learning://transaction/3", common_learning_signal_type::successful_recovery),
        transaction("learning://transaction/4", common_learning_signal_type::successful_recovery),
        transaction("learning://transaction/5", common_learning_signal_type::tool_failure),
    };
    common_flydelta_candidate candidate;
    CHECK(common_flydelta_candidate_from_transactions(transactions, policy, candidate, error));
    CHECK(candidate.verified_observations == 5);
    CHECK(candidate.source == common_adaptation_evidence_source::tool_repair);
    CHECK(candidate.tool_family == "diagnostics");

    auto planning = transactions;
    for (auto & value : planning) {
        value.observation.signals.clear();
        value.observation.signals.push_back({common_learning_signal_type::planning_revision,
            value.observation.source_plan_id, "step", {}, value.id + ":planning-evidence",
            "verified"});
    }
    CHECK(common_flydelta_candidate_from_transactions(planning, policy, candidate, error));
    CHECK(candidate.source == common_adaptation_evidence_source::planning_revision);
    CHECK(candidate.verified_observations == planning.size());
    auto mixed_sources = transactions;
    mixed_sources.front().observation.signals.push_back({common_learning_signal_type::planning_revision,
        "plan", "step", {}, "planning-evidence", "revised"});
    CHECK(!common_flydelta_candidate_from_transactions(mixed_sources, policy, candidate, error));
    transactions[0].observation.cause = common_learning_cause::host_contract;
    CHECK(!common_flydelta_candidate_from_transactions(transactions, policy, candidate, error));

    common_flydelta_evaluation_report report;
    report.revision_id = "flydelta-eval:1";
    report.candidate_id = "flydelta://candidate/1";
    report.baseline_profile_id = "base";
    report.candidate_profile_id = "flydelta-canary";
    report.test_suite_revision = "fixture:1";
    report.intended_behavior_passed = true;
    report.retention_passed = true;
    report.agent_regression_passed = true;
    report.evaluated_turns = 10;
    report.baseline_successes = 7;
    report.candidate_successes = 8;
    report.candidate_interventions = 4;
    report.false_interventions = 1;
    report.status = "passed";
    CHECK(common_flydelta_evaluation_report_validate(report, error));
    common_flydelta_evaluation_report parsed_report;
    CHECK(common_flydelta_evaluation_report_from_json(
        common_flydelta_evaluation_report_to_json(report), parsed_report, error));
    report.false_interventions = 5;
    CHECK(!common_flydelta_evaluation_report_validate(report, error));
    return 0;
}
