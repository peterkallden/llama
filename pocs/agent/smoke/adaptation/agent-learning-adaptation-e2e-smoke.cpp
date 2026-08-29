#include "tools/agent/adaptation/agent-learning-adaptation-orchestrator.h"

#include "agent/adaptation/learning-observation.h"

#include <cassert>
#include <filesystem>
#include <iostream>

namespace {

common_learning_transaction make_transaction(const std::string & id) {
    common_learning_transaction value;
    value.id = id;
    value.created_at = "2026-08-29T00:00:00Z";
    value.observation.id = id;
    value.observation.scope.namespace_id = "local";
    value.observation.scope.session_id = "adaptation-smoke";
    value.observation.scope.turn_id = id;
    value.observation.source_turn_id = id;
    value.observation.source_plan_id = "plan:smoke";
    value.observation.signals.push_back({
        common_learning_signal_type::successful_recovery,
        "plan:smoke", "step:recover", "data.inspect", "evidence:smoke",
        "host verified the repaired action"});
    value.observation.evidence_ids = {"evidence:smoke"};
    value.observation.cause = common_learning_cause::model_behavior;
    value.observation.verification = common_learning_verification::host_verified;
    value.observation.idempotency_key = id + ":recovery";
    value.observation.content_hash = common_learning_observation_hash(value.observation);
    value.observation.collection_allowed = true;
    return value;
}

common_training_candidate make_candidate(const std::string & transaction_id) {
    common_training_candidate value;
    value.id = "learning://candidate/adaptation-smoke";
    value.transaction_ids = {transaction_id};
    value.cause = common_learning_cause::model_behavior;
    value.hypothesis = "The model needs to prefer the verified data inspection action.";
    value.approved_prompt = "Inspect the selected dataset and report its columns.";
    value.approved_target = "Use data.inspect on the selected dataset before answering.";
    value.observed_occurrences = 3;
    value.verified_recoveries = 2;
    value.confidence = 0.95f;
    value.redaction_policy_id = "stub:caller-asserted-v1";
    value.redaction_method = "smoke-fixture-no-sensitive-data";
    value.redaction_status = common_learning_redaction_status::caller_asserted;
    value.status = common_training_candidate_status::approved;
    return value;
}

} // namespace

int main() {
    const auto root = std::filesystem::temp_directory_path() / "llama-agent-adaptation-e2e-smoke";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);

    common_learning_in_memory_transaction_store transactions;
    common_learning_in_memory_lifecycle_store lifecycle;
    const auto transaction = make_transaction("learning://transaction/adaptation-smoke");
    std::string error;
    assert(common_learning_transaction_validate(transaction, 16, error));
    assert(transactions.append(transaction, error));

    agent_learning_adaptation_orchestrator_config config{
        transactions,
        lifecycle,
        root / "queue",
    };
    config.evaluator = [](const common_learning_training_job & job,
            const common_learning_training_result & result,
            common_learning_evaluation_report & report,
            std::string & error) {
        error.clear();
        report.revision_id = result.evaluation_revision;
        report.candidate_adapter_id = result.adapter_id;
        report.baseline_profile_id = "profile:baseline";
        report.candidate_profile_id = "profile:candidate";
        report.corpus_revision_id = job.corpus_revision_id;
        report.corpus_bundle_hash = job.corpus_bundle_hash;
        report.base_training_fingerprint = job.base_training_fingerprint;
        report.test_suite_revision = "smoke:evaluation-fixture-v1";
        report.intended_behavior_passed = true;
        report.retention_passed = true;
        report.agent_regression_passed = true;
        report.evaluated_turns = 3;
        report.runtime_interventions = 1;
        report.status = "passed";
        report.evaluated_at = "2026-08-29T00:00:00Z";
        return true;
    };

    agent_learning_adaptation_orchestrator_result result;
    assert(agent_learning_run_adaptation(
            make_candidate(transaction.id), config, result, error));
    assert(error.empty());
    assert(result.worker_report.state == agent_learning_worker_job_state::succeeded);
    assert(result.training_result.evaluation_status == "not_run");
    assert(result.evaluation.status == "passed");
    assert(result.manifest.status == common_learning_adapter_status::candidate);
    assert(std::filesystem::is_regular_file(result.verified_artifact_path));
    assert(result.profile.adapters.size() == 1);
    assert(result.lifecycle_records == 8);

    const auto lifecycle_records = lifecycle.list(error);
    assert(error.empty());
    assert(lifecycle_records.size() == result.lifecycle_records);
    assert(lifecycle_records.front().status == common_learning_lifecycle_status::approved);
    assert(lifecycle_records.back().status == common_learning_lifecycle_status::active);

    std::cout << "adaptation_orchestration=passed\n"
              << "worker_state=succeeded\n"
              << "training_evaluation=not_run\n"
              << "evaluation=passed\n"
              << "lifecycle_records=" << lifecycle_records.size() << '\n';
    std::filesystem::remove_all(root, ec);
    return 0;
}
