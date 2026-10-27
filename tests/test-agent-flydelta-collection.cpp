#include "agent/adaptation/flydelta/flydelta-collection.h"
#include "agent/adaptation/flydelta/flydelta-search-budget.h"
#include "agent/adaptation/flydelta/flydelta-worker-commit.h"
#include "hash/hash.h"

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

static common_adaptation_evidence evidence() {
    common_adaptation_evidence value;
    value.id = "evidence://repair/collector-1";
    value.source = common_adaptation_evidence_source::tool_repair;
    value.behavior_key = "tool_use/diagnostics/missing-argument";
    value.scope.namespace_id = "local";
    value.scope.project_id = "project";
    value.scope.session_id = "session";
    value.scope.turn_id = "turn";
    value.task_fingerprint = "sha256:task";
    value.baseline_ref = "execution:failed";
    value.candidate_ref = "execution:repaired";
    value.verifier_ref = "verifier:v1";
    value.transaction_ids = {"learning://failed", "learning://repaired"};
    value.cause = common_learning_cause::model_behavior;
    value.host_verified = true;
    return value;
}

static common_flydelta_experiment_collection_request request() {
    common_flydelta_experiment_collection_request value;
    value.enabled = true;
    value.kind = common_flydelta_experiment_job_kind::counterfactual;
    value.evidence = evidence();
    value.behavior_key = "tool_use/diagnostics/missing-argument";
    value.model_profile_fingerprint = "sha256:model";
    value.tokenizer_fingerprint = "sha256:tokenizer";
    value.template_fingerprint = "sha256:template";
    value.execution_context_fingerprint = "sha256:execution-context";
    value.capture_manifest_ids = {"flydelta://capture/manifest-1"};
    value.alpha_search.candidates = {0.05f, 0.1f};
    value.alpha_search.max_candidates = 2;
    value.code_revision = "collector-test:v1";
    return value;
}

int main() {
    std::string error;
    const auto root = std::filesystem::temp_directory_path() /
        "llama-agent-flydelta-collection-test";
    std::error_code ignored;
    std::filesystem::remove_all(root, ignored);

    auto value = request();
    common_flydelta_experiment_collection_result result;
    CHECK(common_flydelta_collect_experiment_job(root, {}, value, result, error));
    CHECK(result == common_flydelta_experiment_collection_result::enqueued);
    CHECK(std::filesystem::exists(root / "pending"));

    CHECK(common_flydelta_collect_experiment_job(root, {}, value, result, error));
    CHECK(result == common_flydelta_experiment_collection_result::already_present);

    common_flydelta_claimed_experiment_job claimed;
    CHECK(common_flydelta_experiment_queue_claim_next(root, {}, claimed, error));
    CHECK(claimed.job.seed.scope.project_id == "project");
    CHECK(claimed.job.seed.scope.turn_id == "turn");
    CHECK(claimed.job.kind == common_flydelta_experiment_job_kind::counterfactual);
    CHECK(claimed.job.capture_manifest_ids.size() == 1);

    auto pipeline_request = request();
    pipeline_request.kind = common_flydelta_experiment_job_kind::search_pipeline;
    pipeline_request.behavior_delta_ids = {"flydelta://delta/pipeline-1"};
    pipeline_request.bootstrap_zoom_state_ref = "flydelta://state/bootstrap-zoom/seed";
    pipeline_request.search_state_ref = "flydelta://state/search/plateau-1";
    pipeline_request.representation_augmentation_state_ref =
        "flydelta://state/augmentation/seed";
    const auto pipeline_root = root / "pipeline";
    CHECK(common_flydelta_collect_experiment_job(
        pipeline_root, {}, pipeline_request, result, error));
    CHECK(result == common_flydelta_experiment_collection_result::enqueued);
    common_flydelta_claimed_experiment_job pipeline_claimed;
    CHECK(common_flydelta_experiment_queue_claim_next(pipeline_root, {}, pipeline_claimed, error));
    CHECK(pipeline_claimed.job.kind == common_flydelta_experiment_job_kind::search_pipeline);
    CHECK(pipeline_claimed.job.capture_manifest_ids.size() == 1);
    CHECK(pipeline_claimed.job.behavior_delta_ids.size() == 1);
    CHECK(pipeline_claimed.job.bootstrap_zoom_state_ref ==
        "flydelta://state/bootstrap-zoom/seed");
    CHECK(pipeline_claimed.job.search_state_ref ==
        "flydelta://state/search/plateau-1");
    CHECK(pipeline_claimed.job.representation_augmentation_state_ref ==
        "flydelta://state/augmentation/seed");

    CHECK(common_flydelta_collect_next_action_job(
        pipeline_root, {}, pipeline_claimed.job,
        common_flydelta_next_action::run_orthogonal_search,
        "flydelta://state/bootstrap-zoom/next",
        "flydelta://state/search/orthogonal",
        "flydelta://state/augmentation/next",
        "",
        result, error));
    CHECK(result == common_flydelta_experiment_collection_result::enqueued);
    CHECK(common_flydelta_collect_next_action_job(
        pipeline_root, {}, pipeline_claimed.job,
        common_flydelta_next_action::run_orthogonal_search,
        "flydelta://state/bootstrap-zoom/next",
        "flydelta://state/search/orthogonal",
        "flydelta://state/augmentation/next",
        "",
        result, error));
    CHECK(result == common_flydelta_experiment_collection_result::already_present);
    common_flydelta_claimed_experiment_job next_action_job;
    CHECK(common_flydelta_experiment_queue_claim_next(
        pipeline_root, {}, next_action_job, error));
    CHECK(next_action_job.job.kind == common_flydelta_experiment_job_kind::search_pipeline);
    CHECK(next_action_job.job.search_state_ref ==
        "flydelta://state/search/orthogonal");
    CHECK(next_action_job.job.representation_augmentation_state_ref ==
        "flydelta://state/augmentation/next");

    // Simulate a crash after publishing the state/intent but before enqueue.
    // Recovery reopens the real journal and uses the ordinary queue seam.
    using json = nlohmann::ordered_json;
    const json intent_state = {{"kind", "flydelta_daemon_orchestration_state"},
        {"phase", "deep_controls"}, {"search_anchor_ref", "resource://anchor"},
        {"search_anchor_pending", true}, {"search_model_call_limit", 256}, {"search_model_calls_used", 40},
        {"anchor_continuation", {
            {"parent_job", json::parse(common_flydelta_experiment_job_to_json(pipeline_claimed.job))},
            {"bootstrap_state_ref", pipeline_claimed.job.bootstrap_zoom_state_ref},
            {"representation_state_ref", pipeline_claimed.job.representation_augmentation_state_ref}}}};
    common_learning_lifecycle_record intent_record;
    intent_record.payload_json = intent_state.dump();
    const auto digest = hash_sha256_hex(intent_record.payload_json.data(), intent_record.payload_json.size());
    intent_record.subject_id = "flydelta://state/orchestration/" + digest.substr(0, 32);
    intent_record.event_id = "intent:" + digest;
    intent_record.idempotency_key = intent_record.event_id;
    intent_record.kind = common_learning_lifecycle_kind::flydelta_experiment;
    intent_record.status = common_learning_lifecycle_status::running;
    intent_record.source_id = "test:anchor-recovery";
    intent_record.namespace_id = pipeline_claimed.job.seed.scope.namespace_id;
    intent_record.project_id = pipeline_claimed.job.seed.scope.project_id;
    intent_record.session_id = pipeline_claimed.job.seed.scope.session_id;
    intent_record.content_hash = "sha256:" + digest;
    intent_record.created_at = "2026-10-27T00:00:00Z";
    const auto journal_path = root / "anchor-lifecycle.jsonl";
    {
        common_learning_jsonl_lifecycle_store journal;
        CHECK(journal.open(journal_path, error));
        CHECK(journal.append(intent_record, error));
    }
    common_learning_jsonl_lifecycle_store reopened;
    CHECK(reopened.open(journal_path, error));
    const auto recovery_root = root / "recovered";
    CHECK(common_flydelta_experiment_queue_enqueue(recovery_root, pipeline_claimed.job, {}, error));
    common_flydelta_claimed_experiment_job abandoned;
    CHECK(common_flydelta_experiment_queue_claim_next(recovery_root, {}, abandoned, error));
    CHECK(abandoned.job.id == pipeline_claimed.job.id);
    CHECK(common_flydelta_recover_anchor_continuations(reopened, recovery_root, {}, error));
    CHECK(common_flydelta_recover_anchor_continuations(reopened, recovery_root, {}, error));
    common_flydelta_claimed_experiment_job recovered;
    CHECK(common_flydelta_experiment_queue_claim_next(recovery_root, {}, recovered, error));
    CHECK(recovered.job.search_state_ref == intent_record.subject_id);
    CHECK(recovered.job.seed.scope.session_id == pipeline_claimed.job.seed.scope.session_id);
    CHECK(recovered.job.teaching_material_group_ref == pipeline_claimed.job.teaching_material_group_ref);
    common_flydelta_claimed_experiment_job duplicate;
    CHECK(common_flydelta_experiment_queue_claim_next(recovery_root, {}, duplicate, error));
    CHECK(duplicate.queue_key.empty());

    // Physical replay reuses logical reservations, but never erases the
    // unresolved consumption of the interrupted attempt.
    common_learning_jsonl_lifecycle_store executions;
    const auto execution_path = root / "execution-lifecycle.jsonl";
    CHECK(executions.open(execution_path, error));
    common_flydelta_search_consumption consumption;
    std::string first_attempt, interrupted_attempt, resumed_attempt;
    CHECK(common_flydelta_begin_search_execution(executions, pipeline_claimed.job, 0, 4,
        {"arm:a", "arm:b"}, "wave:one", 3, first_attempt, consumption, error));
    CHECK(consumption.logical_used == 2 && consumption.physical_rows_started == 2);
    CHECK(common_flydelta_finish_search_execution(executions, pipeline_claimed.job, first_attempt, 2, true, error));
    CHECK(common_flydelta_finish_search_execution(executions, pipeline_claimed.job, first_attempt, 2, true, error));
    CHECK(common_flydelta_begin_search_execution(executions, pipeline_claimed.job, 0, 4,
        {"arm:a", "arm:b"}, "wave:one", 3, interrupted_attempt, consumption, error));
    common_learning_jsonl_lifecycle_store reopened_executions;
    CHECK(reopened_executions.open(execution_path, error));
    CHECK(common_flydelta_load_search_consumption(reopened_executions, pipeline_claimed.job, 0, 4, consumption, error));
    CHECK(consumption.logical_used == 2 && consumption.physical_rows_started == 4);
    CHECK(consumption.physical_rows_completed == 2 && consumption.physical_rows_unresolved == 2);
    CHECK(common_flydelta_begin_search_execution(reopened_executions, pipeline_claimed.job, 0, 4,
        {"arm:a", "arm:b"}, "wave:one", 3, resumed_attempt, consumption, error));
    CHECK(first_attempt != interrupted_attempt && resumed_attempt != interrupted_attempt);
    CHECK(consumption.logical_used == 2 && consumption.physical_rows_started == 6);
    std::string refused;
    CHECK(!common_flydelta_begin_search_execution(reopened_executions, pipeline_claimed.job, 0, 4,
        {"arm:a", "arm:b"}, "wave:one", 3, refused, consumption, error));
    CHECK(!common_flydelta_begin_search_execution(reopened_executions, pipeline_claimed.job, 0, 4,
        {"arm:a", "arm:b", "arm:c", "arm:d", "arm:e"}, "wave:too-large", 3, refused, consumption, error));
    CHECK(!common_flydelta_load_search_consumption(reopened_executions, pipeline_claimed.job, 1, 4, consumption, error));
    CHECK(!common_flydelta_finish_search_execution(reopened_executions, pipeline_claimed.job,
        "flydelta://execution/missing", 1, true, error));
    {
        std::ofstream torn(execution_path, std::ios::binary | std::ios::app);
        torn << "{\"event_id\":";
    }
    common_learning_jsonl_lifecycle_store recovered_tail;
    CHECK(recovered_tail.open(execution_path, error, true));
    CHECK(common_flydelta_load_search_consumption(recovered_tail, pipeline_claimed.job, 0, 4, consumption, error));
    CHECK(consumption.logical_used == 2 && consumption.physical_rows_started == 6);
    CHECK(consumption.physical_rows_completed == 2 && consumption.physical_rows_unresolved == 4);

    // Result persistence precedes acknowledgement. A crash or failed host
    // commit replays the saved commit inputs, never the evaluator/model.
    const auto commit_root = root / "commit-recovery";
    CHECK(common_flydelta_experiment_queue_enqueue(commit_root, pipeline_claimed.job, {}, error));
    common_flydelta_claimed_experiment_job committing;
    CHECK(common_flydelta_experiment_queue_claim_next(commit_root, {}, committing, error));
    common_flydelta_experiment_worker_report saved_report;
    saved_report.job_id = committing.job.id;
    saved_report.completed_job = committing.job;
    saved_report.state = common_flydelta_experiment_queue_state::succeeded;
    saved_report.safe_summary = "saved search result";
    saved_report.has_next_action = true;
    saved_report.next_action = common_flydelta_next_action::run_deep_controls;
    saved_report.search_state_ref = "flydelta://state/search/saved";
    common_learning_jsonl_lifecycle_store commit_store;
    const auto commit_path = root / "commit-lifecycle.jsonl";
    CHECK(commit_store.open(commit_path, error));
    CHECK(common_flydelta_append_worker_commit_intent(commit_store, committing.job, saved_report, error));
    const auto fail_commit = [](const auto &, const auto &, std::string & failure) {
        failure = "injected host persistence failure"; return false;
    };
    CHECK(!common_flydelta_recover_worker_commits(commit_store, commit_root, {}, fail_commit, {}, error));
    CHECK(std::filesystem::exists(commit_root / "running" / committing.queue_key));
    common_learning_jsonl_lifecycle_store restored_commits;
    CHECK(restored_commits.open(commit_path, error));
    size_t persisted = 0, scheduled = 0;
    const auto persist_saved = [&](const auto & job, const auto & report, std::string &) {
        ++persisted; return job.id == committing.job.id && report.search_state_ref == saved_report.search_state_ref;
    };
    const auto schedule_saved = [&](const auto & job, const auto & report, std::string &) {
        ++scheduled; return job.id == committing.job.id && report.next_action == saved_report.next_action;
    };
    CHECK(common_flydelta_recover_worker_commits(restored_commits, commit_root, {}, persist_saved, schedule_saved, error));
    CHECK(common_flydelta_recover_worker_commits(restored_commits, commit_root, {}, persist_saved, schedule_saved, error));
    CHECK(persisted == 1 && scheduled == 1);
    CHECK(std::filesystem::exists(commit_root / "succeeded" / committing.queue_key));
    // Normal completion writes the same idempotent acknowledgement; startup
    // must not replay effects that already completed before the restart.
    const auto normal_root = root / "normal-commit";
    common_learning_in_memory_lifecycle_store normal_commits;
    CHECK(common_flydelta_experiment_queue_enqueue(normal_root, pipeline_claimed.job, {}, error));
    common_flydelta_claimed_experiment_job normal_claimed;
    CHECK(common_flydelta_experiment_queue_claim_next(normal_root, {}, normal_claimed, error));
    CHECK(common_flydelta_append_worker_commit_intent(normal_commits, normal_claimed.job, saved_report, error));
    CHECK(common_flydelta_experiment_queue_complete(normal_root, normal_claimed,
        common_flydelta_experiment_queue_state::succeeded, saved_report.safe_summary, {}, error));
    CHECK(common_flydelta_acknowledge_worker_commit(normal_commits, normal_claimed.job, error));
    CHECK(common_flydelta_acknowledge_worker_commit(normal_commits, normal_claimed.job, error));
    CHECK(common_flydelta_recover_worker_commits(normal_commits, normal_root, {}, persist_saved, schedule_saved, error));
    CHECK(persisted == 1 && scheduled == 1);
    auto long_parent = pipeline_claimed.job;
    long_parent.id.assign(500, 'x');
    CHECK(common_flydelta_collect_next_action_job(root / "long-lineage", {}, long_parent,
        common_flydelta_next_action::run_deep_controls, long_parent.bootstrap_zoom_state_ref,
        intent_record.subject_id, {}, {}, result, error));
    common_learning_in_memory_lifecycle_store wrong_scope;
    auto invalid_intent = intent_record;
    invalid_intent.session_id = "other-session";
    CHECK(wrong_scope.append(invalid_intent, error));
    CHECK(!common_flydelta_recover_anchor_continuations(wrong_scope, root / "invalid-recovery", {}, error));

    // Terminal cancellation wins over an older continuation intent. A failed
    // parent requires reconciliation and must not leave a new pending job.
    for (const auto terminal : {common_flydelta_experiment_queue_state::cancelled,
            common_flydelta_experiment_queue_state::failed}) {
        const auto terminal_root = root / common_flydelta_experiment_queue_state_name(terminal);
        CHECK(common_flydelta_experiment_queue_enqueue(terminal_root, pipeline_claimed.job, {}, error));
        common_flydelta_claimed_experiment_job terminal_claimed;
        CHECK(common_flydelta_experiment_queue_claim_next(terminal_root, {}, terminal_claimed, error));
        CHECK(common_flydelta_experiment_queue_complete(terminal_root, terminal_claimed,
            terminal, "terminal parent", {}, error));
        const bool recovered_terminal = common_flydelta_recover_anchor_continuations(
            reopened, terminal_root, {}, error);
        CHECK(recovered_terminal == (terminal == common_flydelta_experiment_queue_state::cancelled));
        common_flydelta_claimed_experiment_job unexpected;
        CHECK(common_flydelta_experiment_queue_claim_next(terminal_root, {}, unexpected, error));
        CHECK(unexpected.queue_key.empty());
    }
    // Simulate a crash after the terminal record is durable but before the
    // running directory is moved to the terminal queue.
    const auto interrupted_root = root / "interrupted-cancellation";
    CHECK(common_flydelta_experiment_queue_enqueue(interrupted_root, pipeline_claimed.job, {}, error));
    common_flydelta_claimed_experiment_job interrupted;
    CHECK(common_flydelta_experiment_queue_claim_next(interrupted_root, {}, interrupted, error));
    common_learning_in_memory_lifecycle_store interrupted_commits;
    CHECK(common_flydelta_append_worker_commit_intent(interrupted_commits, interrupted.job, saved_report, error));
    {
        std::ofstream state(interrupted_root / "running" / interrupted.queue_key / "state.json");
        state << json{{"job_id", interrupted.job.id}, {"state", "cancelled"},
            {"safe_summary", "cancelled"}}.dump();
        CHECK(state.good());
    }
    CHECK(common_flydelta_recover_worker_commits(interrupted_commits, interrupted_root, {},
        persist_saved, schedule_saved, error));
    CHECK(common_flydelta_recover_anchor_continuations(reopened, interrupted_root, {}, error));
    CHECK(persisted == 1 && scheduled == 1);
    common_flydelta_claimed_experiment_job unexpected_interrupted;
    CHECK(common_flydelta_experiment_queue_claim_next(interrupted_root, {}, unexpected_interrupted, error));
    CHECK(unexpected_interrupted.queue_key.empty());

    value.enabled = false;
    CHECK(common_flydelta_collect_experiment_job(root, {}, value, result, error));
    CHECK(result == common_flydelta_experiment_collection_result::disabled);

    value.enabled = true;
    value.evidence.host_verified = false;
    CHECK(!common_flydelta_collect_experiment_job(root, {}, value, result, error));

    value = request();
    value.behavior_delta_ids = {"flydelta://delta/invalid-for-counterfactual"};
    CHECK(!common_flydelta_collect_experiment_job(root, {}, value, result, error));

    value = request();
    value.behavior_key = "tool_use/diagnostics/wrong-tool";
    CHECK(!common_flydelta_collect_experiment_job(root, {}, value, result, error));

    auto refine_request = request();
    common_flydelta_search_observation observation;
    observation.experiment_id = "flydelta://experiment/collector-refine";
    observation.candidate_id = "flydelta://candidate/refine";
    observation.outcome = common_flydelta_counterfactual_outcome::unknown;
    observation.host_evaluated = true;
    observation.verifier_known = true;
    observation.diagnostics_available = true;
    observation.diagnostics = {1, 2, 0.5f, 0.2f, 0.1f};
    observation.budget_remaining = true;
    common_flydelta_search_decision decision;
    CHECK(common_flydelta_decide_search_disposition(observation, decision, error));
    common_flydelta_candidate_lineage lineage;
    lineage.candidate_id = observation.candidate_id;
    lineage.mutation_kind = "scale_refinement";
    lineage.generation = 2;
    lineage.direction_id = "flydelta://direction/raw";
    lineage.layer_indices = {2};
    lineage.scale = 0.08f;
    lineage.intervention_budget = 0.08f;
    const auto refine_root = root / "refine";
    CHECK(common_flydelta_collect_refinement_job(
        refine_root, {}, refine_request, observation, decision, lineage, result, error));
    CHECK(result == common_flydelta_experiment_collection_result::enqueued);
    CHECK(common_flydelta_collect_refinement_job(
        refine_root, {}, refine_request, observation, decision, lineage, result, error));
    CHECK(result == common_flydelta_experiment_collection_result::already_present);
    common_flydelta_claimed_experiment_job refined;
    CHECK(common_flydelta_experiment_queue_claim_next(refine_root, {}, refined, error));
    CHECK(refined.job.kind == common_flydelta_experiment_job_kind::counterfactual);
    CHECK(refined.job.id.find("/generation/2") != std::string::npos);

    auto pipeline_refine_request = request();
    pipeline_refine_request.kind = common_flydelta_experiment_job_kind::search_pipeline;
    pipeline_refine_request.behavior_delta_ids = {"flydelta://delta/pipeline-refine"};
    pipeline_refine_request.bootstrap_zoom_state_ref = "flydelta://state/bootstrap-zoom/resume";
    pipeline_refine_request.search_state_ref = "flydelta://state/search/resume";
    const auto pipeline_refine_root = root / "pipeline-refine";
    CHECK(common_flydelta_collect_search_pipeline_refinement_job(
        pipeline_refine_root, {}, pipeline_refine_request, observation, decision, lineage,
        result, error));
    CHECK(result == common_flydelta_experiment_collection_result::enqueued);
    common_flydelta_claimed_experiment_job pipeline_refined;
    CHECK(common_flydelta_experiment_queue_claim_next(
        pipeline_refine_root, {}, pipeline_refined, error));
    CHECK(pipeline_refined.job.kind == common_flydelta_experiment_job_kind::search_pipeline);
    CHECK(pipeline_refined.job.behavior_delta_ids.size() == 1);
    CHECK(pipeline_refined.job.bootstrap_zoom_state_ref ==
        "flydelta://state/bootstrap-zoom/resume");
    CHECK(pipeline_refined.job.search_state_ref ==
        "flydelta://state/search/resume");

    std::filesystem::remove_all(root, ignored);
    return 0;
}
