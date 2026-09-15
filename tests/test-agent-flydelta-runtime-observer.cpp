#include "agent/adaptation/flydelta/flydelta-runtime-observer.h"

#include <string>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

static common_learning_transaction transaction() {
    common_learning_transaction value;
    value.id = "learning://observation/turn-1";
    value.created_at = "2026-09-15T00:00:00Z";
    value.observation.id = value.id;
    value.observation.scope.namespace_id = "local";
    value.observation.scope.project_id = "project-1";
    value.observation.scope.session_id = "session-1";
    value.observation.source_turn_id = "turn-1";
    value.observation.source_plan_id = "plan-1";
    value.observation.signals.push_back({common_learning_signal_type::tool_failure,
        "plan-1", "step-1", "data.describe", "failure-1", "wrong tool"});
    value.observation.signals.push_back({common_learning_signal_type::successful_recovery,
        "plan-1", "step-1", "data.inspect", "recovery-1", "repaired tool"});
    value.observation.evidence_ids = {"failure-1", "recovery-1"};
    value.observation.cause = common_learning_cause::model_behavior;
    value.observation.verification = common_learning_verification::host_verified;
    value.observation.idempotency_key = "turn-1:repair";
    value.observation.content_hash = "identity:fnv1a64:runtime-observer";
    value.observation.collection_allowed = true;
    return value;
}

int main() {
    std::string error;
    common_flydelta_capture_candidate_collector collector(
        "sha256:model", "capture:v1", 4);
    common_learning_in_memory_lifecycle_store lifecycle;
    common_flydelta_runtime_candidate_observer observer(collector, &lifecycle);

    common_adaptation_evidence_source_match match;
    match.source = common_adaptation_evidence_source::tool_repair;
    match.behavior_key = "tool_use/diagnostics/wrong-tool";
    match.evidence_refs = {"failure-1", "recovery-1"};
    match.candidate_ready = true;
    auto tx = transaction();
    CHECK(observer.observe(match, tx, error));
    CHECK(observer.observe(match, tx, error));
    CHECK(collector.candidates().size() == 1);
    auto records = lifecycle.list(error);
    CHECK(error.empty() && records.size() == 1);
    CHECK(records.front().kind == common_learning_lifecycle_kind::candidate);
    CHECK(records.front().status == common_learning_lifecycle_status::observed);
    CHECK(records.front().payload_json.find("capture_candidate_discovered") != std::string::npos);
    CHECK(records.front().payload_json.find("wrong-tool") != std::string::npos);

    common_adaptation_evidence_source_match ordinary;
    ordinary.source = common_adaptation_evidence_source::tool_repair;
    ordinary.candidate_ready = false;
    CHECK(observer.observe(ordinary, tx, error));
    CHECK(collector.candidates().size() == 1);
    CHECK(lifecycle.list(error).size() == 1);

    common_learning_in_memory_lifecycle_store second_journal;
    common_flydelta_runtime_candidate_observer second_observer(collector, &second_journal);
    CHECK(second_observer.observe(match, tx, error));
    CHECK(second_journal.list(error).size() == 1);
    return 0;
}
